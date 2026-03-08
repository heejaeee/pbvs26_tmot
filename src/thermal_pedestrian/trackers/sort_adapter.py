from __future__ import annotations

import sys
import warnings
from typing import Optional, Union, List
import numpy as np

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform


from torch import Tensor
import torch

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.trackers import BaseTracker
from thermal_pedestrian.trackers.sort.sort_kalman_bbox import KalmanBBoxTrack
from thermal_pedestrian.core.utils.bbox import batch_bbox_iou
from thermal_pedestrian.core.utils.point import distance_between_points

__all__ = [
	"SORT"
]


# MARK: - SORT

@TRACKERS.register(name="sort")
class SORT(BaseTracker):
	"""SORT (Simple Online Realtime Tracker)

	Attributes:
		Same as ``Tracker``
	"""
	# MARK: Magic Functions

	def __init__(self,
	             name: str = "sort",
	             id_switch_postprocess: bool = False,
	             id_switch_max_distance: float = 50.0,
	             id_switch_max_age: int = 5,
	             id_switch_label_check: bool = True,
	             *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)
		self.tracks: List[KalmanBBoxTrack] = []
		self.track_index                   = 0
		self.id_switch_postprocess         = id_switch_postprocess
		self.id_switch_max_distance        = id_switch_max_distance
		self.id_switch_max_age             = id_switch_max_age
		self.id_switch_label_check         = id_switch_label_check

	# MARK: Update

	def update(self, detections: List[Instance], *args, **kwargs):
		"""Update ``self.tracks`` with new detections.

		Args:
			detections (list):
				The list of newly ``Instance`` objects.

		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty list [].

		Returns:

		"""
		self.frame_count += 1  # Should be the same with VideoReader.frame_idx

		# NOTE: Extract and convert bbox from detections for easier use.
		if len(detections) > 0:
			# dets - a numpy array of detections in the format [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],...]
			dets = np.array([np.append(np.float64(d.bbox), np.float64(d.confidence)) for d in detections])
		else:
			dets = np.empty((0, 5))

		# NOTE: Get predicted locations from existing trackers.
		trks   = np.zeros((len(self.tracks), 5))
		to_del = []
		for t, trk in enumerate(trks):
			pos    = self.tracks[t].predict_motion_state()[0]
			trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
			if np.any(np.isnan(pos)):
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

		# NOTE: Find 3 lists of matches, unmatched_detections and unmatched_trackers
		for t in reversed(to_del):
			self.tracks.pop(t)
		matched, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(dets, trks)

		# NOTE: Update matched trackers with assigned detections
		self.update_matched_tracks(matched=matched, detections=detections)

		# NOTE: Create and initialise new trackers for unmatched detections
		new_tracks = self.create_new_tracks(unmatched_dets=unmatched_dets, detections=detections)

		# NOTE: Post-process potential ID switches.
		self.postprocess_id_switch(new_tracks=new_tracks)

		# NOTE: Remove dead tracklets
		self.delete_dead_tracks()

	def update_matched_tracks(
			self,
			matched   : Union[List, np.ndarray],
			detections: List[Instance]
	):
		"""Update the track that has been matched with new detection

		Args:
			matched (list or np.ndarray):
				Matching between self.tracks index and detection index.
			detections (any):
				The newly detections.
		"""
		for m in matched:
			track_idx     = m[1]
			detection_idx = m[0]
			# HERE, we call ``General_Moving_Object.update_gmo()``. This contains all necessary functions to update the whole General_Moving_Object object.
			self.tracks[track_idx].update_gmo(detections[detection_idx])

		# IF you don't call the function above, then call the following functions:
		# self.tracks[track_idx].update_go_from_detection(detection=detections[detection_idx])
		# self.tracks[track_idx].update_motion_state()

	def create_new_tracks(
			self,
			unmatched_dets: Union[List, np.ndarray],
			detections    : List[Instance]
	):
		"""Create new tracks.

		Args:
			unmatched_dets (list or np.ndarray):
				Index of the newly detection in ``detections`` that has not matched with any tracks.
			detections (any):
				The newly detections.
		"""
		new_tracks = []
		for i in unmatched_dets:
			new_trk           = KalmanBBoxTrack.track_from_detection(detections[i])
			self.track_index += 1
			new_trk.id        = self.track_index
			self.tracks.append(new_trk)
			new_tracks.append(new_trk)
		return new_tracks

	def postprocess_id_switch(self, new_tracks: List[KalmanBBoxTrack]):
		if (not self.id_switch_postprocess or not new_tracks or
			self.id_switch_max_distance <= 0 or self.id_switch_max_age <= 0):
			return

		lost_tracks = [
			trk for trk in self.tracks
			if trk not in new_tracks
			and trk.time_since_update > 0
			and trk.time_since_update <= self.id_switch_max_age
		]
		if not lost_tracks:
			return

		candidates = []
		for new_trk in new_tracks:
			new_center = new_trk.current_bbox_center
			for lost_trk in lost_tracks:
				if self.id_switch_label_check:
					lost_label = lost_trk.labels[-1] if getattr(lost_trk, "labels", None) else None
					new_label  = new_trk.labels[-1] if getattr(new_trk, "labels", None) else None
					if lost_label is not None and new_label is not None:
						try:
							if lost_label["id"] != new_label["id"]:
								continue
						except Exception:
							if lost_label != new_label:
								continue

				dist = distance_between_points(lost_trk.current_bbox_center, new_center)
				if dist <= self.id_switch_max_distance:
					candidates.append((dist, lost_trk, new_trk))

		if not candidates:
			return

		candidates.sort(key=lambda item: item[0])
		used_lost = set()
		used_new  = set()
		for _, lost_trk, new_trk in candidates:
			if id(lost_trk) in used_lost or id(new_trk) in used_new:
				continue
			self._merge_tracklets(lost_trk=lost_trk, new_trk=new_trk)
			used_lost.add(id(lost_trk))
			used_new.add(id(new_trk))

	def _merge_tracklets(self, lost_trk: KalmanBBoxTrack, new_trk: KalmanBBoxTrack):
		instance = self._instance_from_track(new_trk)
		if instance is not None:
			lost_trk.update_gmo(instance)
		if new_trk in self.tracks:
			self.tracks.remove(new_trk)

	def _instance_from_track(self, trk: KalmanBBoxTrack) -> Optional[Instance]:
		if trk is None or not getattr(trk, "bboxes", None):
			return None

		bbox = trk.bboxes[-1]
		label = trk.labels[-1] if getattr(trk, "labels", None) else None
		polygon = trk.polygons[-1] if getattr(trk, "polygons", None) else None
		bbox_id = trk.bboxes_id[-1] if getattr(trk, "bboxes_id", None) else trk.id
		frame_index = trk.last_frame_index if getattr(trk, "frame_indexes", None) else None
		timestamp = trk.last_timestamp if getattr(trk, "timestamps", None) else None

		return Instance(
			id=bbox_id,
			roi_uuid=None,
			bbox=bbox,
			polygon=polygon,
			confidence=trk.confidence,
			class_label=label,
			frame_index=frame_index,
			timestamp=timestamp
		)

	def delete_dead_tracks(self):
		"""Delete dead tracks.
		"""
		i = len(self.tracks)
		for trk in reversed(self.tracks):
			d = trk.current_motion_state()[0]  # Get the current bounding box of Kalman Filter
			#if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
			# ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
			i -= 1
			# NOTE: Remove dead tracklets
			if trk.time_since_update > self.max_age:
				self.tracks.pop(i)

	def associate_detections_to_tracks(
			self,
			dets: np.ndarray,
			trks: np.ndarray,
			**kwargs
	):
		"""Assigns detections to ``self.tracks``

		Args:
			dets (np.ndarray):
				The list of newly ``Instance`` objects.
			trks (np.ndarray):

		Returns:
			3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		if len(trks) == 0:
			return np.empty((0, 2), dtype=int), np.arange(len(dets)), np.empty((0, 5), dtype=int)

		iou_matrix = batch_bbox_iou(dets, trks)

		if min(iou_matrix.shape) > 0:
			a = (iou_matrix > self.iou_threshold).astype(np.int32)
			if a.sum(1).max() == 1 and a.sum(0).max() == 1:
				matched_indices = np.stack(np.where(a), axis=1)
			else:
				matched_indices = linear_assignment(-iou_matrix)
		else:
			matched_indices = np.empty(shape=(0, 2))

		unmatched_detections = []
		for d, det in enumerate(dets):
			if d not in matched_indices[:, 0]:
				unmatched_detections.append(d)

		unmatched_trackers = []
		for t, trk in enumerate(trks):
			if t not in matched_indices[:, 1]:
				unmatched_trackers.append(t)

		# filter out matched with low IOU
		matches = []
		for m in matched_indices:
			if iou_matrix[m[0], m[1]] < self.iou_threshold:
				unmatched_detections.append(m[0])
				unmatched_trackers.append(m[1])
			else:
				matches.append(m.reshape(1, 2))

		if len(matches) == 0:
			matches = np.empty((0, 2), dtype=int)
		else:
			matches = np.concatenate(matches, axis=0)

		return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# MARK: - Utils

def linear_assignment(cost_matrix):
	"""

	Args:
		cost_matrix (np.array):

	Returns:
		object (np.array):

	"""
	try:
		import lap
		_, x, y = lap.lapjv(cost_matrix, extend_cost=True)
		return np.array([[y[i], i] for i in x if i >= 0])  #
	except ImportError:
		from scipy.optimize import linear_sum_assignment
		x, y = linear_sum_assignment(cost_matrix)
		return np.array(list(zip(x, y)))
