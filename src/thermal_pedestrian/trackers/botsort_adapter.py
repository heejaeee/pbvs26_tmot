from __future__ import annotations

from typing import (Any, Union, List)
import sys
import warnings
from typing import Optional

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import platform

from munch import Munch
from torch import Tensor
import torch

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.gmo import General_Moving_Object
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.core.utils.image import to_channel_first
from thermal_pedestrian.trackers import BaseTracker

__all__ = [
	"BOTSORT_Adapter"
]

from ultralytics.trackers import BOTSORT


# MARK: - BOTSORT_Adapter

@TRACKERS.register(name="botsort")
class BOTSORT_Adapter(BaseTracker):
	"""BOTSORT

	Attributes:
		Same as ``Tracker``
	"""
	# MARK: Magic Functions

	def __init__(self, bytetrack_config: dict, **kwargs):
		super().__init__(**kwargs)
		self.bytetrack_config = bytetrack_config
		self.init_model()

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		self.model = BOTSORT(self.bytetrack_config, frame_rate=30)

	# MARK: Update

	def update(self, detections: List[Instance], image: Any, *args, **kwargs):
		"""Update ``self.tracks`` with new detections.

		Args:
			detections (list):
				The list of newly ``Instance`` objects.

		Requires:
			This method must be called once for each frame even with empty detections, just call update with empty list [].

		Returns:

		"""
		img_h, img_w, c = image.shape

		# dets: Nx6 of (x1, y1, x2, y2, score)
		dets = Munch()
		dets.xywh  = []
		dets.conf  = []
		dets.cls   = []
		for det in detections:
			dets.xywh.append(convert_voc_to_yolo((img_w, img_h), det.bbox))
			dets.conf.append(det.confidence)
			dets.cls.append(det.class_label['id'])

		dets.xywh = np.array(dets.xywh)
		dets.conf = np.array(dets.conf)
		dets.cls  = np.array(dets.cls)

		# coords.tolist() + [self.track_id, self.score, self.cls, self.idx]
		results   = self.model.update(dets)

		if len(results) == 0:
			self.tracks = []
			return

		bboxes       = results[:,: 4].astype(float)
		track_ids    = results[:, 4].astype(int)
		confs        = results[:, 5].astype(int)
		clsss        = results[:, 6].astype(int)
		indexes_bbox = results[:, 7].astype(int)

		self.tracks = []
		for track_id, idx in zip(track_ids, indexes_bbox):
			gmo  = General_Moving_Object.gmo_from_detection(detections[idx])
			gmo.id = int(track_id)
			self.tracks.append(gmo)


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
		pass

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
		pass

	def delete_dead_tracks(
			self
	):
		"""Delete dead tracks.
		"""
		pass

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
		pass

	def clear_model_memory(self):
		"""Free the memory of model

		Returns:
			None
		"""
		if self.model is not None:
			del self.model
			torch.cuda.empty_cache()


def convert_voc_to_yolo(size, bbox):
	dw = 1. / (float(size[0]))
	dh = 1. / (float(size[1]))
	x = (float(bbox[0]) + float(bbox[2])) / 2.0 - 1
	y = (float(bbox[1]) + float(bbox[3])) / 2.0 - 1
	w = abs(float(bbox[2]) - float(bbox[0]))
	h = abs(float(bbox[3]) - float(bbox[1]))
	x = x * dw
	w = w * dw
	y = y * dh
	h = h * dh
	return x, y, w, h
