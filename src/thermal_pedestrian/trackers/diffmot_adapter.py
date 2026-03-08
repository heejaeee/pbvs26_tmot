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

from torch import Tensor
import torch

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.gmo import General_Moving_Object
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.trackers import BaseTracker

from DiffMOT.tracker.DiffMOTtracker import diffmottracker
from DiffMOT.models.autoencoder import D2MP
from DiffMOT.models.condition_embedding import History_motion_embedding


__all__ = [
	"DiffMOT_Adapter"
]


# MARK: - BoostTrack

@TRACKERS.register(name="diffmot")
class DiffMOT_Adapter(BaseTracker):
	"""DiffMOT

	Attributes:
		Same as ``Tracker``
	"""
	# MARK: Magic Functions

	def __init__(self, diffmot_config: dict, **kwargs):
		super().__init__(**kwargs)
		self.diffmot_config = diffmot_config
		self.init_model()

	# MARK: Configure

	def init_model(self):
		"""Create and load model from weights."""
		self.encoder = History_motion_embedding()

		self.model   = D2MP(self.diffmot_config, encoder=self.encoder)
		self.model   = self.model.cuda()
		self.model   = self.model.eval()

		self.tracker = diffmottracker(self.diffmot_config)

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

		# dets: Nx6 of (ID, x, y, w, h, score)
		dets = []
		for det in detections:
			dets.append([
				float(det.bbox[0]),
				float(det.bbox[1]),
				float(det.bbox[2]) - float(det.bbox[0]),
				float(det.bbox[3]) - float(det.bbox[1]),
				det.confidence])
		dets = np.array(dets)

		tag = f"{detections[0].video_name}:{detections[0].frame_index}"

		online_targets  = self.tracker.update(dets, self.model, detections[0].frame_index, img_w, img_h, tag, image)
		self.tracks = []
		for t in online_targets:
			tlwh = t.tlwh
			gmo  = General_Moving_Object.gmo_from_detection(Instance(
				frame_index = detections[0].frame_index,
				bbox        = np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]),
				confidence  = t.score,
				class_label = detections[0].class_label
			))
			gmo.id = int(t.track_id)

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

		if self.tracker is not None:
			self.tracker.dump_cache()
			del self.tracker
			torch.cuda.empty_cache()