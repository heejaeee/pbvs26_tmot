from __future__ import annotations

from typing import Any, List, Optional
import os

import numpy as np
import torch

from ultralytics import SAM

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.core.utils.bbox import batch_bbox_iou
from thermal_pedestrian.core.utils.rich import console
from thermal_pedestrian.trackers import BaseTracker
from thermal_pedestrian.trackers.sort.sort_kalman_bbox import KalmanBBoxTrack
from thermal_pedestrian.trackers.sort_adapter import linear_assignment

__all__ = [
	"SAM3_Adapter"
]


@TRACKERS.register(name="sam3")
class SAM3_Adapter(BaseTracker):
	"""SAM3 tracking adapter.

	This adapter refines YOLO detections using SAM3 masks, then tracks using
	Kalman + Hungarian assignment.
	"""

	def __init__(self, sam3_config: Optional[dict] = None, **kwargs):
		super().__init__(**kwargs)
		self.sam3_config = sam3_config or {}
		self.sam_model = None
		self.sam_device = None
		self.sam_imgsz = None
		self.mask_threshold = 0.0

		self.track_index = 0
		self.tracks: List[KalmanBBoxTrack] = []
		self.init_model()

	def init_model(self):
		weights = self.sam3_config.get("weights")
		if isinstance(weights, (list, tuple)):
			weights = weights[0] if weights else None

		if not weights:
			console.log("SAM3: no weights provided, using raw detections.")
			return

		if not os.path.exists(weights):
			raise FileNotFoundError(f"SAM3 weights not found: {weights}")

		self.sam_model = SAM(weights)
		self.sam_device = self.sam3_config.get("device")
		self.sam_imgsz = self.sam3_config.get("imgsz")
		self.mask_threshold = float(self.sam3_config.get("mask_threshold", 0.0))

	def update(self, detections: List[Instance], image: Any, *args, **kwargs):
		"""Refine detections with SAM3 and track with Kalman + Hungarian."""
		self.frame_count += 1
		detections = detections or []

		if detections and self.sam_model is not None and image is not None:
			self._refine_detections_with_sam(detections, image)

		self._update_tracks(detections)

	def _refine_detections_with_sam(self, detections: List[Instance], image: Any):
		bboxes = np.array([det.bbox for det in detections], dtype=np.float32)
		if bboxes.size == 0:
			return

		sam_kwargs = {
			"bboxes": bboxes,
			"verbose": False,
			"save": False,
		}
		if self.sam_device is not None:
			sam_kwargs["device"] = self.sam_device
		if self.sam_imgsz is not None:
			sam_kwargs["imgsz"] = self.sam_imgsz

		results = self.sam_model(image, **sam_kwargs)
		if not results:
			return

		masks = results[0].masks
		if masks is None or masks.data is None:
			return

		mask_data = masks.data
		if hasattr(mask_data, "detach"):
			mask_data = mask_data.detach()
		mask_np = mask_data.cpu().numpy()

		img_h, img_w = image.shape[:2]
		count = min(len(detections), mask_np.shape[0])
		for idx in range(count):
			mask = mask_np[idx] > self.mask_threshold
			if not mask.any():
				continue

			ys, xs = np.where(mask)
			x_min = max(int(xs.min()), 0)
			y_min = max(int(ys.min()), 0)
			x_max = min(int(xs.max()), img_w - 1)
			y_max = min(int(ys.max()), img_h - 1)
			if x_max <= x_min or y_max <= y_min:
				continue

			detections[idx].bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

	def _update_tracks(self, detections: List[Instance]):
		if len(detections) > 0:
			dets = np.array([np.append(np.float64(d.bbox), np.float64(d.confidence)) for d in detections])
		else:
			dets = np.empty((0, 5))

		trks = np.zeros((len(self.tracks), 5))
		to_del = []
		for t, trk in enumerate(trks):
			pos = self.tracks[t].predict_motion_state()[0]
			trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
			if np.any(np.isnan(pos)):
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
		for t in reversed(to_del):
			self.tracks.pop(t)

		matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(dets, trks)

		for m in matched:
			track_idx = m[1]
			detection_idx = m[0]
			self.tracks[track_idx].update_gmo(detections[detection_idx])

		for i in unmatched_dets:
			new_trk = KalmanBBoxTrack.track_from_detection(detections[i])
			self.track_index += 1
			new_trk.id = self.track_index
			self.tracks.append(new_trk)

		i = len(self.tracks)
		for trk in reversed(self.tracks):
			i -= 1
			if trk.time_since_update > self.max_age:
				self.tracks.pop(i)

	def _associate_detections_to_tracks(self, dets: np.ndarray, trks: np.ndarray):
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
		for d in range(len(dets)):
			if d not in matched_indices[:, 0]:
				unmatched_detections.append(d)

		unmatched_trackers = []
		for t in range(len(trks)):
			if t not in matched_indices[:, 1]:
				unmatched_trackers.append(t)

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

	def clear_model_memory(self):
		if self.sam_model is not None:
			del self.sam_model
			self.sam_model = None
			torch.cuda.empty_cache()
		self.tracks = []
