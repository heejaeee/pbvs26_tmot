from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional
import sys

import numpy as np

from thermal_pedestrian.core.factory.builder import TRACKERS
from thermal_pedestrian.core.objects.gmo import General_Moving_Object
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.trackers import BaseTracker

_ROOT_DIR = Path(__file__).resolve().parents[3]
_OCSORT_TRACKERS_DIR = _ROOT_DIR / "third_party" / "OC_SORT" / "trackers"
if _OCSORT_TRACKERS_DIR.exists():
	ocsort_path = str(_OCSORT_TRACKERS_DIR)
	if ocsort_path not in sys.path:
		sys.path.insert(0, ocsort_path)

from ocsort_tracker.ocsort import OCSort

__all__ = [
	"OCSORT_Adapter"
]


@TRACKERS.register(name="ocsort")
class OCSORT_Adapter(BaseTracker):
	"""OC_SORT tracking adapter."""

	def __init__(self, ocsort_config: Optional[dict] = None, **kwargs):
		super().__init__(**kwargs)
		self.ocsort_config = ocsort_config or {}
		self.model = None
		self._last_class_label = None
		self.init_model()

	def init_model(self):
		det_thresh = float(self.ocsort_config.get("det_thresh", 0.5))
		delta_t = int(self.ocsort_config.get("delta_t", 3))
		asso_func = self.ocsort_config.get("asso_func", "iou")
		inertia = float(self.ocsort_config.get("inertia", 0.2))
		use_byte = bool(self.ocsort_config.get("use_byte", False))

		self.model = OCSort(
			det_thresh,
			max_age=self.max_age,
			min_hits=self.min_hits,
			iou_threshold=self.iou_threshold,
			delta_t=delta_t,
			asso_func=asso_func,
			inertia=inertia,
			use_byte=use_byte,
		)

	def update(self, detections: List[Instance], image: Any, *args, **kwargs):
		if image is None:
			self.tracks = []
			return

		img_h, img_w, _ = image.shape

		if detections:
			dets = np.array(
				[[float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3]), float(d.confidence)]
				 for d in detections],
				dtype=np.float32
			)
		else:
			dets = np.empty((0, 5), dtype=np.float32)

		results = self.model.update(dets, (img_h, img_w), (img_h, img_w))
		self.tracks = []
		if results is None or len(results) == 0:
			return

		if detections:
			self._last_class_label = detections[0].class_label
		class_label = self._last_class_label
		frame_index = detections[0].frame_index if detections else None
		video_name = detections[0].video_name if detections else None
		if class_label is None:
			return

		for row in results:
			bbox = np.array(row[:4], dtype=np.float32)
			track_id = int(row[4])
			instance = Instance(
				frame_index=frame_index,
				bbox=bbox,
				confidence=1.0,
				class_label=class_label,
				video_name=video_name,
			)
			gmo = General_Moving_Object.gmo_from_detection(instance)
			gmo.id = track_id
			self.tracks.append(gmo)

	def clear_model_memory(self):
		self.model = None
