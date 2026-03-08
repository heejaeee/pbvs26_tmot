from __future__ import annotations

from typing import Any, Dict, List, Optional

from thermal_pedestrian.core.utils.point import distance_between_points


class IDSwitchPostprocessor:
	"""Post-process track IDs to reduce short ID switches.

	This keeps a short memory of recently lost tracks and merges newborn
	tracklets when their centers are close in space and time.
	"""

	def __init__(
		self,
		max_distance: float = 50.0,
		max_age: int = 5,
		label_check: bool = True,
	):
		self.max_distance = float(max_distance)
		self.max_age = int(max_age)
		self.label_check = bool(label_check)
		self.enabled = (self.max_distance > 0 and self.max_age > 0)
		self.reset()

	def reset(self):
		self.frame_count = 0
		self.id_remap: Dict[Any, Any] = {}
		self.raw_last_seen: Dict[Any, int] = {}
		self.output_last_seen: Dict[Any, int] = {}
		self.output_last_center: Dict[Any, Any] = {}
		self.output_last_label: Dict[Any, Any] = {}

	def apply(self, tracks: List[Any], frame_index: Optional[int] = None):
		if not self.enabled:
			return tracks

		if frame_index is None:
			self.frame_count += 1
			frame_index = self.frame_count
		else:
			self.frame_count = max(self.frame_count, int(frame_index))

		self._prune(frame_index)

		reserved_output_ids = set()
		for trk in tracks:
			raw_id = self._normalize_id(getattr(trk, "id", None))
			if raw_id is None:
				continue
			output_id = getattr(trk, "output_id", None)
			if output_id is None:
				output_id = self.id_remap.get(raw_id, raw_id)
			reserved_output_ids.add(output_id)

		current = []
		for trk in tracks:
			if hasattr(trk, "time_since_update") and trk.time_since_update > 0:
				continue
			raw_id = self._normalize_id(getattr(trk, "id", None))
			center = getattr(trk, "current_bbox_center", None)
			if raw_id is None or center is None:
				continue
			label = self._extract_label(trk)
			current.append({
				"trk": trk,
				"raw_id": raw_id,
				"center": center,
				"label": label,
			})

		if not current:
			return tracks

		current_raw_ids = {item["raw_id"] for item in current}
		for raw_id in current_raw_ids:
			self.raw_last_seen[raw_id] = frame_index

		for item in current:
			item["output_id"] = self.id_remap.get(item["raw_id"], item["raw_id"])

		current_output_ids = {item["output_id"] for item in current}
		lost = self._collect_lost_tracks(current_output_ids, frame_index, reserved_output_ids)

		candidates = []
		for item in current:
			if item["raw_id"] in self.id_remap:
				continue
			for lost_item in lost:
				if self.label_check and not self._labels_match(item["label"], lost_item["label"]):
					continue
				dist = distance_between_points(lost_item["center"], item["center"])
				if dist <= self.max_distance:
					candidates.append((dist, lost_item["output_id"], item))

		candidates.sort(key=lambda x: x[0])
		used_lost = set()
		used_new = set()
		for _, lost_output_id, item in candidates:
			if lost_output_id in used_lost or item["raw_id"] in used_new:
				continue
			self.id_remap[item["raw_id"]] = lost_output_id
			item["output_id"] = lost_output_id
			used_lost.add(lost_output_id)
			used_new.add(item["raw_id"])

		used_output_ids = set()
		reserved_output_ids = {oid for oid in reserved_output_ids if oid is not None}
		for item in current:
			output_id = item["output_id"]
			if output_id in used_output_ids:
				candidate = item["raw_id"]
				if candidate in used_output_ids or candidate in reserved_output_ids:
					candidate = self._next_available_id(used_output_ids | reserved_output_ids)
				item["output_id"] = candidate
			used_output_ids.add(item["output_id"])

		for item in current:
			self.id_remap[item["raw_id"]] = item["output_id"]
			item["trk"].output_id = item["output_id"]
			self.output_last_seen[item["output_id"]] = frame_index
			self.output_last_center[item["output_id"]] = item["center"]
			self.output_last_label[item["output_id"]] = item["label"]

		return tracks

	def _collect_lost_tracks(self, current_output_ids: set, frame_index: int, reserved_output_ids: Optional[set] = None):
		lost = []
		for output_id, last_seen in self.output_last_seen.items():
			if output_id in current_output_ids:
				continue
			if reserved_output_ids and output_id in reserved_output_ids:
				continue
			if frame_index - last_seen > self.max_age:
				continue
			center = self.output_last_center.get(output_id)
			if center is None:
				continue
			lost.append({
				"output_id": output_id,
				"center": center,
				"label": self.output_last_label.get(output_id),
			})
		return lost

	def _prune(self, frame_index: int):
		stale_raw = [
			raw_id for raw_id, last_seen in self.raw_last_seen.items()
			if frame_index - last_seen > self.max_age
		]
		for raw_id in stale_raw:
			self.raw_last_seen.pop(raw_id, None)
			self.id_remap.pop(raw_id, None)

		stale_output = [
			output_id for output_id, last_seen in self.output_last_seen.items()
			if frame_index - last_seen > self.max_age
		]
		for output_id in stale_output:
			self.output_last_seen.pop(output_id, None)
			self.output_last_center.pop(output_id, None)
			self.output_last_label.pop(output_id, None)

	def _normalize_id(self, raw_id: Any):
		if isinstance(raw_id, list) and raw_id:
			return raw_id[-1]
		return raw_id

	def _extract_label(self, trk: Any):
		if hasattr(trk, "labels") and trk.labels:
			return trk.labels[-1]
		return None

	def _labels_match(self, label_a: Any, label_b: Any) -> bool:
		if label_a is None or label_b is None:
			return True
		key_a = label_a.get("id") if isinstance(label_a, dict) else label_a
		key_b = label_b.get("id") if isinstance(label_b, dict) else label_b
		return key_a == key_b

	def _next_available_id(self, used_ids: set) -> int:
		used_numeric = [oid for oid in used_ids if isinstance(oid, int)]
		if not used_numeric:
			return 1
		candidate = max(used_numeric) + 1
		while candidate in used_ids:
			candidate += 1
		return candidate
