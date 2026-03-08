# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #
# from __future__ import annotations

from typing import List
from typing import Union

import numpy as np
from filterpy.kalman import KalmanFilter

from thermal_pedestrian.core.objects.gmo import General_Moving_Object
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.core.utils.bbox import (
	bbox_xyxy_to_z,
	batch_bbox_iou,
	x_to_bbox_xyxy
)

np.random.seed(0)

__all__ = [
	"KalmanBBoxTrack"
]


# MARK: - Track

class KalmanBBoxTrack(General_Moving_Object):
	"""Kalman Bounding Box Track
	"""

	# MARK: Magic Functions

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		# NOTE: Define Kalman Filter (constant velocity model)
		self.kf   = KalmanFilter(dim_x=7, dim_z=4)
		self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
		self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
		
		self.kf.R[2:, 2:] *= 10.0
		self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
		self.kf.P         *= 10.0
		self.kf.Q[-1, -1] *= 0.01
		self.kf.Q[4:, 4:] *= 0.01
		
		# Here we assume that the ``General_Moving_Object`` object has already been init().
		# So ``self.current_bbox`` return the first bbox value.
		self.kf.x[:4] = bbox_xyxy_to_z(self.current_bbox)

	@classmethod
	def track_from_detection(cls, instance: Instance, **kwargs):
		"""Create ``General_Moving_Object`` object from ``Instance`` object.
		
		Args:
			instance (Instance):
		
		Returns:
			gmo (General_Moving_Object):
				The General_Moving_Object object.
		"""
		return cls(
			frame_index = instance.frame_index,
			timestamp   = instance.timestamp,
			bbox        = instance.bbox,
			polygon     = instance.polygon,
			confidence  = instance.confidence,
			label       = instance.label,
			roi_uuid    = instance.roi_uuid,
			bbox_id     = instance.id,
			**kwargs
		)
	
	# MARK: Property
	
	@property
	def matching_features(self):
		"""Return the features used to matched tracked objects with new detections.
		"""
		return self.current_bbox
		
	# MARK: Motion Model
	
	def update_motion_state(self, **kwargs):
		"""Updates the state of the motion model with observed bbox.
		"""
		self.time_since_update = 0
		self.history           = []
		self.hits             += 1
		self.hit_streak       += 1
		self.kf.update(bbox_xyxy_to_z(self.matching_features))
		
	def predict_motion_state(self):
		"""Advances the state of the motion model and returns the predicted estimate.
		"""
		if (self.kf.x[6] + self.kf.x[2]) <= 0:
			self.kf.x[6] *= 0.0
		self.kf.predict()
		self.age += 1
		if self.time_since_update > 0:
			self.hit_streak = 0
		self.time_since_update += 1
		self.history.append(x_to_bbox_xyxy(self.kf.x))
		return self.history[-1]

	def current_motion_state(self):
		"""
		Returns the current motion model estimate.
		"""
		return x_to_bbox_xyxy(self.kf.x)


