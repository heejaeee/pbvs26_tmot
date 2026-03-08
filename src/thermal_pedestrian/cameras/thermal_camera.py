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
from __future__ import annotations

import colorsys
import itertools
import json
import os
import pickle
import sys
import threading
import uuid
import glob
import copy
import random
import colorsys
from functools import cmp_to_key
from queue import Queue
from operator import itemgetter
from timeit import default_timer as timer
from typing import Union, Optional
import cv2
import torch
import numpy as np
from tqdm import tqdm
from thermal_pedestrian.configuration import (
	root_dir,
	config_dir,
)
from thermal_pedestrian.cameras.base import BaseCamera
from thermal_pedestrian.core.factory.builder import CAMERAS, DETECTORS, TRACKERS
from thermal_pedestrian.core.data.class_label import ClassLabels
from thermal_pedestrian.core.io.frame import FrameLoader, FrameWriter
from thermal_pedestrian.core.io.video import VideoLoader
from thermal_pedestrian.core.io.filedir import (
	is_basename,
	is_json_file
)
from thermal_pedestrian.core.utils.rich import console
from thermal_pedestrian.detectors.basedetector import BaseDetector
from thermal_pedestrian.trackers.basetracker import BaseTracker
from thermal_pedestrian.core.objects.instance import Instance
from thermal_pedestrian.core.utils.bbox import bbox_xyxy_to_xywh
from thermal_pedestrian.trackers.id_switch_postprocessor import IDSwitchPostprocessor

__all__ = [
	"ThermalCamera"
]

# NOTE: only for PBVS 2025
classes_pbvs = ['person']


# MARK: - ThermalCamera

# noinspection PyAttributeOutsideInit

@CAMERAS.register(name="thermal_camera")
class ThermalCamera(BaseCamera):

	# MARK: Magic Functions

	def __init__(
			self,
			data         : dict,
			dataset      : str,
			name         : str,
			detector     : dict,
			tracker      : dict,
			data_loader  : dict,
			data_writer  : Union[FrameWriter,  dict],
			process      : dict,
			id_          : Union[int, str] = uuid.uuid4().int,
			verbose      : bool            = False,
			drawing      : bool            = False,
			queue_size   : int             = 10,
			*args, **kwargs
	):
		"""

		Args:
			dataset (str):
				Dataset name. It is also the name of the directory inside
				`data_dir`.
			subset (str):
				Subset name. One of: [`dataset_a`, `dataset_b`].
			name (str):
				Camera name. It is also the name of the camera's config
				files.
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
			data_loader (FrameLoader, dict):
				Data loader object or a data loader's config dictionary.
			data_writer (VideoWriter, dict):
				Data writer object or a data writer's config dictionary.
			id_ (int, str):
				Camera's unique ID.
			verbose (bool):
				Verbosity mode. Default: `False`.
			queue_size (int):
				Size of queue store the information
		"""
		super().__init__(id_=id_, dataset=dataset, name=name)
		# NOTE: Init attributes
		self.start_time = None
		self.pbar       = None

		# NOTE: Define attributes
		self.process         = process
		self.verbose         = verbose
		self.drawing         = drawing

		# NOTE: Define configurations
		self.data_cfg        = data
		self.detector_cfg    = detector
		self.tracker_cfg     = tracker
		self.data_loader_cfg = data_loader
		self.data_writer_cfg = data_writer
		self.id_switch_postprocessor = None

		# NOTE: Queue
		self.frames_queue                 = Queue(maxsize = self.data_loader_cfg['queue_size'])
		self.detections_queue_identifier  = Queue(maxsize = self.detector_cfg['queue_size'])
		self.writer_queue                 = Queue(maxsize = self.data_writer_cfg['queue_size'])

		# NOTE: Init modules
		self.init_dirs()

	# MARK: Configure

	def init_dirs(self):
		"""Initialize dirs.
		"""
		self.root_dir    = os.path.join(root_dir)
		self.configs_dir = os.path.join(config_dir)
		self.outputs_dir = os.path.join(self.data_writer_cfg["output_dir"])

	def init_class_labels(self, class_labels: Union[ClassLabels, dict]):
		"""Initialize class_labels.

		Args:
			class_labels (ClassLabels, dict):
				ClassLabels object or a config dictionary.
		"""
		if isinstance(class_labels, ClassLabels):
			self.class_labels = class_labels
		elif isinstance(class_labels, str):
			self.class_labels = ClassLabels.create_from_file(class_labels)
		elif isinstance(class_labels, dict):
			file = class_labels["file"]
			if is_json_file(file):
				self.class_labels = ClassLabels.create_from_file(file)
			elif is_basename(file):
				file              = os.path.join(self.root_dir, file)
				self.class_labels = ClassLabels.create_from_file(file)
		else:
			file              = os.path.join(self.root_dir, f"class_labels.json")
			self.class_labels = ClassLabels.create_from_file(file)
			print(f"Cannot initialize class_labels from {class_labels}. "
			      f"Attempt to load from {file}.")

	def init_detector(self, detector: Union[BaseDetector, dict]):
		"""Initialize detector.

		Args:
			detector (BaseDetector, dict):
				Detector object or a detector's config dictionary.
		"""
		console.log(f"Initiate {self.detector_cfg['name']}.")
		if isinstance(detector, BaseDetector):
			self.detector = detector
		elif isinstance(detector, dict):
			detector["class_labels"] = self.class_labels
			self.detector = DETECTORS.build(**detector)
		else:
			raise ValueError(f"Cannot initialize detector with {detector}.")

	def init_tracker(self, tracker: Union[BaseTracker, dict]):
		"""Initialize tracker.

		Args:
			tracker (BaseTracker, dict):
				Tracking object or a tracker's config dictionary.
		"""
		console.log(f"Initiate {self.tracker_cfg['name']}.")
		if isinstance(tracker, BaseTracker):
			self.tracker = tracker
		elif isinstance(tracker, dict):
			self.tracker = TRACKERS.build(**tracker)
		else:
			raise ValueError(f"Cannot initialize detector with {tracker}.")

		self.id_switch_postprocessor = None
		if isinstance(self.tracker_cfg, dict) and self.tracker_cfg.get("id_switch_postprocess", False):
			self.id_switch_postprocessor = IDSwitchPostprocessor(
				max_distance=self.tracker_cfg.get("id_switch_max_distance", 50.0),
				max_age=self.tracker_cfg.get("id_switch_max_age", 5),
				label_check=self.tracker_cfg.get("id_switch_label_check", True),
			)

	# MARK: Run

	def run_detector(self):
		"""Run detection model"""
		# create directory to store result
		folder_det_ou = os.path.join(
			self.data_writer_cfg['output_dir'],
			"detection",
			self.detector_cfg['folder_out'],
			self.data_writer_cfg['seq_cur'],
			"thermal/yolo"
		)
		os.makedirs(folder_det_ou, exist_ok=True)

		# DEBUG: draw
		if self.drawing:
			folder_img_ou = os.path.join(
				self.data_writer_cfg['output_dir'],
				"detection",
				self.detector_cfg['folder_out'],
				self.data_writer_cfg['seq_cur'],
				"thermal/img_draw"
			)
			os.makedirs(folder_img_ou, exist_ok=True)

		# load dataloader
		self.data_loader = FrameLoader(data=self.data_loader_cfg['data_path'], batch_size=self.data_loader_cfg['batch_size'])

		pbar = tqdm(total=self.data_loader.num_frames, desc=f"Detection: {self.data_writer_cfg['seq_cur']}")

		# run detection
		with torch.no_grad():  # phai them cai nay khong la bi memory leak
			for images, indexes, files_path, rel_paths in self.data_loader:

				# Detect batch of instances
				batch_instances = self.detector.detect(
					indexes=indexes, images=images
				)

				# Process the detection result of each image
				for index_batch, (index_image, file_path_img, batch) in enumerate(zip(indexes, files_path, batch_instances)):

					# DEBUG: draw
					if self.drawing:
						image_draw = images[index_batch].copy()

					# init output file
					file_path_txt_ou = os.path.join(
						folder_det_ou,
						f"{os.path.splitext(os.path.basename(file_path_img))[0]}.txt"
					)

					# process each detection
					with open(file_path_txt_ou, 'w') as f_write:
						for index_in, instance in enumerate(batch):
							# if instance.confidence < self.data_writer_cfg['min_confidence']:
							# 	continue
							class_id   = instance.class_id
							bbox_xyxyn = instance.bbox
							score      = instance.confidence
							f_write.write(f"{class_id} {bbox_xyxyn[0]:.6f} {bbox_xyxyn[1]:.6f} {bbox_xyxyn[2]:.6f} {bbox_xyxyn[3]:.6f} {score:.6f}\n")

							# DEBUG: draw
							if self.drawing:
								image_draw = plot_one_box_yolo(
									bbox = bbox_xyxyn,
									img  = image_draw,
									label= f"{instance.label.name}_{score:.2f}"
								)

					# DEBUG: draw
					if self.drawing:
						cv2.imwrite(os.path.join(folder_img_ou, os.path.basename(file_path_img)), image_draw)

				pbar.update(len(indexes))
			pbar.close()

	def run_tracker_forward(self):
		"""Run tracking model"""
		# input folder directory to load detection
		folder_det_in = os.path.join(
			self.data_writer_cfg['output_dir'],
			"detection",
			self.detector_cfg['folder_out'],
			self.data_writer_cfg['seq_cur'],
			"thermal/yolo"
		)

		# create file to store result
		file_mot_ou = os.path.join(
			self.data_writer_cfg['output_dir'],
			"tracking",
			self.tracker_cfg['folder_out'],
			self.data_writer_cfg['seq_cur'],
			"thermal",
			f"{self.data_writer_cfg['seq_cur']}_thermal.txt"
		)
		os.makedirs(os.path.dirname(file_mot_ou), exist_ok=True)

		# DEBUG: draw
		if self.drawing:
			folder_img_ou = os.path.join(
				self.data_writer_cfg['output_dir'],
				"tracking",
				self.tracker_cfg['folder_out'],
				self.data_writer_cfg['seq_cur'],
				"thermal/img_draw"
			)
			os.makedirs(folder_img_ou, exist_ok=True)

			folder_img_ou_ori_det = os.path.join(
				self.data_writer_cfg['output_dir'],
				"tracking",
				self.tracker_cfg['folder_out'],
				self.data_writer_cfg['seq_cur'],
				"thermal/img_draw_ori"
			)
			os.makedirs(folder_img_ou_ori_det, exist_ok=True)

		# load list images FORWARDING
		list_imgs = [s for s in sorted(os.listdir(self.data_loader_cfg['data_path']), key=cmp_to_key(compare_filenames), reverse=False)]

		if self.id_switch_postprocessor is not None:
			self.id_switch_postprocessor.reset()

		with (open(file_mot_ou, 'w') as f_write):
			# run tracking
			for img_index, img_name in enumerate(tqdm(list_imgs, desc=f"Tracking forward: {self.data_writer_cfg['seq_cur']}")):
				# init
				img_path = os.path.join(self.data_loader_cfg['data_path'], img_name)
				det_path = os.path.join(folder_det_in, f"{os.path.splitext(img_name)[0]}.txt")

				# load image
				img = cv2.imread(img_path)

				# DEBUG: draw
				if self.drawing:
					image_draw         = img.copy()
					image_draw_ori_det = img.copy()

				# load yolo detection
				# class_id, c_xn, c_yn, wn, hw, score
				# dets = np.loadtxt(det_path, dtype=np.float32, delimiter=' ').reshape(-1, 6)  # for detection has score

				instances = []
				with open(det_path, "r") as f_read:
					for line in f_read:
						line   = line.strip().split(" ")
						det    = [float(x) for x in line]
						det[0] = int(det[0])
						if len(det) == 5:  # class_id, c_xn, c_yn, wn, hw, and does not have score
							det.append(float(random.randint(60, 90) / 100))   # random generate

						if det[5] < self.tracker_cfg['min_confidence_det']:
							continue

						instance = Instance(
									video_name   = self.data_writer_cfg['seq_cur'],
									frame_index  = img_index,
									bbox         = np.asarray(covert_bbox_yolo_to_voc_format(det[1: 5], img)),
									confidence   = det[5],
									class_label  = self.class_labels.class_labels[det[0]]
						)

						# filter detection
						# if width and height equal 0
						# if (abs(int(instance.bbox[0] - instance.bbox[2])) < 2 or
						# 		abs(int(instance.bbox[1] - instance.bbox[3])) < 2):
						# 	continue

						instances.append(instance)

						# DEBUG: draw
						if self.drawing:
							image_draw_ori_det = plot_one_box_voc(
								bbox  = instance.bbox,
								img   = image_draw_ori_det,
								color = create_unique_color_uchar(instance.id % 1000),
								label = f"{instance.id % 1000}_{instance.confidence:.2f}"
							)

				# tracking process
				self.tracker.update(detections=instances, image=img)
				gmos = self.tracker.tracks
				if self.id_switch_postprocessor is not None:
					self.id_switch_postprocessor.apply(gmos, frame_index=img_index)

				# write result
				# '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
				for gmo in gmos:
					if gmo.time_since_update > 0:  # the track does not have any dets match
						continue
					track_id = getattr(gmo, "output_id", gmo.id)
					bbox = bbox_xyxy_to_xywh(gmo.current_bbox)

					str_out = (f"{img_index + 1},"  # because frame start from 1, PBVS rule
								f"{int(track_id)},"  
								f"{abs(bbox[0])},"
								f"{abs(bbox[1])},"
								f"{abs(bbox[2])},"
								f"{abs(bbox[3])},"
								f"{gmo.confidence:.3f},"
								f"{gmo.current_label['id']},"
								f"-1,-1\n")

					f_write.write(str_out)

					# DEBUG: draw
					if self.drawing:
						image_draw = plot_one_box_voc(
							bbox  = gmo.current_bbox,
							img   = image_draw,
							color = create_unique_color_uchar(int(track_id) % 1000),
							label = f"{int(track_id) % 1000}_{gmo.confidence:.2f}"
						)

				# DEBUG: draw
				if self.drawing:
					cv2.imwrite(os.path.join(folder_img_ou, os.path.basename(img_path)), image_draw)
					cv2.imwrite(os.path.join(folder_img_ou_ori_det, os.path.basename(img_path)), image_draw_ori_det)

	def run_tracker_backward(self):
		"""Run tracking model"""
		# input folder directory to load detection
		folder_det_in = os.path.join(
			self.data_writer_cfg['output_dir'],
			"detection",
			self.detector_cfg['folder_out'],
			self.data_writer_cfg['seq_cur'],
			"thermal/yolo"
		)

		# create file to store result
		file_mot_ou = os.path.join(
			self.data_writer_cfg['output_dir'],
			"tracking",
			f"{self.tracker_cfg['folder_out']}_backward",
			self.data_writer_cfg['seq_cur'],
			"thermal",
			f"{self.data_writer_cfg['seq_cur']}_thermal.txt"
		)
		os.makedirs(os.path.dirname(file_mot_ou), exist_ok=True)

		# DEBUG: draw
		if self.drawing:
			folder_img_ou = os.path.join(
				self.data_writer_cfg['output_dir'],
				"tracking",
				f"{self.tracker_cfg['folder_out']}_backward",
				self.data_writer_cfg['seq_cur'],
				"thermal/img_draw"
			)
			os.makedirs(folder_img_ou, exist_ok=True)

			folder_img_ou_ori_det = os.path.join(
				self.data_writer_cfg['output_dir'],
				"tracking",
				f"{self.tracker_cfg['folder_out']}_backward",
				self.data_writer_cfg['seq_cur'],
				"thermal/img_draw_ori"
			)
			os.makedirs(folder_img_ou_ori_det, exist_ok=True)

		# load list images BACKWARDING
		list_imgs = [s for s in sorted(os.listdir(self.data_loader_cfg['data_path']), key=cmp_to_key(compare_filenames), reverse=True)]

		if self.id_switch_postprocessor is not None:
			self.id_switch_postprocessor.reset()

		with (open(file_mot_ou, 'w') as f_write):
			# run tracking
			for img_index, img_name in enumerate(tqdm(list_imgs, desc=f"Tracking backward: {self.data_writer_cfg['seq_cur']}")):
				# init
				img_path = os.path.join(self.data_loader_cfg['data_path'], img_name)
				det_path = os.path.join(folder_det_in, f"{os.path.splitext(img_name)[0]}.txt")

				# load image
				img = cv2.imread(img_path)

				# DEBUG: draw
				if self.drawing:
					image_draw         = img.copy()
					image_draw_ori_det = img.copy()

				# load yolo detection
				# class_id, c_xn, c_yn, wn, hw, score
				# dets = np.loadtxt(det_path, dtype=np.float32, delimiter=' ').reshape(-1, 6)  # for detection has score

				instances = []
				with open(det_path, "r") as f_read:
					for line in f_read:
						line   = line.strip().split(" ")
						det    = [float(x) for x in line]
						det[0] = int(det[0])
						if len(det) == 5:  # class_id, c_xn, c_yn, wn, hw, and does not have score
							det.append(float(random.randint(60, 90) / 100))   # random generate

						if det[5] < self.tracker_cfg['min_confidence_det']:
							continue

						instance = Instance(
									video_name   = self.data_writer_cfg['seq_cur'],
									frame_index  = img_index,
									bbox         = np.asarray(covert_bbox_yolo_to_voc_format(det[1: 5], img)),
									confidence   = det[5],
									class_label  = self.class_labels.class_labels[det[0]]
						)

						# filter detection
						# if width and height equal 0
						# if (abs(int(instance.bbox[0] - instance.bbox[2])) < 2 or
						# 		abs(int(instance.bbox[1] - instance.bbox[3])) < 2):
						# 	continue

						instances.append(instance)

						# DEBUG: draw
						if self.drawing:
							image_draw_ori_det = plot_one_box_voc(
								bbox  = instance.bbox,
								img   = image_draw_ori_det,
								color = create_unique_color_uchar(instance.id % 1000),
								label = f"{instance.id % 1000}_{instance.confidence:.2f}"
							)

				# tracking process
				self.tracker.update(detections=instances, image=img)
				gmos = self.tracker.tracks
				if self.id_switch_postprocessor is not None:
					self.id_switch_postprocessor.apply(gmos, frame_index=img_index)

				# write result
				# '{frame},{id},{x1},{y1},{w},{h},{s},{label},-1,-1\n'
				for gmo in gmos:
					if gmo.time_since_update > 0:  # the track does not have any dets match
						continue
					track_id = getattr(gmo, "output_id", gmo.id)
					bbox = bbox_xyxy_to_xywh(gmo.current_bbox)

					str_out = (f"{img_index + 1},"  # because frame start from 1, PBVS rule
								f"{int(track_id)},"  
								f"{abs(bbox[0])},"
								f"{abs(bbox[1])},"
								f"{abs(bbox[2])},"
								f"{abs(bbox[3])},"
								f"{gmo.confidence:.3f},"
								f"{gmo.current_label['id']},"
								f"-1,-1\n")

					f_write.write(str_out)

					# DEBUG: draw
					if self.drawing:
						image_draw = plot_one_box_voc(
							bbox  = gmo.current_bbox,
							img   = image_draw,
							color = create_unique_color_uchar(int(track_id) % 1000),
							label = f"{int(track_id) % 1000}_{gmo.confidence:.2f}"
						)

				# DEBUG: draw
				if self.drawing:
					cv2.imwrite(os.path.join(folder_img_ou, os.path.basename(img_path)), image_draw)
					cv2.imwrite(os.path.join(folder_img_ou_ori_det, os.path.basename(img_path)), image_draw_ori_det)

		# reverse mot result
		reverse_mot_result(file_mot_ou)

	def run_heuristic(self):
		"""Run heuristic model"""
		# NOTE: init parameter
		pass

	def writing_final_result(self, data_path_start, data_path_end):
		"""Write the final result to the file."""
		# NOTE: run writing
		pass

	def run(self):
		"""Main run loop."""
		self.run_routine_start()

		for seq in tqdm(self.data_loader_cfg.data_dir_seq):
			# set the current path to load data
			self.data_loader_cfg['data_path'] = os.path.join(
				self.data_loader_cfg.data_dir_prefix,
				seq,
				self.data_loader_cfg.data_dir_postfix,
			)
			self.data_writer_cfg['seq_cur'] = seq
			self.init_class_labels(class_labels=os.path.join(self.root_dir, self.data_cfg["class_labels"]["file"]))

			# NOTE: Detection process
			if self.process["function_detection"]:
				if (not hasattr(self, "detector")) or self.detector is None:
					self.init_detector(detector=self.detector_cfg)
				self.run_detector()


			# NOTE: Tracking process
			if self.process["function_tracking"]:
				if (not hasattr(self, "tracker")) or self.tracker is None:
					self.init_tracker(tracker=self.tracker_cfg)
				elif self.tracker_cfg['is_reload_model']:
					self.tracker.clear_model_memory()
					self.tracker = None
					self.init_tracker(tracker=self.tracker_cfg)
				self.run_tracker_forward()

				# if self.tracker_cfg['is_reload_model']:
				# 	self.tracker.clear_model_memory()
				# 	self.tracker = None
				# 	self.init_tracker(tracker=self.tracker_cfg)
				# self.run_tracker_backward()

		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.start_time = timer()
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		# clear detector
		if hasattr(self, 'detector') and self.detector is not None:
			self.detector.clear_model_memory()
			self.detector = None

		# clear tracker
		if hasattr(self, 'tracker') and self.tracker is not None:
			self.tracker.clear_model_memory()
			self.tracker = None

		cv2.destroyAllWindows()
		self.stop_time = timer()
		if self.pbar is not None:
			self.pbar.close()


# MARK - Ultilies

def covert_bbox_yolo_to_voc_format(bbox, img):
	"""Convert bbox from YOLO format to VOC format

	Args:
		bbox: YOLO format
		img: nparray, cv2 image

	Returns:

	"""
	h, w, _ = img.shape
	x_min = int(w * max(float(bbox[0]) - float(bbox[2]) / 2, 0))
	x_max = int(w * min(float(bbox[0]) + float(bbox[2]) / 2, 1))
	y_min = int(h * max(float(bbox[1]) - float(bbox[3]) / 2, 0))
	y_max = int(h * min(float(bbox[1]) + float(bbox[3]) / 2, 1))
	return x_min, y_min, x_max, y_max


def plot_one_box_yolo(bbox, img, color=None, label=None, line_thickness=1):
	"""Plots one bounding box on image img

	Args:
		bbox: YOLO format
		img: nparray, cv2 image
		color:
		label:
		line_thickness:

	Returns:

	"""
	x_min, y_min, x_max, y_max =  covert_bbox_yolo_to_voc_format(bbox, img)
	# h, w, _ = img.shape
	# x_min = int(w * max(float(bbox[0]) - float(bbox[2]) / 2, 0))
	# x_max = int(w * min(float(bbox[0]) + float(bbox[2]) / 2, 1))
	# y_min = int(h * max(float(bbox[1]) - float(bbox[3]) / 2, 0))
	# y_max = int(h * min(float(bbox[1]) + float(bbox[3]) / 2, 1))

	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (x_min, y_min), (x_max, y_max)
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	return img

def plot_one_box_voc(bbox, img, color=None, label=None, line_thickness=1):
	"""Plots one bounding box on image img

	Args:
		bbox: YOLO format
		img: nparray, cv2 image
		color:
		label:
		line_thickness:

	Returns:

	"""
	x_min, y_min, x_max, y_max =  bbox
	# h, w, _ = img.shape
	# x_min = int(w * max(float(bbox[0]) - float(bbox[2]) / 2, 0))
	# x_max = int(w * min(float(bbox[0]) + float(bbox[2]) / 2, 1))
	# y_min = int(h * max(float(bbox[1]) - float(bbox[3]) / 2, 0))
	# y_max = int(h * min(float(bbox[1]) + float(bbox[3]) / 2, 1))

	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

	return img


def create_unique_color_float(tag, hue_step=0.41):
	h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
	r, g, b = colorsys.hsv_to_rgb(h, 1., v)
	return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
	r, g, b = create_unique_color_float(tag, hue_step)
	return (int(255 * r), int(255 * g), int(255 * b))


def compare_filenames(x, y):
	"""Compare filenames
		-1 means x < y, 0 means x == y, 1 means x > y
	"""
	if int(os.path.splitext(os.path.basename(x))[0]) < int(os.path.splitext(os.path.basename(y))[0]):
		return -1
	elif int(os.path.splitext(os.path.basename(x))[0]) > int(os.path.splitext(os.path.basename(y))[0]):
		return 1
	else:
		return 0


def reverse_mot_result(file_path):
	"""Reverse the mot result file"""
	with open(file_path, "r") as f_read:
		lines = f_read.readlines()

	if len(lines) == 0:
		return

	results = []
	for line in lines:
		results.append(np.array(line.strip().split(",")))
	results = np.array(results, dtype=np.float32)

	# Max Track_ID
	index_frame_max = np.max(results[:, 0])
	track_id_max    = np.max(results[:, 1])

	with open(file_path, "w") as f_write:
		for result in reversed(results):
			result[0] = index_frame_max - result[0] + 1
			result[1] = track_id_max - result[1] + 1
			str_out = (f"{int(result[0])},"  # because frame start from 1, PBVS rule
						f"{int(result[1])},"
						f"{float(result[2])},"
						f"{float(result[3])},"
						f"{float(result[4])},"
						f"{float(result[5])},"
						f"{float(result[6]):.3f},"
						f"{int(result[7])},"
						f"-1,-1\n")
			f_write.write(str_out)
