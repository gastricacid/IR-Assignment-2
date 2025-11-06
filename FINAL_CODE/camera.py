"""Camera utilities for the visual servoing demos."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3

from helpers import asSE3


class VirtualCamera:
	"""Simulated eye-in-hand camera used for visual servoing."""

	def __init__(self, robot, camera_offset=SE3(0, 0, 0.15) * SE3.RPY([0, np.pi, 0])):
		self.robot = robot
		self.camera_offset = camera_offset
		self.image_width = 640
		self.image_height = 480
		self.focal_length = 800.0

	def get_camera_pose(self) -> SE3:
		"""Return current camera pose in the world frame."""

		return self.robot.fkine(self.robot.q) * self.camera_offset

	def world_to_camera(self, point_world: np.ndarray) -> np.ndarray:
		"""Transform a 3D world point into the camera frame."""

		T_cam = self.get_camera_pose()
		return T_cam.inv() * point_world

	def project_to_image(self, point_world: np.ndarray) -> Tuple[float, float, float]:
		"""Project a 3D world point into image coordinates."""

		point_cam = np.asarray(self.world_to_camera(point_world), dtype=float).reshape(-1)
		if point_cam.size < 3:
			return None, None, None

		x, y, z = point_cam[:3]
		if z < 0.01:
			return None, None, None

		u = float(self.focal_length * x / z + self.image_width / 2)
		v = float(self.focal_length * y / z + self.image_height / 2)
		depth = float(z)

		if 0 <= u <= self.image_width and 0 <= v <= self.image_height:
			return u, v, depth
		return None, None, None

	def detect_objects(
		self,
		objects: List[geometry.Mesh],
		*,
		max_distance: float = 1.5,
	) -> List[Tuple[geometry.Mesh, float, float, float]]:
		"""Return visible objects with pixel coordinates and depth."""

		visible: List[Tuple[geometry.Mesh, float, float, float]] = []
		T_cam = self.get_camera_pose()
		cam_pos = T_cam.t

		for obj in objects:
			obj_se3 = asSE3(obj)
			obj_pos = obj_se3.t
			if np.linalg.norm(obj_pos - cam_pos) > max_distance:
				continue

			u, v, depth = self.project_to_image(obj_pos)
			if u is not None:
				visible.append((obj, u, v, depth))

		return visible


__all__ = ["VirtualCamera"]
