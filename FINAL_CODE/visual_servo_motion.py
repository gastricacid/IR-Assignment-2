"""Single-robot visual-servo motion sequences."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3

from camera import VirtualCamera
from helpers import (
	_bucket_drop_pose,
	_nearest_bucket_of_color,
	_set_mesh_pose,
	asSE3,
	dist_xy,
	ik_posonly,
	movej,
)
from ibvs import ibvs_control_law


def visual_servo_to_object(
	robot,
	camera: VirtualCamera,
	target_obj: geometry.Mesh,
	env,
	*,
	desired_image_pos: Tuple[float, float] = (320.0, 240.0),
	max_iterations: int = 150,
	tolerance: float = 5.0,
	dt: float = 0.05,
) -> bool:
	"""Use IBVS to centre the target object in the camera image."""

	obj_se3 = asSE3(target_obj)
	target_pos = obj_se3.t
	desired_features = np.array(desired_image_pos)

	print(f"[IBVS] Starting visual servoing to object at {obj_se3.t.round(3)}")

	for iteration in range(max_iterations):
		u, v, depth = camera.project_to_image(target_pos)
		if u is None:
			print("[IBVS] Object not visible in camera!")
			return False

		current_features = np.array([u, v])
		error = np.linalg.norm(desired_features - current_features)
		if error < tolerance:
			print(f"[IBVS] Converged! Error: {error:.2f} pixels")
			return True

		if iteration % 20 == 0:
			print(f"[IBVS] Iter {iteration}: error={error:.2f}px, depth={depth:.3f}m")

		v_cam, _ = ibvs_control_law(
			current_features,
			desired_features,
			np.array([depth]),
			camera.focal_length,
			lambda_gain=0.3,
		)

		R_cam_ee = camera.camera_offset.R
		v_ee_linear = R_cam_ee.T @ v_cam[:3]
		omega_ee = R_cam_ee.T @ v_cam[3:]
		v_ee = np.concatenate([v_ee_linear, omega_ee])

		try:
			J = robot.jacob0(robot.q)
			lambda_damping = 0.01
			J_damped = J.T @ np.linalg.inv(J @ J.T + lambda_damping ** 2 * np.eye(6))
			q_dot = J_damped @ v_ee
			q_dot = np.clip(q_dot, -1.0, 1.0)
			robot.q = robot.q + q_dot * dt
			env.step(dt)
		except Exception as exc:  # pragma: no cover - safety net around toolbox call
			print(f"[IBVS] Jacobian error: {exc}")
			return False

	print(f"[IBVS] Failed to converge after {max_iterations} iterations")
	return False


def visual_servo_approach(
	robot,
	camera: VirtualCamera,
	target_obj: geometry.Mesh,
	env,
	*,
	approach_height: float = 0.18,
	final_height: float = 0.03,
	dt: float = 0.05,
) -> bool:
	"""Perform a two-stage approach: hover then descend."""

	print("[APPROACH] Visual servo approach to object")

	obj_se3 = asSE3(target_obj)
	obj_pos = obj_se3.t
	print(f"[APPROACH] Target object at: {obj_pos.round(3)}")

	T_coarse = SE3(obj_pos[0], obj_pos[1], obj_pos[2] + approach_height) * SE3.RPY([0, np.pi, 0])
	print("[APPROACH] Moving to coarse position above object...")
	sol = ik_posonly(robot, T_coarse, ilimit=200)
	if not sol.success:
		print("[APPROACH] Failed to find IK solution for coarse approach")
		return False

	movej(robot, sol.q, env, T=2.0, steps=80)
	print("[APPROACH] Reached coarse position")

	target_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + approach_height])
	print("[APPROACH] Starting visual servoing for fine positioning...")

	for iteration in range(100):
		u, v, depth = camera.project_to_image(target_pos)
		if u is None:
			print("[APPROACH] Object not visible!")
			return False

		current_features = np.array([u, v])
		desired_features = np.array([320.0, 240.0])
		error = np.linalg.norm(desired_features - current_features)

		if iteration % 20 == 0:
			print(f"[APPROACH] VS iteration {iteration}: error={error:.2f}px, depth={depth:.3f}m")

		if error < 15.0:
			print(f"[APPROACH] Visual servoing converged! Error: {error:.2f}px")
			break

		v_cam, _ = ibvs_control_law(
			current_features,
			desired_features,
			np.array([depth]),
			camera.focal_length,
			lambda_gain=0.2,
		)

		v_cam[2] = 0.0
		v_cam[3:] = 0.0

		R_cam_ee = camera.camera_offset.R
		v_ee_linear = R_cam_ee.T @ v_cam[:3]
		v_ee = np.concatenate([v_ee_linear, np.zeros(3)])

		try:
			J = robot.jacob0(robot.q)
			lambda_damping = 0.02
			J_damped = J.T @ np.linalg.inv(J @ J.T + lambda_damping ** 2 * np.eye(6))
			q_dot = J_damped @ v_ee
			q_dot = np.clip(q_dot, -0.3, 0.3)
			robot.q = robot.q + q_dot * dt
			env.step(dt)
		except Exception:
			print("[APPROACH] Jacobian computation failed")
			break

	print("[APPROACH] Descending to grasp height...")
	T_grasp = SE3(obj_pos[0], obj_pos[1], obj_pos[2] + final_height) * SE3.RPY([0, np.pi, 0])
	sol = ik_posonly(robot, T_grasp, ilimit=200)
	if sol.success:
		movej(robot, sol.q, env, T=1.0, steps=50)
		print("[APPROACH] Successfully reached grasp position")
		return True

	print("[APPROACH] Failed to reach grasp position")
	return False


def run_visual_servo_sort(
	env,
	robot,
	camera: VirtualCamera,
	assets,
	*,
	attach_offset=SE3(0, 0, 0.08),
	exclude: Optional[set] = None,
) -> List[geometry.Mesh]:
	"""Sort gold and trash items using visual-servoing."""

	q_home = np.array(robot.q, dtype=float)
	baseT = asSE3(robot.base)
	buckets = assets["buckets"]

	skip = exclude or set()
	remaining = [(m, "gold") for m in assets["gold"] if id(m) not in skip]
	remaining += [(m, "trash") for m in assets["trash"] if id(m) not in skip]

	moved: List[geometry.Mesh] = []
	step = 1

	print("\n[INIT] Moving to initial search position...")
	q_search = np.array([0, -np.pi / 4, np.pi / 3, 0, np.pi / 4, 0])
	movej(robot, q_search, env, T=2.0, steps=80)

	while remaining:
		visible_objects = camera.detect_objects([m for m, _ in remaining], max_distance=1.5)
		print(f"\n[SCAN] Found {len(visible_objects)} visible objects")

		if not visible_objects:
			print("[VISUAL SERVO] No objects visible in camera range")
			search_poses = [
				SE3(0.2, -0.4, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
				SE3(-0.2, -0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
				SE3(0.0, -0.2, 0.6) * SE3.RPY([0, np.pi * 0.7, 0]),
				SE3(0.3, 0.0, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
				SE3(-0.3, 0.0, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
			]

			for search_pose in search_poses:
				print("[SCAN] Trying search position...")
				sol = ik_posonly(robot, search_pose, ilimit=200)
				if sol.success:
					movej(robot, sol.q, env, T=1.5, steps=60)
					visible_objects = camera.detect_objects([m for m, _ in remaining], max_distance=1.5)
					print(f"[SCAN] Now see {len(visible_objects)} objects")
					if visible_objects:
						break

			if not visible_objects:
				print("[VISUAL SERVO] No more objects found after searching. Mission complete.")
				break

		target_obj, u, v, depth = min(visible_objects, key=lambda x: dist_xy(x[0], baseT))

		kind = next((k for mesh, k in remaining if mesh is target_obj), None)
		if kind is None:
			continue

		bucket = _nearest_bucket_of_color(buckets, want_gold=(kind == "gold"), ref_pose=baseT)
		if not bucket:
			remaining = [(mesh, k) for mesh, k in remaining if mesh is not target_obj]
			continue

		obj_se3 = asSE3(target_obj)
		drop_str = (
			f"\n{'=' * 60}\n[VISUAL SERVO {step}] Sorting {kind.upper()}\n"
			f"  Object position: {obj_se3.t.round(3)}\n"
			f"  Image position: ({u:.1f}, {v:.1f}) pixels, depth: {depth:.3f}m\n"
			f"{'=' * 60}"
		)
		print(drop_str)
		step += 1

		success = visual_servo_approach(
			robot,
			camera,
			target_obj,
			env,
			approach_height=0.18,
			final_height=0.03,
		)

		if not success:
			print("[VISUAL SERVO] Failed to reach object, skipping")
			remaining = [(mesh, k) for mesh, k in remaining if mesh is not target_obj]
			continue

		carried = target_obj
		_set_mesh_pose(carried, robot.fkine(robot.q) * attach_offset)
		print("[GRASP] Object grasped!")

		q_current = np.array(robot.q)
		T_lift = robot.fkine(q_current) * SE3(0, 0, 0.15)
		sol = ik_posonly(robot, T_lift)
		if sol.success:
			movej(robot, sol.q, env, T=0.8, steps=40, carried=carried, attach_offset=attach_offset)

		bucketT = asSE3(bucket)
		bucket_drop_target = _bucket_drop_pose(bucket, depth=0.1, x_offset=0.05)
		drop_point = bucket_drop_target.t
		print(f"[TRANSPORT] Moving to bucket at {bucketT.t.round(3)}")
		T_above_bucket = SE3(drop_point[0], drop_point[1], bucketT.t[2] + 0.18) * SE3.RPY([0, np.pi, 0])
		T_drop = SE3(drop_point[0], drop_point[1], drop_point[2] + 0.03) * SE3.RPY([0, np.pi, 0])

		sol = ik_posonly(robot, T_above_bucket)
		if sol.success:
			movej(robot, sol.q, env, T=1.5, steps=60, carried=carried, attach_offset=attach_offset)

		sol = ik_posonly(robot, T_drop)
		if sol.success:
			movej(robot, sol.q, env, T=0.8, steps=40, carried=carried, attach_offset=attach_offset)

		_set_mesh_pose(carried, bucket_drop_target)
		env.step(0.05)
		print(f"[DROP] Dropped {kind} in bucket!")

		sol = ik_posonly(robot, T_above_bucket)
		if sol.success:
			movej(robot, sol.q, env, T=0.8, steps=40)

		remaining = [(mesh, k) for mesh, k in remaining if mesh is not target_obj]
		moved.append(carried)
		print(f"[PROGRESS] {len(moved)}/{len(moved) + len(remaining)} objects sorted")

	movej(robot, q_home, env, T=2.0, steps=100)
	print("\n[VISUAL SERVO] Mission complete. Returned home.")
	return moved


__all__ = [
	"visual_servo_to_object",
	"visual_servo_approach",
	"run_visual_servo_sort",
]
