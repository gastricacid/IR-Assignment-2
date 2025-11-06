"""High-level demo orchestration for the visual servoing assessment."""

from __future__ import annotations

import os
import threading
import time
from math import pi
from typing import Dict, List, Tuple

try:
	from pynput import keyboard  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
	keyboard = None

try:
	import swift
except Exception:  # pragma: no cover - runtime availability check
	swift = None

from spatialmath import SE3

from camera import VirtualCamera
from dual_robot_sequences import run_dual_robot_visual_servo, run_dual_robot_visual_servo_with_tracking
from environment_assets import spawn_environment, spawn_parol6, spawn_robot
from helpers import _set_mesh_pose, check_robot_collision, check_robot_self_collision, configure_movej_runtime, movej


class VisualServoDemo:
	"""Wrap the Swift environment, robots, and GUI interactions."""

	def __init__(self):
		self.env = None
		self.assets: Dict[str, List] = {}
		self.robots: List = []
		self.thread_running = False
		self.estop_active = False
		self.keyboard_listener = None
		self.teach_positions: List[Dict[str, object]] = []
		self.carried_objects: Dict[int, Tuple[object, object]] = {}
		self.update_thread = None
		self.update_thread_running = False

	# ----- Internal helpers -----
	def _estop_check(self) -> bool:
		return self.estop_active

	def _collect_obstacles(self) -> List:
		obstacles: List = []
		for key in ("furniture", "extras"):
			obstacles.extend(self.assets.get(key, []))
		return obstacles

	def _env_collision(self, robot, obstacles):
		return check_robot_collision(robot, obstacles, safety_margin=0.04)

	def _self_collision(self, robot):
		return check_robot_self_collision(robot, safety_margin=0.04)

	def _update_carried_objects_loop(self):
		while self.update_thread_running:
			try:
				if self.env and self.robots:
					for robot_idx, (carried_mesh, attach_offset) in list(self.carried_objects.items()):
						if robot_idx < len(self.robots):
							robot = self.robots[robot_idx]
							_set_mesh_pose(carried_mesh, robot.fkine(robot.q) * attach_offset)
					self.env.step(0.01)
			except Exception:
				pass
			time.sleep(0.02)

	def _start_carried_update_thread(self):
		if not self.update_thread or not self.update_thread.is_alive():
			self.update_thread_running = True
			self.update_thread = threading.Thread(
				target=self._update_carried_objects_loop, daemon=True
			)
			self.update_thread.start()

	def _stop_carried_update_thread(self):
		self.update_thread_running = False
		if self.update_thread:
			self.update_thread.join(timeout=1.0)

	def set_carried_object(self, robot_idx: int, mesh, attach_offset):
		if mesh is None:
			self.carried_objects.pop(robot_idx, None)
		else:
			self.carried_objects[robot_idx] = (mesh, attach_offset)

	def clear_all_carried_objects(self):
		self.carried_objects.clear()

	# ----- Public controls -----
	def run(self):
		if swift is None:
			print("[GUI] Swift simulator not available. Cannot start demo.")
			return

		if self.thread_running:
			print("[GUI] Simulation already running.")
			return

		self.thread_running = True
		self.estop_active = False
		self.start_keyboard_listener()
		self._start_carried_update_thread()

		try:
			self.env = swift.Swift()
			self.env.launch(realtime=True)

			current_dir = os.path.dirname(os.path.abspath(__file__))
			self.assets = spawn_environment(self.env, current_dir)

			print("[GUI] Spawning robots for visual servo demo...")
			fanuc = spawn_robot(self.env, base=SE3(0.0, -0.6, 0.0), rz=pi / 2)
			parol = spawn_parol6(self.env, base=SE3(0.0, 0.2, 0.0), rz=-pi / 2)
			self.robots = [parol, fanuc]

			camera_offset = SE3(0, 0, 0.15) * SE3.RPY([0, pi, 0])
			camera_fanuc = VirtualCamera(fanuc, camera_offset=camera_offset)
			camera_parol = VirtualCamera(parol, camera_offset=camera_offset)

			configure_movej_runtime(
				estop_check=self._estop_check,
				obstacles=self._collect_obstacles(),
				env_collision_fn=self._env_collision,
				halt_on_collision=False,
			)

			run_dual_robot_visual_servo_with_tracking(
				self.env,
				fanuc,
				camera_fanuc,
				parol,
				camera_parol,
				self.assets,
				demo=self,
			)

			print("[GUI] Visual servo mission complete. Holding window...")
			self.env.hold()

		except Exception as exc:  # pragma: no cover - runtime guard
			print(f"[GUI] Simulation error: {exc}")

		finally:
			configure_movej_runtime(
				estop_check=None,
				obstacles=None,
				env_collision_fn=None,
				self_collision_fn=None,
				halt_on_collision=None,
			)
			self._stop_carried_update_thread()
			self.clear_all_carried_objects()
			self.stop_keyboard_listener()
			self.thread_running = False

	def stop(self):
		self.estop_active = True
		configure_movej_runtime(estop_check=None)
		if self.env is not None:
			try:
				self.env.close()
			except Exception as exc:  # pragma: no cover
				print(f"[GUI] Failed to close Swift: {exc}")
		self.env = None
		self.thread_running = False

	def emergency_stop(self):
		self.estop_active = True
		print("[E-STOP] Emergency stop activated.")

	def resume(self):
		self.estop_active = False
		print("[RESUME] Emergency stop cleared.")

	def move_joint(self, robot_idx: int, joint_idx: int, delta: float) -> bool:
		if not self.robots or robot_idx >= len(self.robots):
			return False
		robot = self.robots[robot_idx]
		q_new = robot.q.copy()
		if joint_idx >= len(q_new):
			return False
		q_new[joint_idx] += delta
		robot.q = q_new
		if self.env:
			self.env.step(0.01)
		return True

	def record_position(self, robot_idx: int) -> bool:
		if not self.robots or robot_idx >= len(self.robots):
			return False
		robot = self.robots[robot_idx]
		self.teach_positions.append(
			{
				"robot_idx": robot_idx,
				"joint_config": robot.q.copy(),
				"end_effector": robot.fkine(robot.q),
			}
		)
		return True

	def clear_positions(self):
		self.teach_positions = []

	def playback_positions(self, speed: int = 60):
		if not self.teach_positions:
			print("[TEACH] No positions to playback.")
			return
		for entry in self.teach_positions:
			if self.estop_active:
				print("[TEACH] Aborted due to E-stop.")
				break
			idx = entry["robot_idx"]
			if idx >= len(self.robots):
				continue
			robot = self.robots[idx]
			movej(
				robot,
				entry["joint_config"],
				self.env,
				T=max(0.1, 2.0 * (60 / max(speed, 1))),
			)

	def get_robot_status(self, robot_idx: int):
		if not self.robots or robot_idx >= len(self.robots):
			return None
		robot = self.robots[robot_idx]
		return {
			"joint_config": robot.q.copy(),
			"end_effector": robot.fkine(robot.q),
		}

	# ----- Keyboard listener -----
	def start_keyboard_listener(self):
		if keyboard is None:
			print("[INFO] pynput not available; keyboard E-stop disabled.")
			return
		if self.keyboard_listener:
			return

		def on_press(key):
			try:
				if key.char and key.char.lower() == "e":
					self.emergency_stop()
				elif key.char and key.char.lower() == "r":
					self.resume()
			except AttributeError:
				pass

		self.keyboard_listener = keyboard.Listener(on_press=on_press)
		self.keyboard_listener.start()
		print("[INFO] Hardware E-Stop enabled: 'E' to stop, 'R' to resume")

	def stop_keyboard_listener(self):
		if self.keyboard_listener:
			self.keyboard_listener.stop()
			self.keyboard_listener = None


def run_visual_servo_cli():
	if swift is None:
		print("Swift simulator not available. CLI demo cannot run.")
		return

	env = swift.Swift()
	env.launch(realtime=True)
	current_dir = os.path.dirname(os.path.abspath(__file__))
	assets = spawn_environment(env, current_dir)

	print("\n" + "=" * 60)
	print("VISUAL SERVOING DEMO - SEQUENTIAL DUAL ROBOT")
	print("=" * 60)
	print("PAROL6 will pick objects and place them on the table")
	print("Fanuc will then pick from table and sort to buckets")
	print("=" * 60 + "\n")

	print("[SETUP] Spawning Fanuc LRMate200iC...")
	fanuc = spawn_robot(env, base=SE3(0.0, -0.6, 0.0), rz=pi / 2)

	print("[SETUP] Spawning PAROL6...")
	parol = spawn_parol6(env, base=SE3(0.0, 0.2, 0.0), rz=-pi / 2)

	camera_offset = SE3(0, 0, 0.15) * SE3.RPY([0, pi, 0])
	camera_fanuc = VirtualCamera(fanuc, camera_offset=camera_offset)
	camera_parol = VirtualCamera(parol, camera_offset=camera_offset)

	print("[SETUP] Cameras initialized for both robots\n")

	moved_fanuc, moved_parol = run_dual_robot_visual_servo(
		env, fanuc, camera_fanuc, parol, camera_parol, assets
	)

	total_moved = len(moved_fanuc) + len(moved_parol)
	total_objects = len(assets["gold"]) + len(assets["trash"])

	print("\n" + "=" * 60)
	print("MISSION COMPLETE - SUMMARY")
	print("=" * 60)
	print(f"PAROL6 placed: {len(moved_parol)} objects on table")
	print(f"Fanuc sorted:  {len(moved_fanuc)} objects to buckets")
	print(f"Total sorted: {total_moved}/{total_objects} objects")
	print("=" * 60)
	print("\n[INFO] Close window to exit.")
	env.hold()


__all__ = ["VisualServoDemo", "run_visual_servo_cli"]
