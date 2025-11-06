"""Helper utilities for the visual servoing assessment.

This module centralises shared math helpers, collision checks, and
motion primitives so they can be reused without keeping everything
inside `aseessment 2.py`.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import spatialgeometry as geometry
from spatialmath import SE3
import roboticstoolbox as rtb


def asSE3(obj) -> SE3:
    if isinstance(obj, SE3):
        return obj
    for attr in ("T", "pose", "A"):
        val = getattr(obj, attr, None)
        if val is not None:
            return SE3(val)
    if isinstance(obj, (np.ndarray, list)) and np.shape(obj) == (4, 4):
        return SE3(obj)
    raise TypeError(f"Cannot extract SE3 from type {type(obj)}")


def planar_xy(Tlike) -> np.ndarray:
    T = asSE3(Tlike)
    return np.array([T.t[0], T.t[1]], dtype=float)


def dist_xy(a, b) -> float:
    return float(np.linalg.norm(planar_xy(a) - planar_xy(b)))


def _nearest_bucket_of_color(
    buckets: List[geometry.Mesh], *, want_gold: bool, ref_pose: SE3
):
    if not buckets:
        return None
    subset = buckets[:2] if want_gold else buckets[2:]
    if not subset:
        subset = buckets
    return min(subset, key=lambda m: dist_xy(m, ref_pose))


def point_to_line_segment_distance(
    point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
) -> float:
    """Compute minimum distance from a 3D point to a line segment."""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_length_sq = float(np.dot(line_vec, line_vec))

    if line_length_sq == 0.0:
        return float(np.linalg.norm(point - line_start))

    t = max(0.0, min(1.0, float(np.dot(point_vec, line_vec) / line_length_sq)))
    closest_point = line_start + t * line_vec
    return float(np.linalg.norm(point - closest_point))


def check_robot_collision(
    robot,
    obstacles: List[geometry.Mesh],
    *,
    safety_margin: float = 0.05,
    ignore_gold: bool = True,
) -> Tuple[bool, List[Dict[str, np.ndarray]]]:
    """Detect collisions between robot links and environment obstacles."""
    collisions: List[Dict[str, np.ndarray]] = []

    link_positions: List[np.ndarray] = []
    for i in range(robot.n + 1):
        if i == 0:
            T = asSE3(robot.base)
        else:
            T = asSE3(robot.fkine(robot.q, end=robot.links[i - 1]))
        link_positions.append(T.t.copy())

    for idx in range(len(link_positions) - 1):
        start = link_positions[idx]
        end = link_positions[idx + 1]

        for obs_idx, obstacle in enumerate(obstacles):
            color = getattr(obstacle, "color", None)
            if ignore_gold and color is not None and len(color) >= 3:
                if color[0] >= 0.9 and color[1] >= 0.8:
                    continue

            obs_pose = asSE3(obstacle)
            obs_pos = obs_pose.t
            dist = point_to_line_segment_distance(obs_pos, start, end)

            obstacle_radius = 0.1
            if dist < (obstacle_radius + safety_margin):
                collisions.append(
                    {
                        "link": idx,
                        "obstacle": obs_idx,
                        "distance": dist,
                        "link_start": start,
                        "link_end": end,
                        "obstacle_pos": obs_pos,
                    }
                )

    return (len(collisions) > 0), collisions


def check_robot_self_collision(
    robot, *, safety_margin: float = 0.05
) -> Tuple[bool, List[Dict[str, float]]]:
    """Rudimentary self-collision detection via link midpoint distances."""
    collisions: List[Dict[str, float]] = []

    link_positions: List[np.ndarray] = []
    for i in range(robot.n + 1):
        if i == 0:
            T = asSE3(robot.base)
        else:
            T = asSE3(robot.fkine(robot.q, end=robot.links[i - 1]))
        link_positions.append(T.t.copy())

    for i in range(len(link_positions) - 1):
        for j in range(i + 2, len(link_positions) - 1):
            mid1 = (link_positions[i] + link_positions[i + 1]) / 2.0
            mid2 = (link_positions[j] + link_positions[j + 1]) / 2.0
            dist = float(np.linalg.norm(mid1 - mid2))
            link_radius = 0.05

            if dist < (2.0 * link_radius + safety_margin):
                collisions.append({"link1": i, "link2": j, "distance": dist})

    return (len(collisions) > 0), collisions


def _bucket_drop_pose(bucket: geometry.Mesh, depth: float = 0.1, x_offset: float = 0.0) -> SE3:
    """Return a pose inside the bucket by moving inside along local +Z."""
    bucketT = asSE3(bucket)
    return bucketT * SE3(x_offset, 0, depth)


def _set_mesh_pose(mesh, T: SE3):
    if hasattr(mesh, "T"):
        mesh.T = T
    elif hasattr(mesh, "pose"):
        mesh.pose = T
    elif hasattr(mesh, "A"):
        mesh.A = T.A


MOVEJ_RUNTIME_OPTIONS: Dict[str, object] = {}


def configure_movej_runtime(**kwargs):
    """Update global runtime options for movej safety hooks."""
    for key, value in kwargs.items():
        if value is None:
            MOVEJ_RUNTIME_OPTIONS.pop(key, None)
        else:
            MOVEJ_RUNTIME_OPTIONS[key] = value


def ik_posonly(
    robot,
    T_goal: SE3,
    q0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    ilimit: int = 300,
):
    qseed = np.array(robot.q if q0 is None else q0, dtype=float)
    mask = [1, 1, 1, 0, 0, 0]
    try:
        return robot.ikine_LM(
            T_goal, q0=qseed, mask=mask, joint_limits=True, ilimit=ilimit, slimit=100, tol=tol
        )
    except Exception:
        return type("Dummy", (), {"success": False})()


def movej(
    robot,
    q_target,
    env,
    *,
    T: float = 1.6,
    steps: int = 70,
    carried: Optional[geometry.Mesh] = None,
    attach_offset: Optional[SE3] = None,
    estop_check: Optional[Callable[[], bool]] = None,
    obstacles: Optional[List[geometry.Mesh]] = None,
    env_collision_fn: Optional[Callable[[object, List[geometry.Mesh]], Tuple[bool, List[Dict[str, np.ndarray]]]]] = None,
    self_collision_fn: Optional[Callable[[object], Tuple[bool, List[Dict[str, float]]]]] = None,
    halt_on_collision: Optional[bool] = None,
) -> bool:
    opts = MOVEJ_RUNTIME_OPTIONS
    if estop_check is None:
        estop_check = opts.get("estop_check")
    if obstacles is None:
        obstacles = opts.get("obstacles")
    if env_collision_fn is None:
        env_collision_fn = opts.get("env_collision_fn")
    if self_collision_fn is None:
        self_collision_fn = opts.get("self_collision_fn")
    if halt_on_collision is None:
        halt_on_collision = opts.get("halt_on_collision", False)

    q0 = np.array(robot.q, dtype=float)
    q1 = np.array(q_target, dtype=float)
    traj = rtb.jtraj(q0, q1, steps).q
    dt = max(T / max(1, steps - 1), 1e-3)
    collision_stop_after = max(1, int(opts.get("collision_stop_after", 3)))
    env_collision_streak = 0
    self_collision_streak = 0

    for step_idx, q in enumerate(traj):
        if estop_check and estop_check():
            print("[MOVEJ] Emergency stop triggered; halting trajectory")
            return False

        robot.q = q

        collided = False

        env_hit = False
        if env_collision_fn and obstacles:
            env_hit, _ = env_collision_fn(robot, obstacles)
            if env_hit:
                env_collision_streak += 1
                if env_collision_streak == 1:
                    print("[MOVEJ] Environment collision detected")
                if halt_on_collision and env_collision_streak >= collision_stop_after:
                    print("[MOVEJ] Environment collision detected repeatedly; stopping trajectory")
                    collided = True
            else:
                env_collision_streak = 0
        else:
            env_collision_streak = 0

        self_hit = False
        if self_collision_fn:
            self_hit, _ = self_collision_fn(robot)
            if self_hit:
                self_collision_streak += 1
                if self_collision_streak == 1:
                    print("[MOVEJ] Self-collision detected")
                if halt_on_collision and self_collision_streak >= collision_stop_after:
                    print("[MOVEJ] Self-collision detected repeatedly; stopping trajectory")
                    collided = True
            else:
                self_collision_streak = 0
        else:
            self_collision_streak = 0

        if collided:
            print("[MOVEJ] Trajectory stopped due to collision")
            return False

        if carried is not None and attach_offset is not None:
            _set_mesh_pose(carried, robot.fkine(q) * attach_offset)
        env.step(dt)

    return True
