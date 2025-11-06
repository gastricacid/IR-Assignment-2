"""Environment and STL asset utilities for the visual servoing demo."""

import os
import sys
import pathlib
from math import pi
from typing import Dict, List, Optional, Tuple

import spatialgeometry as geometry
from spatialmath import SE3

_HERE = pathlib.Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent  # .../Standard200IC
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from ir_support.robots.liambot.lrmate200ic_3d_armonly import LRMate200iC
from ir_support.robots.PAROL6 import PAROL6

Color = Tuple[float, float, float, float]


def _safe_add_mesh(
    env, stl_path: str, *, pose: SE3, scale, color: Color, strict: bool = False
):
    if not os.path.exists(stl_path):
        msg = f"[MISSING] STL not found: {stl_path}"
        print(msg)
        if strict:
            raise FileNotFoundError(msg)
        return None
    m = geometry.Mesh(stl_path, pose=pose, scale=scale, color=color)
    env.add(m)
    return m


def spawn_environment(
    env, current_dir: str, *, strict: bool = False
) -> Dict[str, List[geometry.Mesh]]:
    results: Dict[str, List[geometry.Mesh]] = {
        "gold": [],
        "trash": [],
        "buckets": [],
        "furniture": [],
        "extras": [],
    }

    colors: Dict[str, Color] = {
        "bed": (0.45, 0.70, 1.00, 1.0),
        "chair": (1.00, 0.60, 0.20, 1.0),
        "desk": (0.30, 0.30, 0.35, 1.0),
        "table": (0.20, 0.70, 0.30, 1.0),
        "table_wood": (0.55, 0.27, 0.07, 1.0),
        "gold": (1.0, 0.85, 0.2, 1.0),
        "trash": (0.4, 0.4, 0.4, 1.0),
        "bucket_gold": (1.0, 0.85, 0.2, 1.0),
        "bucket_grey": (0.5, 0.5, 0.5, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
    }

    bucket_positions = [
        SE3(-0.45, -0.55, 0.0),
        SE3(0.25, -0.55, 0.0),
    ]
    bucket_stl = os.path.join(current_dir, "Bucket.stl")
    for i, pose in enumerate(bucket_positions):
        color = colors["bucket_gold"] if i == 0 else colors["bucket_grey"]
        m = _safe_add_mesh(
            env,
            bucket_stl,
            pose=pose * SE3.Rx(pi / 2),
            scale=[0.0035] * 3,
            color=color,
            strict=strict,
        )
        if m is not None:
            results["buckets"].append(m)

    object_poses = {
        "bed": (SE3(0.5, -0.45, 0.0) * SE3.Rz(pi) * SE3.Rx(pi / 2), [0.00035, 0.00035, 0.00035]),
    }
    for name in ["bed"]:
        stl = os.path.join(current_dir, f"{name}.stl")
        pose, scale = object_poses[name]
        m = _safe_add_mesh(env, stl, pose=pose, scale=scale, color=colors[name], strict=strict)
        if m is not None:
            results["furniture"].append(m)

    table2_pose = SE3(2.5, 0.1, 0.0) * SE3.Rz(-pi / 2)
    table2_stl = os.path.join(current_dir, "table.stl")
    table2 = _safe_add_mesh(
        env,
        table2_stl,
        pose=table2_pose,
        scale=[0.005] * 3,
        color=colors["table_wood"],
        strict=strict,
    )
    if table2 is not None:
        results["furniture"].append(table2)
        button_stl = os.path.join(current_dir, "button.stl")
        fire_stl = os.path.join(current_dir, "fire.stl")
        button_pose = table2_pose * SE3(0, -0.1, 0.32)
        fire_pose = table2_pose * SE3(-0.525, 0.0, 0.05) * SE3.Rx(pi / 2)
        m = _safe_add_mesh(
            env,
            button_stl,
            pose=button_pose,
            scale=[0.01] * 3,
            color=colors["red"],
            strict=strict,
        )
        if m is not None:
            results["extras"].append(m)
        m = _safe_add_mesh(
            env,
            fire_stl,
            pose=fire_pose,
            scale=[0.0006] * 3,
            color=colors["red"],
            strict=strict,
        )
        if m is not None:
            results["extras"].append(m)

    gold_positions = [
        SE3(-0.25, 0.38, 0.0),
        SE3(0.0, 0.5, 0.0),
        SE3(0.25, 0.4, 0.0),
    ]
    trash_positions = [
        SE3(-0.3, 0.19, 0.0),
        SE3(0.13, 0.45, 0.0),
        SE3(0.28, 0.17, 0.0),
    ]
    gold_stl = os.path.join(current_dir, "gold.stl")
    trash_stl = os.path.join(current_dir, "trash.stl")
    for pose in gold_positions:
        m = _safe_add_mesh(
            env,
            gold_stl,
            pose=pose,
            scale=[0.0007] * 3,
            color=colors["gold"],
            strict=strict,
        )
        if m is not None:
            results["gold"].append(m)
    for pose in trash_positions:
        m = _safe_add_mesh(
            env,
            trash_stl,
            pose=pose,
            scale=[0.0007] * 3,
            color=colors["trash"],
            strict=strict,
        )
        if m is not None:
            results["trash"].append(m)

    env.step(0.01)
    print(
        f"[SPAWN] gold={len(results['gold'])}, trash={len(results['trash'])}, "
        f"buckets={len(results['buckets'])}, furniture={len(results['furniture'])}, extras={len(results['extras'])}"
    )
    return results


def spawn_robot(env, *, base=SE3(0.0, 0.0, 0.0), rz=0.0):
    robot = LRMate200iC()
    robot.base = base * SE3.Rz(rz)
    robot.add_to_env(env)
    return robot


def spawn_parol6(env, *, base=SE3(0, 0.2, 0.0), rz=-pi / 2):
    robot = PAROL6()
    robot.base = base * SE3.Rz(rz)
    robot.add_to_env(env)
    return robot
