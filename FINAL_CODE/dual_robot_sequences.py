"""Dual-robot visual servoing sequences."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as geometry
from spatialmath import SE3

from camera import VirtualCamera
from helpers import (
    _bucket_drop_pose,
    _set_mesh_pose,
    asSE3,
    dist_xy,
    ik_posonly,
    movej,
)


def run_dual_robot_visual_servo(
    env,
    robot1,
    camera1: VirtualCamera,
    robot2,
    camera2: VirtualCamera,
    assets,
    *,
    attach_offset_fanuc: Optional[SE3] = None,
    attach_offset_parol: Optional[SE3] = None,
) -> Tuple[List[geometry.Mesh], List[geometry.Mesh]]:
    """Sequentially coordinate Fanuc and PAROL6 robots to sort objects."""

    attach_offset_fanuc = SE3(0, 0, 0.08) if attach_offset_fanuc is None else attach_offset_fanuc
    attach_offset_parol = SE3() if attach_offset_parol is None else attach_offset_parol

    q_home1 = np.array(robot1.q, dtype=float)
    q_home2 = np.array(robot2.q, dtype=float)
    baseT2 = asSE3(robot2.base)
    buckets = assets["buckets"]

    TABLE_DROP_POS = SE3(0.0, -0.1, 0.23)

    remaining = [(m, "gold") for m in assets["gold"]]
    remaining += [(m, "trash") for m in assets["trash"]]

    moved1: List[geometry.Mesh] = []
    moved2: List[geometry.Mesh] = []

    print("\n[SEQUENTIAL ROBOT] Starting sequential operation...")
    print("[SEQUENTIAL ROBOT] PAROL6 will place objects on table")
    print("[SEQUENTIAL ROBOT] Fanuc will then sort them to buckets\n")

    q_fanuc_wait = np.array([0, -np.pi / 4, np.pi / 3, 0, np.pi / 4, 0])
    print("[FANUC] Moving to waiting position near table...")
    movej(robot1, q_fanuc_wait, env, T=2.0, steps=80)

    q_parol_search = np.array([np.pi / 4, -np.pi / 4, np.pi / 3, 0, np.pi / 4, 0])
    q_parol_clear = q_parol_search.copy()
    print("[PAROL6] Moving to search position...")
    movej(robot2, q_parol_search, env, T=2.0, steps=80)

    object_counter = 1

    while remaining:
        q_parol_return_pose = None
        parol_moved_out = False

        print("\n" + "=" * 60)
        print(
            f"[CYCLE {object_counter}] Processing object {object_counter}/{len(remaining) + object_counter - 1}"
        )
        print("=" * 60)

        print("\n[PHASE 1] PAROL6 - Finding and picking object...")

        available = [m for m, _ in remaining]
        visible_parol = camera2.detect_objects(available, max_distance=1.5)

        if not visible_parol:
            print("[PAROL6] No objects visible, searching...")
            search_poses = [
                SE3(0.0, 0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
                SE3(-0.3, 0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
                SE3(0.3, 0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
                SE3(0.0, 0.5, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
            ]

            for pose in search_poses:
                sol = ik_posonly(robot2, pose, ilimit=150)
                if sol.success:
                    movej(robot2, sol.q, env, T=1.5, steps=60)
                    visible_parol = camera2.detect_objects(available, max_distance=1.5)
                    if visible_parol:
                        break

            if not visible_parol:
                print("[PAROL6] No objects found after searching. Mission complete.")
                break

        target_obj, _, _, _ = min(visible_parol, key=lambda x: dist_xy(x[0], baseT2))

        target_tuple = next(((mesh, k) for mesh, k in remaining if mesh is target_obj), None)
        if target_tuple is None:
            break

        target_kind = target_tuple[1]
        obj_pos = asSE3(target_obj).t
        print(f"[PAROL6] Targeting {target_kind} at {obj_pos.round(3)}")

        T_approach = SE3(obj_pos[0], obj_pos[1], obj_pos[2] + 0.18) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_approach, ilimit=150)
        if not sol.success:
            print(f"[PAROL6] Cannot reach {target_kind}, skipping")
            remaining.remove(target_tuple)
            continue

        movej(robot2, sol.q, env, T=1.5, steps=60)

        T_grasp = SE3(obj_pos[0], obj_pos[1], obj_pos[2] + 0.03) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_grasp, ilimit=150)
        if not sol.success:
            print(f"[PAROL6] Cannot grasp {target_kind}, skipping")
            remaining.remove(target_tuple)
            continue

        movej(robot2, sol.q, env, T=1.0, steps=40)

        carried_parol = target_obj
        _set_mesh_pose(carried_parol, robot2.fkine(robot2.q) * attach_offset_parol)
        env.step(0.01)
        print(f"[PAROL6] Grasped {target_kind}!")

        T_lift = robot2.fkine(robot2.q) * SE3(0, 0, 0.15)
        sol = ik_posonly(robot2, T_lift, ilimit=150)
        if sol.success:
            movej(robot2, sol.q, env, T=0.8, steps=40, carried=carried_parol, attach_offset=attach_offset_parol)

        print(f"[PAROL6] Transporting {target_kind} to table...")
        T_table_above = SE3(0.0, -0.1, 0.23 + 0.15) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_table_above, ilimit=150)
        if not sol.success:
            print("[PAROL6] Cannot reach table, skipping")
            remaining.remove(target_tuple)
            continue

        movej(robot2, sol.q, env, T=1.5, steps=60, carried=carried_parol, attach_offset=attach_offset_parol)

        T_table_drop = SE3(0.0, -0.1, 0.23) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_table_drop, ilimit=150)
        if sol.success:
            movej(robot2, sol.q, env, T=0.8, steps=40, carried=carried_parol, attach_offset=attach_offset_parol)

        _set_mesh_pose(carried_parol, TABLE_DROP_POS)
        print(f"[PAROL6] Placed {target_kind} on table at (0, -0.1, 0.23)")
        moved2.append(carried_parol)

        sol = ik_posonly(robot2, T_table_above, ilimit=150)
        if sol.success:
            movej(robot2, sol.q, env, T=0.8, steps=40)
            q_parol_return_pose = np.array(robot2.q, dtype=float)
            if not np.allclose(robot2.q, q_parol_clear, atol=1e-4):
                if movej(robot2, q_parol_clear, env, T=1.0, steps=40):
                    parol_moved_out = True

        print("\n[PHASE 2] FANUC - Picking from table and sorting to bucket...")

        T_fanuc_above_table = SE3(0.0, -0.1, 0.23 + 0.15) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_fanuc_above_table, ilimit=150)
        if not sol.success:
            print("[FANUC] Cannot reach table position")
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        movej(robot1, sol.q, env, T=1.5, steps=60)

        T_fanuc_grasp_table = SE3(0.0, -0.1, 0.23) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_fanuc_grasp_table, ilimit=150)
        if not sol.success:
            print("[FANUC] Cannot grasp from table")
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        movej(robot1, sol.q, env, T=1.0, steps=40)

        carried_fanuc = carried_parol
        _set_mesh_pose(carried_fanuc, robot1.fkine(robot1.q) * attach_offset_fanuc)
        env.step(0.01)
        print(f"[FANUC] Grasped {target_kind} from table!")

        T_fanuc_lift = robot1.fkine(robot1.q) * SE3(0, 0, 0.15)
        sol = ik_posonly(robot1, T_fanuc_lift, ilimit=150)
        if sol.success:
            movej(robot1, sol.q, env, T=0.8, steps=40, carried=carried_fanuc, attach_offset=attach_offset_fanuc)

        if target_kind == "gold":
            bucket = buckets[0]
            print(f"[FANUC] Sorting GOLD to bucket at {asSE3(bucket).t.round(3)}")
        else:
            bucket = buckets[1]
            print(f"[FANUC] Sorting TRASH to bucket at {asSE3(bucket).t.round(3)}")

        if not bucket:
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        bucketT = asSE3(bucket)
        bucket_drop_target = _bucket_drop_pose(bucket, depth=0.1, x_offset=0.05)
        drop_point = bucket_drop_target.t
        print(f"[FANUC] Moving to bucket at {bucketT.t.round(3)}")

        T_bucket_above = SE3(drop_point[0], drop_point[1], bucketT.t[2] + 0.18) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_bucket_above, ilimit=150)
        if not sol.success:
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        movej(robot1, sol.q, env, T=1.5, steps=60, carried=carried_fanuc, attach_offset=attach_offset_fanuc)

        T_bucket_drop = SE3(drop_point[0], drop_point[1], drop_point[2] + 0.03) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_bucket_drop, ilimit=150)
        if sol.success:
            movej(robot1, sol.q, env, T=0.8, steps=40, carried=carried_fanuc, attach_offset=attach_offset_fanuc)

        _set_mesh_pose(carried_fanuc, bucket_drop_target)
        print(f"[FANUC] Dropped {target_kind} in bucket!")
        moved1.append(carried_fanuc)

        sol = ik_posonly(robot1, T_bucket_above, ilimit=150)
        if sol.success:
            movej(robot1, sol.q, env, T=0.8, steps=40)

        movej(robot1, q_fanuc_wait, env, T=1.5, steps=60)

        if parol_moved_out and q_parol_return_pose is not None:
            movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)

        remaining.remove(target_tuple)
        object_counter += 1
        print(f"\n[PROGRESS] {object_counter - 1} objects sorted, {len(remaining)} remaining")

    print("\n[SEQUENTIAL ROBOT] All objects processed!")
    print("[SEQUENTIAL ROBOT] Returning both robots to home positions...")

    steps = 100
    traj1 = rtb.jtraj(robot1.q, q_home1, steps).q
    traj2 = rtb.jtraj(robot2.q, q_home2, steps).q
    for i in range(steps):
        robot1.q = traj1[i]
        robot2.q = traj2[i]
        env.step(0.02)

    print("[SEQUENTIAL ROBOT] Both robots returned home.")
    return moved1, moved2


def run_dual_robot_visual_servo_with_tracking(
    env,
    robot1,
    camera1: VirtualCamera,
    robot2,
    camera2: VirtualCamera,
    assets,
    *,
    attach_offset_fanuc: Optional[SE3] = None,
    attach_offset_parol: Optional[SE3] = None,
    demo=None,
) -> Tuple[List[geometry.Mesh], List[geometry.Mesh]]:
    """Variant that informs the GUI about carried objects for E-stop safety."""

    attach_offset_fanuc = SE3(0, 0, 0.08) if attach_offset_fanuc is None else attach_offset_fanuc
    attach_offset_parol = SE3() if attach_offset_parol is None else attach_offset_parol

    q_home1 = np.array(robot1.q, dtype=float)
    q_home2 = np.array(robot2.q, dtype=float)
    baseT2 = asSE3(robot2.base)
    buckets = assets["buckets"]

    TABLE_DROP_POS = SE3(0.0, -0.1, 0.23)

    remaining = [(m, "gold") for m in assets["gold"]]
    remaining += [(m, "trash") for m in assets["trash"]]

    moved1: List[geometry.Mesh] = []
    moved2: List[geometry.Mesh] = []

    print("\n[SEQUENTIAL ROBOT] Starting sequential operation...")
    print("[SEQUENTIAL ROBOT] PAROL6 will place objects on table")
    print("[SEQUENTIAL ROBOT] Fanuc will then sort them to buckets\n")

    q_fanuc_wait = np.array([0, -np.pi / 4, np.pi / 3, 0, np.pi / 4, 0])
    print("[FANUC] Moving to waiting position near table...")
    movej(robot1, q_fanuc_wait, env, T=2.0, steps=80)

    q_parol_search = np.array([np.pi / 4, -np.pi / 4, np.pi / 3, 0, np.pi / 4, 0])
    q_parol_clear = q_parol_search.copy()
    print("[PAROL6] Moving to search position...")
    movej(robot2, q_parol_search, env, T=2.0, steps=80)

    object_counter = 1

    while remaining:
        q_parol_return_pose = None
        parol_moved_out = False

        print("\n" + "=" * 60)
        print(
            f"[CYCLE {object_counter}] Processing object {object_counter}/{len(remaining) + object_counter - 1}"
        )
        print("=" * 60)

        print("\n[PHASE 1] PAROL6 - Finding and picking object...")

        available = [m for m, _ in remaining]
        visible_parol = camera2.detect_objects(available, max_distance=1.5)

        if not visible_parol:
            print("[PAROL6] No objects visible, searching...")
            search_poses = [
                SE3(0.0, 0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
                SE3(-0.3, 0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
                SE3(0.3, 0.3, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
                SE3(0.0, 0.5, 0.5) * SE3.RPY([0, np.pi * 0.7, 0]),
            ]

            for pose in search_poses:
                sol = ik_posonly(robot2, pose, ilimit=150)
                if sol.success:
                    movej(robot2, sol.q, env, T=1.5, steps=60)
                    visible_parol = camera2.detect_objects(available, max_distance=1.5)
                    if visible_parol:
                        break

            if not visible_parol:
                print("[PAROL6] No objects found after searching. Mission complete.")
                break

        target_obj, _, _, _ = min(visible_parol, key=lambda x: dist_xy(x[0], baseT2))

        target_tuple = next(((mesh, k) for mesh, k in remaining if mesh is target_obj), None)
        if target_tuple is None:
            break

        target_kind = target_tuple[1]
        obj_pos = asSE3(target_obj).t
        print(f"[PAROL6] Targeting {target_kind} at {obj_pos.round(3)}")

        T_approach = SE3(obj_pos[0], obj_pos[1], obj_pos[2] + 0.18) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_approach, ilimit=150)
        if not sol.success:
            print(f"[PAROL6] Cannot reach {target_kind}, skipping")
            remaining.remove(target_tuple)
            continue

        movej(robot2, sol.q, env, T=1.5, steps=60)

        T_grasp = SE3(obj_pos[0], obj_pos[1], obj_pos[2] + 0.03) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_grasp, ilimit=150)
        if not sol.success:
            print(f"[PAROL6] Cannot grasp {target_kind}, skipping")
            remaining.remove(target_tuple)
            continue

        movej(robot2, sol.q, env, T=1.0, steps=40)

        carried_parol = target_obj
        _set_mesh_pose(carried_parol, robot2.fkine(robot2.q) * attach_offset_parol)
        if demo:
            demo.set_carried_object(0, carried_parol, attach_offset_parol)
        env.step(0.01)
        print(f"[PAROL6] Grasped {target_kind}!")

        T_lift = robot2.fkine(robot2.q) * SE3(0, 0, 0.15)
        sol = ik_posonly(robot2, T_lift, ilimit=150)
        if sol.success:
            movej(robot2, sol.q, env, T=0.8, steps=40, carried=carried_parol, attach_offset=attach_offset_parol)

        print(f"[PAROL6] Transporting {target_kind} to table...")
        T_table_above = SE3(0.0, -0.1, 0.23 + 0.15) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_table_above, ilimit=150)
        if not sol.success:
            if demo:
                demo.set_carried_object(0, None, attach_offset_parol)
            print("[PAROL6] Cannot reach table, skipping")
            remaining.remove(target_tuple)
            continue

        movej(robot2, sol.q, env, T=1.5, steps=60, carried=carried_parol, attach_offset=attach_offset_parol)

        T_table_drop = SE3(0.0, -0.1, 0.23) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot2, T_table_drop, ilimit=150)
        if sol.success:
            movej(robot2, sol.q, env, T=0.8, steps=40, carried=carried_parol, attach_offset=attach_offset_parol)

        _set_mesh_pose(carried_parol, TABLE_DROP_POS)
        if demo:
            demo.set_carried_object(0, None, attach_offset_parol)
        print(f"[PAROL6] Placed {target_kind} on table at (0, -0.1, 0.23)")
        moved2.append(carried_parol)

        sol = ik_posonly(robot2, T_table_above, ilimit=150)
        if sol.success:
            movej(robot2, sol.q, env, T=0.8, steps=40)
            q_parol_return_pose = np.array(robot2.q, dtype=float)
            if not np.allclose(robot2.q, q_parol_clear, atol=1e-4):
                if movej(robot2, q_parol_clear, env, T=1.0, steps=40):
                    parol_moved_out = True

        print("\n[PHASE 2] FANUC - Picking from table and sorting to bucket...")

        T_fanuc_above_table = SE3(0.0, -0.1, 0.23 + 0.15) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_fanuc_above_table, ilimit=150)
        if not sol.success:
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        movej(robot1, sol.q, env, T=1.5, steps=60)

        T_fanuc_grasp_table = SE3(0.0, -0.1, 0.23) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_fanuc_grasp_table, ilimit=150)
        if not sol.success:
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        movej(robot1, sol.q, env, T=1.0, steps=40)

        carried_fanuc = carried_parol
        _set_mesh_pose(carried_fanuc, robot1.fkine(robot1.q) * attach_offset_fanuc)
        if demo:
            demo.set_carried_object(1, carried_fanuc, attach_offset_fanuc)
        env.step(0.01)
        print(f"[FANUC] Grasped {target_kind} from table!")

        T_fanuc_lift = robot1.fkine(robot1.q) * SE3(0, 0, 0.15)
        sol = ik_posonly(robot1, T_fanuc_lift, ilimit=150)
        if sol.success:
            movej(robot1, sol.q, env, T=0.8, steps=40, carried=carried_fanuc, attach_offset=attach_offset_fanuc)

        if target_kind == "gold":
            bucket = buckets[0]
            print(f"[FANUC] Sorting GOLD to bucket at {asSE3(bucket).t.round(3)}")
        else:
            bucket = buckets[1]
            print(f"[FANUC] Sorting TRASH to bucket at {asSE3(bucket).t.round(3)}")

        if not bucket:
            if demo:
                demo.set_carried_object(1, None, attach_offset_fanuc)
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        bucketT = asSE3(bucket)
        bucket_drop_target = _bucket_drop_pose(bucket, depth=0.1, x_offset=0.05)
        drop_point = bucket_drop_target.t
        print(f"[FANUC] Moving to bucket at {bucketT.t.round(3)}")

        T_bucket_above = SE3(drop_point[0], drop_point[1], bucketT.t[2] + 0.18) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_bucket_above, ilimit=150)
        if not sol.success:
            if demo:
                demo.set_carried_object(1, None, attach_offset_fanuc)
            remaining.remove(target_tuple)
            if parol_moved_out and q_parol_return_pose is not None:
                movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)
            continue

        movej(robot1, sol.q, env, T=1.5, steps=60, carried=carried_fanuc, attach_offset=attach_offset_fanuc)

        T_bucket_drop = SE3(drop_point[0], drop_point[1], drop_point[2] + 0.03) * SE3.RPY([0, np.pi, 0])
        sol = ik_posonly(robot1, T_bucket_drop, ilimit=150)
        if sol.success:
            movej(robot1, sol.q, env, T=0.8, steps=40, carried=carried_fanuc, attach_offset=attach_offset_fanuc)

        _set_mesh_pose(carried_fanuc, bucket_drop_target)
        if demo:
            demo.set_carried_object(1, None, attach_offset_fanuc)
        print(f"[FANUC] Dropped {target_kind} in bucket!")
        moved1.append(carried_fanuc)

        sol = ik_posonly(robot1, T_bucket_above, ilimit=150)
        if sol.success:
            movej(robot1, sol.q, env, T=0.8, steps=40)

        movej(robot1, q_fanuc_wait, env, T=1.5, steps=60)

        if parol_moved_out and q_parol_return_pose is not None:
            movej(robot2, q_parol_return_pose, env, T=0.8, steps=40)

        remaining.remove(target_tuple)
        object_counter += 1
        print(f"\n[PROGRESS] {object_counter - 1} objects sorted, {len(remaining)} remaining")

    print("\n[SEQUENTIAL ROBOT] All objects processed!")
    print("[SEQUENTIAL ROBOT] Returning both robots to home positions...")

    steps = 100
    traj1 = rtb.jtraj(robot1.q, q_home1, steps).q
    traj2 = rtb.jtraj(robot2.q, q_home2, steps).q
    for i in range(steps):
        robot1.q = traj1[i]
        robot2.q = traj2[i]
        env.step(0.02)

    print("[SEQUENTIAL ROBOT] Both robots returned home.")
    return moved1, moved2


__all__ = [
    "run_dual_robot_visual_servo",
    "run_dual_robot_visual_servo_with_tracking",
]
