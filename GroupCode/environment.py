# scene_assets.py
import os
from math import pi
from typing import Dict, List, Tuple

from spatialmath import SE3
import spatialgeometry as geometry

try:
    import swift  
except Exception:
    swift = None  

Color = Tuple[float, float, float, float]

def _safe_add_mesh(env, stl_path: str, *, pose: SE3, scale, color: Color):
    """Add a mesh if the STL exists; warn otherwise. Returns the mesh or None."""
    if not os.path.exists(stl_path):
        print(f"[WARN] STL not found, skipping: {stl_path}")
        return None
    m = geometry.Mesh(stl_path, pose=pose, scale=scale, color=color)
    env.add(m)
    return m

def spawn_environment(env, current_dir: str) -> Dict[str, List[geometry.Mesh]]:

    results: Dict[str, List[geometry.Mesh]] = {
        "gold": [],
        "trash": [],
        "buckets": [],
        "furniture": [],
        "extras": [],
    }


    # Colors
  
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

    # Furniture transforms & scales

    object_poses = {
        "bed": (SE3(0.4, 0.3, 0.0) * SE3.Rz(pi) * SE3.Rx(pi/2), [0.00035, 0.00035, 0.00035]),
        "chair": (SE3(0, -0.3, 0.2) * SE3.Rz(-pi/2),           [0.006,   0.006,   0.006]),
        "desk": (SE3(-0.3, -0.25, 0.1) * SE3.Rz(pi/2),         [0.003,   0.003,   0.003]),
        "table": (SE3(0.55, -0.3, 0.0) * SE3.Rz(-pi/2),        [0.0035,  0.0035,  0.0035]),
    }


    # Buckets
 
    bucket_positions = [
        SE3(-0.2, 1.65, 0.0),
        SE3(0, -1.25, 0.0),
        SE3(1.25, 0, 0.0),
        SE3(-0.95, -0.78, 0.0),
    ]

    bucket_stl = os.path.join(current_dir, "Bucket.stl")
    for i, pose in enumerate(bucket_positions):
        color = colors["bucket_gold"] if i < 2 else colors["bucket_grey"]
        mesh = _safe_add_mesh(
            env,
            bucket_stl,
            pose=pose * SE3.Rx(pi/2),
            scale=[0.0035, 0.0035, 0.0035],
            color=color,
        )
        if mesh is not None:
            results["buckets"].append(mesh)


    # Furniture (bed, chair, desk, table)
  
    for name in ["bed", "chair", "desk", "table"]:
        stl = os.path.join(current_dir, f"{name}.stl")
        pose, scale = object_poses.get(name, (SE3(0, 0, 0), [0.003, 0.003, 0.003]))
        mesh = _safe_add_mesh(
            env,
            stl,
            pose=pose,
            scale=scale,
            color=colors.get(name, (0.8, 0.8, 0.8, 1.0)),
        )
        if mesh is not None:
            results["furniture"].append(mesh)


    # Second table (wood)

    table2_pose = SE3(2.5, 0.1, 0.0) * SE3.Rz(-pi/2)
    table2_stl = os.path.join(current_dir, "table.stl")
    table2_mesh = _safe_add_mesh(
        env,
        table2_stl,
        pose=table2_pose,
        scale=[0.005, 0.005, 0.005],
        color=colors["table_wood"],
    )
    if table2_mesh is not None:
        results["furniture"].append(table2_mesh)

        # Button & Fire on second table (optional)
        button_stl = os.path.join(current_dir, "button.stl")
        fire_stl = os.path.join(current_dir, "fire.stl")

        button_pose = table2_pose * SE3(0, -0.1, 0.32)
        fire_pose = table2_pose * SE3(-0.525, 0.0, 0.05) * SE3.Rx(pi/2)

        button_mesh = _safe_add_mesh(
            env,
            button_stl,
            pose=button_pose,
            scale=[0.01, 0.01, 0.01],
            color=colors["red"],
        )
        if button_mesh is not None:
            results["extras"].append(button_mesh)

        fire_mesh = _safe_add_mesh(
            env,
            fire_stl,
            pose=fire_pose,
            scale=[0.0006, 0.0006, 0.0006],
            color=colors["red"],
        )
        if fire_mesh is not None:
            results["extras"].append(fire_mesh)


    # Gold & Trash pieces

    gold_positions = [
        SE3(-0.4, -0.5, 0.3),
        SE3(0.4, -0.2, 0.23),
        SE3(0.2, 0.5, 0.2),
        SE3(0.4, 0.4, 0.0),
    ]
    trash_positions = [
        SE3(0.4, -0.3, 0.23),
        SE3(-0.4, -0.2, 0.3),
        SE3(-0.1, -0.35, 0.19),
        SE3(-0.4, 0.4, 0.2),
    ]

    gold_stl = os.path.join(current_dir, "gold.stl")
    trash_stl = os.path.join(current_dir, "trash.stl")

    for pose in gold_positions:
        m = _safe_add_mesh(
            env,
            gold_stl,
            pose=pose,
            scale=[0.0007, 0.0007, 0.0007],
            color=colors["gold"],
        )
        if m is not None:
            results["gold"].append(m)

    for pose in trash_positions:
        m = _safe_add_mesh(
            env,
            trash_stl,
            pose=pose,
            scale=[0.0007, 0.0007, 0.0007],
            color=colors["trash"],
        )
        if m is not None:
            results["trash"].append(m)

    # One small step to ensure Swift renders first frame smoothly
    env.step(0.01)

    return results

# Self-test
def _print_summary(assets: Dict[str, List[geometry.Mesh]]):
    print("\n[SCENE SUMMARY]")
    for k in ["furniture", "buckets", "gold", "trash", "extras"]:
        print(f"  {k:10s}: {len(assets[k])} item(s)")

def _run_self_test():
    if swift is None:
        raise RuntimeError(
            "swift is not available."
        )

    env = swift.Swift()
    env.launch(realtime=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] Using asset directory: {current_dir}")

    assets = spawn_environment(env, current_dir)
    _print_summary(assets)

    print("[INFO] Scene spawned. Close the Swift window or press Ctrl+C to exit.")
    try:
        env.hold()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("[INFO] Self-test finished.")

if __name__ == "__main__":
    _run_self_test()
