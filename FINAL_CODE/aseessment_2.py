
"""Facade module for the visual servoing assessment."""

from __future__ import annotations

import importlib
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_THIS_DIR = _HERE.parent
_PKG_ROOT = _HERE.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT.parent))

from camera import VirtualCamera
from dual_robot_sequences import (
    run_dual_robot_visual_servo,
    run_dual_robot_visual_servo_with_tracking,
)
from ibvs import compute_image_jacobian, ibvs_control_law
from visual_servo_demo import VisualServoDemo, run_visual_servo_cli
from visual_servo_motion import (
    run_visual_servo_sort,
    visual_servo_approach,
    visual_servo_to_object,
)

__all__ = [
    "VirtualCamera",
    "compute_image_jacobian",
    "ibvs_control_law",
    "visual_servo_to_object",
    "visual_servo_approach",
    "run_visual_servo_sort",
    "run_dual_robot_visual_servo",
    "run_dual_robot_visual_servo_with_tracking",
    "VisualServoDemo",
    "run_visual_servo_cli",
]


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_visual_servo_cli()
    else:
        gui_module = importlib.import_module("visual_servo_gui")
        demo = VisualServoDemo()
        gui_module.build_gui(demo)
