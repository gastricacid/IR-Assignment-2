"""Image-based visual servo (IBVS) helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_image_jacobian(u: float, v: float, depth: float, focal_length: float) -> np.ndarray:
	"""Compute the 2x6 image Jacobian for a point feature."""

	f = float(focal_length)
	Z = float(depth)
	u = float(u)
	v = float(v)

	x = (u - 320.0) / f
	y = (v - 240.0) / f

	return np.array(
		[
			[-f / Z, 0.0, x / Z, x * y / f, -(f + x ** 2 / f), y],
			[0.0, -f / Z, y / Z, (f + y ** 2 / f), -x * y / f, -x],
		],
		dtype=float,
	)


def ibvs_control_law(
	current_features: np.ndarray,
	desired_features: np.ndarray,
	depths: np.ndarray,
	focal_length: float,
	*,
	lambda_gain: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return the desired camera twist and pixel error."""

	current_features = np.asarray(current_features, dtype=float).flatten()
	desired_features = np.asarray(desired_features, dtype=float).flatten()
	depths = np.asarray(depths, dtype=float).flatten()

	n_features = len(current_features) // 2
	L = []
	for i in range(n_features):
		u = float(current_features[2 * i])
		v = float(current_features[2 * i + 1])
		Z = float(depths[i])
		L.append(compute_image_jacobian(u, v, Z, focal_length))

	L_stack = np.vstack(L)
	error = desired_features - current_features
	L_pinv = np.linalg.pinv(L_stack)
	v_cam = lambda_gain * L_pinv @ error

	return v_cam, error


__all__ = ["compute_image_jacobian", "ibvs_control_law"]
