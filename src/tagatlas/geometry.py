"""SE(3) and AprilTag geometry helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .models import FloatArray, Pose


def tag_object_points(size: float) -> FloatArray:
    """Return tag corners in the detector's counter-clockwise order."""

    if size <= 0.0:
        raise ValueError("tag size must be positive")
    half = size / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


def pose_from_matrix(matrix: FloatArray) -> Pose:
    """Convert a homogeneous matrix into a :class:`Pose`."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("matrix must be homogeneous")
    return Pose(value[:3, :3], value[:3, 3])


def inverse_pose(pose: Pose) -> Pose:
    """Return the inverse rigid transform."""

    rotation = pose.rotation.T
    return Pose(rotation, -rotation @ pose.translation)


def compose_pose(first: Pose, second: Pose) -> Pose:
    """Return ``first @ second``."""

    return Pose(
        first.rotation @ second.rotation,
        first.rotation @ second.translation + first.translation,
    )


def transform_points(pose: Pose, points: FloatArray) -> FloatArray:
    """Transform points using a rigid transform."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return values @ pose.rotation.T + pose.translation


def pose_from_pnp(
    object_points: FloatArray,
    image_points: FloatArray,
    camera_matrix: FloatArray,
    distortion_coefficients: FloatArray,
) -> Pose | None:
    """Estimate the object-to-camera pose from 2D-3D correspondences."""

    object_values = np.asarray(object_points, dtype=np.float64)
    image_values = np.asarray(image_points, dtype=np.float64)
    if object_values.shape[0] != image_values.shape[0] or object_values.shape[0] < 4:
        raise ValueError("PnP requires at least four matching points")
    pnp_matrix: FloatArray = np.asarray(camera_matrix, dtype=np.float64)
    pnp_distortion: FloatArray = np.asarray(distortion_coefficients, dtype=np.float64)

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_values,
            image_values,
            pnp_matrix,
            pnp_distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        success = False
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
    if not success:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_values,
                image_values,
                pnp_matrix,
                pnp_distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            return None
    if not success:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return Pose(
        np.asarray(rotation, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )
