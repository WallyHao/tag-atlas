"""SE(3) and AprilTag geometry helpers."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .models import FloatArray, Pose

logger = logging.getLogger(__name__)


def tag_object_points(size: float) -> FloatArray:
    """Return tag corners in the detector's counter-clockwise order."""

    if not np.isfinite(size) or size <= 0.0:
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
    if object_values.ndim != 2 or object_values.shape[1:] != (3,):
        raise ValueError("object_points must have shape (N, 3)")
    if image_values.ndim != 2 or image_values.shape[1:] != (2,):
        raise ValueError("image_points must have shape (N, 2)")
    if object_values.shape[0] != image_values.shape[0] or object_values.shape[0] < 4:
        raise ValueError("PnP requires at least four matching points")
    if not np.isfinite(object_values).all() or not np.isfinite(image_values).all():
        raise ValueError("PnP points must contain finite values")
    pnp_matrix: FloatArray = np.asarray(camera_matrix, dtype=np.float64)
    pnp_distortion: FloatArray = np.asarray(
        distortion_coefficients, dtype=np.float64
    ).reshape(-1)
    if pnp_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if not np.isfinite(pnp_matrix).all() or not np.isfinite(pnp_distortion).all():
        raise ValueError("camera calibration must contain finite values")

    for flag in (cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
        pose = _solve_pnp(
            object_values,
            image_values,
            pnp_matrix,
            pnp_distortion,
            flag,
        )
        if pose is not None:
            return pose
    logger.debug("PnP failed for %d correspondences", object_values.shape[0])
    return None


def _solve_pnp(
    object_points: FloatArray,
    image_points: FloatArray,
    camera_matrix: FloatArray,
    distortion_coefficients: FloatArray,
    flag: int,
) -> Pose | None:
    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=flag,
        )
    except cv2.error:
        logger.debug("OpenCV PnP solver failed for flag %d", flag)
        return None
    if not success:
        return None
    try:
        rotation, _ = cv2.Rodrigues(rvec)
    except cv2.error:
        return None
    rotation_value: FloatArray = np.asarray(rotation, dtype=np.float64)
    translation_value: FloatArray = np.asarray(tvec, dtype=np.float64).reshape(3)
    points_camera = object_points @ rotation_value.T + translation_value
    if not np.isfinite(points_camera).all() or np.any(points_camera[:, 2] <= 1e-9):
        logger.debug("OpenCV PnP returned a pose with invalid point depth")
        return None
    return Pose(rotation_value, translation_value)
