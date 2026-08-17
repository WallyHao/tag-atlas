"""Quality metrics for localization results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .camera import CameraModel
from .geometry import inverse_pose, tag_object_points, transform_points
from .models import Detection, FloatArray, Pose


def reprojection_rmse(
    camera_pose: Pose,
    detections: Sequence[Detection],
    tag_poses: Mapping[int, Pose],
    tag_sizes: Mapping[int, float],
    camera: CameraModel,
) -> float:
    """Return the root mean square pixel reprojection error."""

    camera_from_world = inverse_pose(camera_pose)
    errors: list[FloatArray] = []
    for detection in detections:
        errors.append(
            _reprojection_error(
                camera_from_world,
                detection,
                tag_poses,
                tag_sizes,
                camera,
            )
        )
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def detection_reprojection_rmse(
    camera_pose: Pose,
    detection: Detection,
    tag_poses: Mapping[int, Pose],
    tag_sizes: Mapping[int, float],
    camera: CameraModel,
) -> float:
    """Return the pixel RMSE for one Tag observation."""

    try:
        error = _reprojection_error(
            inverse_pose(camera_pose), detection, tag_poses, tag_sizes, camera
        )
    except ValueError:
        return float("inf")
    return float(np.sqrt(np.mean(error**2)))


def _reprojection_error(
    camera_from_world: Pose,
    detection: Detection,
    tag_poses: Mapping[int, Pose],
    tag_sizes: Mapping[int, float],
    camera: CameraModel,
) -> FloatArray:
    tag_pose = tag_poses[detection.tag_id]
    points_world = transform_points(
        tag_pose, tag_object_points(tag_sizes[detection.tag_id])
    )
    points_camera = transform_points(camera_from_world, points_world)
    predicted = camera.project(points_camera)
    return (predicted - detection.corners).reshape(-1)
