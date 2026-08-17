"""PnP-based pose initialization for the localization pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .camera import CameraModel
from .geometry import (
    compose_pose,
    inverse_pose,
    pose_from_pnp,
    tag_object_points,
    transform_points,
)
from .models import Detection, FloatArray, Pose


def seed_camera_pose(
    detections: Sequence[Detection],
    known_tag_poses: Mapping[int, Pose],
    tag_sizes: Mapping[int, float],
    camera: CameraModel,
    fallback: Pose | None = None,
) -> Pose | None:
    """Estimate the world-frame camera pose from mapped tag detections."""

    world_points: list[FloatArray] = []
    image_points: list[FloatArray] = []
    for detection in detections:
        tag_pose = known_tag_poses.get(detection.tag_id)
        if tag_pose is None:
            continue
        local_points = tag_object_points(tag_sizes[detection.tag_id])
        world_points.append(transform_points(tag_pose, local_points))
        image_points.append(detection.corners)
    if not world_points:
        return None
    assert camera.distortion_coefficients is not None
    camera_from_world = pose_from_pnp(
        np.vstack(world_points),
        np.vstack(image_points),
        camera.matrix,
        camera.distortion_coefficients,
    )
    if camera_from_world is None:
        return fallback
    return inverse_pose(camera_from_world)


def seed_tag_pose(
    detection: Detection,
    camera_pose: Pose,
    tag_sizes: Mapping[int, float],
    camera: CameraModel,
) -> Pose | None:
    """Estimate a new tag's world-frame pose from one camera observation."""

    local_points = tag_object_points(tag_sizes[detection.tag_id])
    assert camera.distortion_coefficients is not None
    camera_from_tag = pose_from_pnp(
        local_points,
        detection.corners,
        camera.matrix,
        camera.distortion_coefficients,
    )
    if camera_from_tag is None:
        return None
    return compose_pose(camera_pose, camera_from_tag)
