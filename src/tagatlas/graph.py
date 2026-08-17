"""GTSAM-backed pose graph for tag projection measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import CameraModel
from .geometry import tag_object_points
from .types import FloatArray, Pose

gtsam: Any = None
try:
    import gtsam as _gtsam
except ImportError:  # pragma: no cover - exercised when dependencies are missing.
    pass
else:
    gtsam = _gtsam


@dataclass(frozen=True)
class ProjectionConstraint:
    """One tag observation represented by four image-space corners."""

    camera_key: int
    tag_key: int
    tag_size: float
    image_corners: FloatArray

    def __post_init__(self) -> None:
        corners = np.asarray(self.image_corners, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError("image_corners must have shape (4, 2)")
        object.__setattr__(self, "image_corners", corners.copy())


def _residual(
    camera_pose: Any,
    tag_pose: Any,
    constraint: ProjectionConstraint,
    camera: CameraModel,
) -> FloatArray:
    """Evaluate a pixel residual for a GTSAM Pose3 pair."""

    camera_to_world = camera_pose
    tag_to_world = tag_pose
    points_tag = tag_object_points(constraint.tag_size)
    points_world = np.asarray(
        [tag_to_world.transformFrom(point) for point in points_tag], dtype=np.float64
    )
    points_camera = np.asarray(
        [camera_to_world.inverse().transformFrom(point) for point in points_world],
        dtype=np.float64,
    )
    projected = camera.project(points_camera)
    return (projected - constraint.image_corners).reshape(-1)


def _numeric_jacobian(
    pose: Any,
    other_pose: Any,
    constraint: ProjectionConstraint,
    camera: CameraModel,
    pose_is_camera: bool,
) -> FloatArray:
    """Compute a local-coordinate Jacobian for a CustomFactor."""

    epsilon = 1e-6
    result = np.empty((8, 6), dtype=np.float64)
    for index in range(6):
        delta = np.zeros(6, dtype=np.float64)
        delta[index] = epsilon
        plus = pose.retract(delta)
        minus = pose.retract(-delta)
        if pose_is_camera:
            plus_error = _residual(plus, other_pose, constraint, camera)
            minus_error = _residual(minus, other_pose, constraint, camera)
        else:
            plus_error = _residual(other_pose, plus, constraint, camera)
            minus_error = _residual(other_pose, minus, constraint, camera)
        result[:, index] = (plus_error - minus_error) / (2.0 * epsilon)
    return result


class GtsamGraph:
    """Incremental graph containing camera and static tag Pose3 variables."""

    def __init__(
        self,
        reference_tag_id: int,
        pixel_noise: float,
        camera: CameraModel,
    ) -> None:
        if gtsam is None:
            raise RuntimeError(
                "GTSAM is required; install the project with `uv sync --dev`"
            )
        self._camera_model = camera
        self._pixel_noise = pixel_noise
        self._isam = gtsam.ISAM2()
        self._factor_graph = gtsam.NonlinearFactorGraph()
        self._estimate = gtsam.Values()
        self._known_keys: set[int] = set()
        self._factor_count = 0
        self._reference_key = self.tag_key(reference_tag_id)

        initial = gtsam.Values()
        initial.insert(self._reference_key, gtsam.Pose3())
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(6, 1e-6, dtype=np.float64)
        )
        prior = gtsam.PriorFactorPose3(self._reference_key, gtsam.Pose3(), prior_noise)
        self._factor_graph.add(prior)
        self._isam.update(self._factor_graph, initial)
        self._estimate = self._isam.calculateEstimate()
        self._known_keys.add(self._reference_key)

    @staticmethod
    def camera_key(frame_id: int) -> int:
        """Return a stable GTSAM key for a camera pose."""

        if gtsam is None:
            return frame_id
        return int(gtsam.symbol("c", frame_id))

    @staticmethod
    def tag_key(tag_id: int) -> int:
        """Return a stable GTSAM key for a tag pose."""

        if gtsam is None:
            return tag_id
        return int(gtsam.symbol("t", tag_id))

    @property
    def factor_count(self) -> int:
        """Return the number of factors currently in the graph."""

        return self._factor_count

    @property
    def known_keys(self) -> frozenset[int]:
        """Return keys that have been inserted into GTSAM."""

        return frozenset(self._known_keys)

    def _make_factor(self, constraint: ProjectionConstraint) -> object:
        if gtsam is None:  # pragma: no cover
            raise RuntimeError("GTSAM is unavailable")
        noise = gtsam.noiseModel.Isotropic.Sigma(8, self._pixel_noise)

        def error_function(
            _factor: Any,
            values: Any,
            jacobians: list[FloatArray] | None,
        ) -> FloatArray:
            camera_pose = values.atPose3(constraint.camera_key)
            tag_pose = values.atPose3(constraint.tag_key)
            error = _residual(camera_pose, tag_pose, constraint, self._camera_model)
            if jacobians is not None:
                jacobians[0] = _numeric_jacobian(
                    camera_pose,
                    tag_pose,
                    constraint,
                    self._camera_model,
                    pose_is_camera=True,
                )
                jacobians[1] = _numeric_jacobian(
                    tag_pose,
                    camera_pose,
                    constraint,
                    self._camera_model,
                    pose_is_camera=False,
                )
            return error

        return gtsam.CustomFactor(
            noise,
            [constraint.camera_key, constraint.tag_key],
            error_function,
        )

    def update(
        self,
        camera_key: int,
        camera_initial: Pose,
        new_tags: Sequence[tuple[int, Pose]],
        constraints: Sequence[ProjectionConstraint],
    ) -> None:
        """Insert one frame and optimize the graph incrementally."""

        if gtsam is None:  # pragma: no cover
            raise RuntimeError("GTSAM is unavailable")
        values = gtsam.Values()
        values.insert(camera_key, _to_gtsam_pose(camera_initial))
        self._known_keys.add(camera_key)
        for tag_id, pose in new_tags:
            key = self.tag_key(tag_id)
            if key in self._known_keys:
                continue
            values.insert(key, _to_gtsam_pose(pose))
            self._known_keys.add(key)

        factors = gtsam.NonlinearFactorGraph()
        for constraint in constraints:
            factors.add(self._make_factor(constraint))
        for index in range(factors.size()):
            self._factor_graph.add(factors.at(index))
        self._factor_count += len(constraints)
        self._isam.update(factors, values)
        self._estimate = self._isam.calculateEstimate()

    def pose_for_key(self, key: int) -> Pose:
        """Return the current optimized pose for a key."""

        return _from_gtsam_pose(self._estimate.atPose3(key))


def _to_gtsam_pose(pose: Pose) -> object:
    if gtsam is None:  # pragma: no cover
        raise RuntimeError("GTSAM is unavailable")
    return gtsam.Pose3(gtsam.Rot3(pose.rotation), pose.translation)


def _from_gtsam_pose(pose: Any) -> Pose:
    rotation = np.asarray(pose.rotation().matrix(), dtype=np.float64)
    translation = np.asarray(pose.translation(), dtype=np.float64).reshape(3)
    return Pose(rotation, translation)
