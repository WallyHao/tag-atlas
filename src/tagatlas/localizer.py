"""Multi-frame AprilTag localization backed by GTSAM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .camera import CameraModel
from .detector import DetectorProtocol, PupilAprilTagDetector, normalize_detection
from .geometry import (
    compose_pose,
    inverse_pose,
    pose_from_pnp,
    tag_object_points,
    transform_points,
)
from .graph import GtsamGraph, ProjectionConstraint
from .types import (
    Detection,
    FloatArray,
    LocalizationResult,
    LocalizerConfig,
    Pose,
)


@dataclass(frozen=True)
class _GraphUpdate:
    camera_key: int
    camera_initial: Pose
    new_tags: tuple[tuple[int, Pose], ...]
    constraints: tuple[ProjectionConstraint, ...]


class Localizer:
    """Estimate camera pose and discover a static tag map over multiple frames.

    ``T_world_camera`` is returned in the coordinate frame of the configured
    reference tag, whose pose is fixed to the identity transform.
    """

    def __init__(
        self,
        tag_sizes: Mapping[int, float],
        camera_matrix: FloatArray,
        distortion_coefficients: FloatArray | None = None,
        *,
        reference_tag_id: int = 0,
        tag_family: str = "tag36h11",
        distortion_model: str = "radtan",
        pixel_noise: float = 1.0,
        max_reprojection_error: float = 8.0,
        min_decision_margin: float = 0.0,
        detector: DetectorProtocol | Any | None = None,
    ) -> None:
        self.config = LocalizerConfig(
            reference_tag_id=reference_tag_id,
            tag_sizes=tag_sizes,
            tag_family=tag_family,
            pixel_noise=pixel_noise,
            max_reprojection_error=max_reprojection_error,
            min_decision_margin=min_decision_margin,
        )
        self.camera = CameraModel(
            camera_matrix,
            distortion_coefficients,
            distortion_model=distortion_model,
        )
        self._detector = detector or PupilAprilTagDetector(tag_family)
        self._frame_id = 0
        self._last_camera_pose: Pose | None = None
        self._known_tag_poses: dict[int, Pose] = {
            self.config.reference_tag_id: Pose.identity()
        }
        self._accepted_updates: list[_GraphUpdate] = []
        self._graph = GtsamGraph(
            self.config.reference_tag_id,
            self.config.pixel_noise,
            self.camera,
        )

    @classmethod
    def from_config(
        cls,
        config: LocalizerConfig | Mapping[str, object],
        camera: CameraModel,
        *,
        detector: DetectorProtocol | Any | None = None,
    ) -> Localizer:
        """Construct a localizer from a config mapping and camera model."""

        parsed = (
            config
            if isinstance(config, LocalizerConfig)
            else LocalizerConfig.from_mapping(config)
        )
        return cls(
            parsed.tag_sizes,
            camera.matrix,
            camera.distortion_coefficients,
            reference_tag_id=parsed.reference_tag_id,
            tag_family=parsed.tag_family,
            distortion_model=camera.distortion_model,
            pixel_noise=parsed.pixel_noise,
            max_reprojection_error=parsed.max_reprojection_error,
            min_decision_margin=parsed.min_decision_margin,
            detector=detector,
        )

    @property
    def tag_sizes(self) -> Mapping[int, float]:
        """Return configured tag sizes in meters."""

        return self.config.tag_sizes

    @property
    def tag_poses(self) -> Mapping[int, Pose]:
        """Return the latest optimized tag map."""

        return dict(self._known_tag_poses)

    @property
    def factor_count(self) -> int:
        """Return the number of projection factors in the graph."""

        return self._graph.factor_count

    def reset(self) -> None:
        """Clear the trajectory and discovered map while keeping calibration."""

        self._frame_id = 0
        self._last_camera_pose = None
        self._known_tag_poses = {self.config.reference_tag_id: Pose.identity()}
        self._accepted_updates = []
        self._graph = GtsamGraph(
            self.config.reference_tag_id,
            self.config.pixel_noise,
            self.camera,
        )

    def _failure(
        self,
        frame_id: int,
        reason: str,
        used_tag_ids: Sequence[int] = (),
    ) -> LocalizationResult:
        return LocalizationResult(
            success=False,
            frame_id=frame_id,
            camera_pose=None,
            tag_poses=dict(self._known_tag_poses),
            used_tag_ids=tuple(used_tag_ids),
            reprojection_rmse=None,
            reason=reason,
        )

    def _detections(self, image: NDArray[Any]) -> tuple[Detection, ...]:
        raw = self._detector.detect(image)
        selected: dict[int, Detection] = {}
        for value in raw:
            detection = (
                value if isinstance(value, Detection) else normalize_detection(value)
            )
            if detection.tag_family != self.config.tag_family:
                continue
            if detection.tag_id not in self.config.tag_sizes:
                continue
            if detection.decision_margin < self.config.min_decision_margin:
                continue
            previous = selected.get(detection.tag_id)
            if previous is None or detection.decision_margin > previous.decision_margin:
                selected[detection.tag_id] = detection
        return tuple(selected.values())

    def _camera_seed(self, detections: Sequence[Detection]) -> Pose | None:
        assert self.camera.distortion_coefficients is not None
        world_points: list[FloatArray] = []
        image_points: list[FloatArray] = []
        for detection in detections:
            tag_pose = self._known_tag_poses.get(detection.tag_id)
            if tag_pose is None:
                continue
            local_points = tag_object_points(self.config.tag_sizes[detection.tag_id])
            world_points.append(transform_points(tag_pose, local_points))
            image_points.append(detection.corners)
        if not world_points:
            return None
        camera_from_world = pose_from_pnp(
            np.vstack(world_points),
            np.vstack(image_points),
            self.camera.matrix,
            self.camera.distortion_coefficients,
        )
        if camera_from_world is None:
            return self._last_camera_pose
        return inverse_pose(camera_from_world)

    def _new_tag_seed(self, detection: Detection, camera_pose: Pose) -> Pose | None:
        assert self.camera.distortion_coefficients is not None
        local_points = tag_object_points(self.config.tag_sizes[detection.tag_id])
        camera_from_tag = pose_from_pnp(
            local_points,
            detection.corners,
            self.camera.matrix,
            self.camera.distortion_coefficients,
        )
        if camera_from_tag is None:
            return None
        return compose_pose(camera_pose, camera_from_tag)

    def _rmse(
        self,
        camera_pose: Pose,
        detections: Sequence[Detection],
        tag_poses: Mapping[int, Pose],
    ) -> float:
        camera_from_world = inverse_pose(camera_pose)
        errors: list[FloatArray] = []
        for detection in detections:
            tag_pose = tag_poses[detection.tag_id]
            points_world = transform_points(
                tag_pose, tag_object_points(self.config.tag_sizes[detection.tag_id])
            )
            points_camera = transform_points(camera_from_world, points_world)
            predicted = self.camera.project(points_camera)
            errors.append((predicted - detection.corners).reshape(-1))
        if not errors:
            return float("inf")
        return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))

    def _rebuild_graph(self) -> None:
        self._graph = GtsamGraph(
            self.config.reference_tag_id,
            self.config.pixel_noise,
            self.camera,
        )
        for update in self._accepted_updates:
            self._graph.update(
                update.camera_key,
                update.camera_initial,
                update.new_tags,
                update.constraints,
            )

    def locate(self, image: NDArray[Any]) -> LocalizationResult:
        """Process one image and return the optimized camera pose."""

        frame_id = self._frame_id
        self._frame_id += 1
        detections = self._detections(image)
        if not detections:
            return self._failure(frame_id, "no configured AprilTags detected")

        known_detections = tuple(
            detection
            for detection in detections
            if detection.tag_id in self._known_tag_poses
        )
        if not known_detections:
            return self._failure(
                frame_id,
                "observations are not connected to the current tag map",
            )
        camera_pose = self._camera_seed(known_detections)
        if camera_pose is None:
            return self._failure(frame_id, "unable to initialize camera pose")

        new_tag_poses: list[tuple[int, Pose]] = []
        candidate_tag_poses = dict(self._known_tag_poses)
        for detection in detections:
            if detection.tag_id in candidate_tag_poses:
                continue
            pose = self._new_tag_seed(detection, camera_pose)
            if pose is not None:
                candidate_tag_poses[detection.tag_id] = pose
                new_tag_poses.append((detection.tag_id, pose))

        camera_key = GtsamGraph.camera_key(frame_id)
        constraints = [
            ProjectionConstraint(
                camera_key=camera_key,
                tag_key=GtsamGraph.tag_key(detection.tag_id),
                tag_size=self.config.tag_sizes[detection.tag_id],
                image_corners=detection.corners,
            )
            for detection in detections
            if detection.tag_id in candidate_tag_poses
        ]
        if not constraints:
            return self._failure(frame_id, "no valid projection factors to add")

        accepted_detections = tuple(
            detection
            for detection in detections
            if detection.tag_id in candidate_tag_poses
        )
        initial_rmse = self._rmse(camera_pose, accepted_detections, candidate_tag_poses)
        if initial_rmse > self.config.max_reprojection_error:
            return self._failure(
                frame_id,
                "initial reprojection error exceeds configured limit",
                [detection.tag_id for detection in accepted_detections],
            )

        update = _GraphUpdate(
            camera_key=camera_key,
            camera_initial=camera_pose,
            new_tags=tuple(new_tag_poses),
            constraints=tuple(constraints),
        )

        try:
            self._graph.update(camera_key, camera_pose, new_tag_poses, constraints)
        except Exception as exc:
            self._rebuild_graph()
            return self._failure(frame_id, f"GTSAM update failed: {exc}")

        optimized_camera = self._graph.pose_for_key(camera_key)
        optimized_tags = dict(self._known_tag_poses)
        for tag_id in candidate_tag_poses:
            optimized_tags[tag_id] = self._graph.pose_for_key(
                GtsamGraph.tag_key(tag_id)
            )
        used_ids = tuple(
            detection.tag_id
            for detection in accepted_detections
            if detection.tag_id in optimized_tags
        )
        rmse = self._rmse(optimized_camera, accepted_detections, optimized_tags)
        success = rmse <= self.config.max_reprojection_error
        if not success:
            self._rebuild_graph()
            return self._failure(
                frame_id,
                "reprojection error exceeds configured limit",
                used_ids,
            )
        self._accepted_updates.append(update)
        self._known_tag_poses = optimized_tags
        self._last_camera_pose = optimized_camera
        return LocalizationResult(
            success=success,
            frame_id=frame_id,
            camera_pose=optimized_camera if success else None,
            tag_poses=dict(optimized_tags),
            used_tag_ids=used_ids,
            reprojection_rmse=rmse,
            reason=None if success else "reprojection error exceeds configured limit",
        )
