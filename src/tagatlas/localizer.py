"""Multi-frame AprilTag localization backed by GTSAM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from .apriltag_detector import (
    DetectorProtocol,
    PupilAprilTagDetector,
    filter_detections,
)
from .camera import CameraModel
from .initialization import seed_camera_pose, seed_tag_pose
from .metrics import reprojection_rmse
from .models import LocalizationResult, LocalizerConfig, Pose
from .pose_graph import GtsamGraph, ProjectionConstraint


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
        config: LocalizerConfig,
        camera: CameraModel,
        *,
        detector: DetectorProtocol | Any | None = None,
    ) -> None:
        self.config = config
        self.camera = camera
        self._detector = detector or PupilAprilTagDetector(config.tag_family)
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
        """Construct a localizer from typed or mapping configuration."""

        parsed = (
            config
            if isinstance(config, LocalizerConfig)
            else LocalizerConfig.from_mapping(config)
        )
        return cls(
            parsed,
            camera,
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
        detections = filter_detections(
            self._detector.detect(image),
            tag_family=self.config.tag_family,
            tag_sizes=self.config.tag_sizes,
            min_decision_margin=self.config.min_decision_margin,
        )
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
        camera_pose = seed_camera_pose(
            known_detections,
            self._known_tag_poses,
            self.config.tag_sizes,
            self.camera,
            fallback=self._last_camera_pose,
        )
        if camera_pose is None:
            return self._failure(frame_id, "unable to initialize camera pose")

        new_tag_poses: list[tuple[int, Pose]] = []
        candidate_tag_poses = dict(self._known_tag_poses)
        for detection in detections:
            if detection.tag_id in candidate_tag_poses:
                continue
            pose = seed_tag_pose(
                detection, camera_pose, self.config.tag_sizes, self.camera
            )
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

        accepted_detections = tuple(
            detection
            for detection in detections
            if detection.tag_id in candidate_tag_poses
        )
        initial_rmse = reprojection_rmse(
            camera_pose,
            accepted_detections,
            candidate_tag_poses,
            self.config.tag_sizes,
            self.camera,
        )
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
        rmse = reprojection_rmse(
            optimized_camera,
            accepted_detections,
            optimized_tags,
            self.config.tag_sizes,
            self.camera,
        )
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
