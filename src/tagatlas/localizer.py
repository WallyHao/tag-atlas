"""Multi-frame AprilTag localization backed by GTSAM."""

from __future__ import annotations

import logging
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
from .pose_graph import GtsamGraph, GtsamGraphError, ProjectionConstraint
from .tag_map import TagMap

logger = logging.getLogger(__name__)


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
        *,
        camera: CameraModel,
        tag_map: TagMap,
        config: LocalizerConfig | None = None,
        detector: DetectorProtocol | None = None,
    ) -> None:
        self.config = config or LocalizerConfig()
        self.camera = camera
        self.tag_map = tag_map
        self._detector = detector or PupilAprilTagDetector(self.config.tag_family)
        self._frame_id = 0
        self._last_camera_pose: Pose | None = None
        self._known_tag_poses: dict[int, Pose] = {
            self.tag_map.reference_tag_id: Pose.identity()
        }
        self._accepted_updates: list[_GraphUpdate] = []
        self._graph = GtsamGraph(
            self.tag_map.reference_tag_id,
            self.config.pixel_noise,
            self.camera,
            robust_loss=self.config.robust_loss,
            robust_scale=self.config.robust_scale,
        )
        logger.debug(
            "created localizer reference_tag_id=%d tag_count=%d",
            self.tag_map.reference_tag_id,
            len(self.tag_map.tag_sizes),
        )

    @classmethod
    def from_config(
        cls,
        *,
        camera: CameraModel,
        tag_map: TagMap,
        config: LocalizerConfig | Mapping[str, object] | None = None,
        detector: DetectorProtocol | None = None,
    ) -> Localizer:
        """Construct a localizer from typed or mapping algorithm configuration."""

        if config is None:
            parsed = LocalizerConfig()
        elif isinstance(config, LocalizerConfig):
            parsed = config
        else:
            parsed = LocalizerConfig.from_mapping(config)
        return cls(
            camera=camera,
            tag_map=tag_map,
            config=parsed,
            detector=detector,
        )

    @property
    def tag_sizes(self) -> Mapping[int, float]:
        """Return configured tag sizes in meters."""

        return self.tag_map.tag_sizes

    @property
    def reference_tag_id(self) -> int:
        """Return the fixed world-frame reference Tag ID."""

        return self.tag_map.reference_tag_id

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
        self._known_tag_poses = {self.tag_map.reference_tag_id: Pose.identity()}
        self._accepted_updates = []
        self._graph = GtsamGraph(
            self.tag_map.reference_tag_id,
            self.config.pixel_noise,
            self.camera,
            robust_loss=self.config.robust_loss,
            robust_scale=self.config.robust_scale,
        )
        logger.info("localizer state reset")

    def _failure(
        self,
        frame_id: int,
        reason: str,
        used_tag_ids: Sequence[int] = (),
    ) -> LocalizationResult:
        logger.debug(
            "frame rejected frame_id=%d reason=%s used_tag_ids=%s",
            frame_id,
            reason,
            tuple(used_tag_ids),
        )
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
            self.tag_map.reference_tag_id,
            self.config.pixel_noise,
            self.camera,
            robust_loss=self.config.robust_loss,
            robust_scale=self.config.robust_scale,
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
            tag_sizes=self.tag_map.tag_sizes,
            min_decision_margin=self.config.min_decision_margin,
            min_tag_area=self.config.min_tag_area,
        )
        logger.debug("processing frame_id=%d detections=%d", frame_id, len(detections))
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
            self.tag_map.tag_sizes,
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
                detection, camera_pose, self.tag_map.tag_sizes, self.camera
            )
            if pose is not None:
                candidate_tag_poses[detection.tag_id] = pose
                new_tag_poses.append((detection.tag_id, pose))

        camera_key = GtsamGraph.camera_key(frame_id)
        constraints = [
            ProjectionConstraint(
                camera_key=camera_key,
                tag_key=GtsamGraph.tag_key(detection.tag_id),
                tag_size=self.tag_map.tag_sizes[detection.tag_id],
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
            self.tag_map.tag_sizes,
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
        except GtsamGraphError as exc:
            logger.warning(
                "graph update failed frame_id=%d; rebuilding graph: %s",
                frame_id,
                exc,
                exc_info=True,
            )
            try:
                self._rebuild_graph()
            except GtsamGraphError as rebuild_exc:
                logger.exception("graph rebuild failed after update failure")
                return self._failure(
                    frame_id,
                    f"GTSAM update failed: {exc}; graph rebuild failed: {rebuild_exc}",
                )
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
            self.tag_map.tag_sizes,
            self.camera,
        )
        success = rmse <= self.config.max_reprojection_error
        if not success:
            try:
                self._rebuild_graph()
            except GtsamGraphError as rebuild_exc:
                logger.exception("graph rebuild failed after reprojection rejection")
                return self._failure(
                    frame_id,
                    "reprojection error exceeds configured limit; "
                    f"graph rebuild failed: {rebuild_exc}",
                    used_ids,
                )
            return self._failure(
                frame_id,
                "reprojection error exceeds configured limit",
                used_ids,
            )
        self._accepted_updates.append(update)
        self._known_tag_poses = optimized_tags
        self._last_camera_pose = optimized_camera
        logger.info(
            "frame accepted frame_id=%d used_tag_ids=%s rmse=%.3f",
            frame_id,
            used_ids,
            rmse,
        )
        return LocalizationResult(
            success=success,
            frame_id=frame_id,
            camera_pose=optimized_camera if success else None,
            tag_poses=dict(optimized_tags),
            used_tag_ids=used_ids,
            reprojection_rmse=rmse,
            reason=None if success else "reprojection error exceeds configured limit",
        )
