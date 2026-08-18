"""Multi-frame AprilTag localization backed by GTSAM."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from numpy.typing import NDArray

from .apriltag_detector import (
    DetectorProtocol,
    PupilAprilTagDetector,
    filter_detections,
)
from .camera import CameraModel
from .initialization import seed_camera_pose, seed_tag_pose
from .metrics import detection_reprojection_rmse, reprojection_rmse
from .models import (
    Detection,
    LocalizationDiagnostics,
    LocalizationResult,
    LocalizerConfig,
    Pose,
)
from .pose_graph import GtsamGraph, GtsamGraphError, ProjectionConstraint
from .tag_map import TagMap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GraphUpdate:
    camera_key: int
    camera_initial: Pose
    new_tags: tuple[tuple[int, Pose], ...]
    constraints: tuple[ProjectionConstraint, ...]


def _detection_reprojection_rmses(
    camera_pose: Pose,
    detections: Sequence[Detection],
    tag_poses: Mapping[int, Pose],
    tag_sizes: Mapping[int, float],
    camera: CameraModel,
) -> dict[int, float]:
    """Return one pixel RMSE per Tag detection in a single projection pass."""

    return {
        detection.tag_id: detection_reprojection_rmse(
            camera_pose,
            detection,
            tag_poses,
            tag_sizes,
            camera,
        )
        for detection in detections
    }


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
        self._lock = RLock()
        self._detector = (
            detector
            if detector is not None
            else PupilAprilTagDetector(
                families=self.config.tag_family,
                nthreads=self.config.detector.nthreads,
                quad_decimate=self.config.detector.quad_decimate,
                quad_sigma=self.config.detector.quad_sigma,
                refine_edges=self.config.detector.refine_edges,
                decode_sharpening=self.config.detector.decode_sharpening,
            )
        )
        self._frame_id = 0
        self._last_camera_pose: Pose | None = None
        self._known_tag_poses: dict[int, Pose] = {
            self.tag_map.reference_tag_id: Pose.identity()
        }
        if self.config.map_mode == "fixed":
            self._known_tag_poses.update(self.tag_map.tag_poses)
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

        with self._lock:
            return dict(self._known_tag_poses)

    @property
    def factor_count(self) -> int:
        """Return the number of projection factors in the graph."""

        with self._lock:
            return self._graph.factor_count

    def reset(self) -> None:
        """Clear state while keeping calibration; safe during a concurrent locate."""

        with self._lock:
            self._reset()

    def _reset(self) -> None:
        """Reset implementation; caller must hold ``self._lock``."""

        self._frame_id = 0
        self._last_camera_pose = None
        self._known_tag_poses = {self.tag_map.reference_tag_id: Pose.identity()}
        if self.config.map_mode == "fixed":
            self._known_tag_poses.update(self.tag_map.tag_poses)
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
        diagnostics: LocalizationDiagnostics | None = None,
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
            diagnostics=diagnostics or LocalizationDiagnostics(),
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

    def _recover_graph_failure(
        self,
        frame_id: int,
        reason: str,
        used_tag_ids: Sequence[int] = (),
    ) -> LocalizationResult:
        """Restore the last committed graph before reporting a graph failure."""

        try:
            self._rebuild_graph()
        except GtsamGraphError as rebuild_exc:
            logger.exception("graph rebuild failed after graph operation failure")
            reason = f"{reason}; graph rebuild failed: {rebuild_exc}"
        return self._failure(frame_id, reason, used_tag_ids)

    def locate(self, image: NDArray[Any]) -> LocalizationResult:
        """Process one image; calls are serialized per Localizer instance."""

        with self._lock:
            return self._locate(image)

    def _locate(self, image: NDArray[Any]) -> LocalizationResult:
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
        diagnostics = LocalizationDiagnostics(
            detected_count=len(detections),
            configured_count=len(detections),
        )
        logger.debug("processing frame_id=%d detections=%d", frame_id, len(detections))
        if not detections:
            return self._failure(
                frame_id,
                "no configured AprilTags detected",
                diagnostics=diagnostics,
            )

        known_detections = tuple(
            detection
            for detection in detections
            if detection.tag_id in self._known_tag_poses
        )
        if not known_detections:
            return self._failure(
                frame_id,
                "observations are not connected to the current tag map",
                diagnostics=diagnostics,
            )
        camera_pose = seed_camera_pose(
            known_detections,
            self._known_tag_poses,
            self.tag_map.tag_sizes,
            self.camera,
            fallback=self._last_camera_pose,
        )
        if camera_pose is None:
            return self._failure(
                frame_id, "unable to initialize camera pose", diagnostics=diagnostics
            )

        candidate_tag_poses = dict(self._known_tag_poses)
        candidate_new_tags: list[tuple[int, Pose]] = []
        if self.config.map_mode == "fixed":
            candidate_new_tags = [
                (tag_id, pose)
                for tag_id, pose in self.tag_map.tag_poses.items()
                if GtsamGraph.tag_key(tag_id) not in self._graph.known_keys
            ]
        else:
            for detection in detections:
                if detection.tag_id in candidate_tag_poses:
                    continue
                pose = seed_tag_pose(
                    detection, camera_pose, self.tag_map.tag_sizes, self.camera
                )
                if pose is not None:
                    candidate_tag_poses[detection.tag_id] = pose
                    candidate_new_tags.append((detection.tag_id, pose))

        detection_rmses = _detection_reprojection_rmses(
            camera_pose,
            detections,
            candidate_tag_poses,
            self.tag_map.tag_sizes,
            self.camera,
        )
        accepted_detections = tuple(
            detection
            for detection in detections
            if detection.tag_id in candidate_tag_poses
            and detection_rmses[detection.tag_id] <= self.config.max_reprojection_error
        )
        diagnostics = LocalizationDiagnostics(
            detected_count=len(detections),
            configured_count=len(detections),
            accepted_count=len(accepted_detections),
            rejected_tag_ids=tuple(
                detection.tag_id
                for detection in detections
                if detection not in accepted_detections
            ),
            new_tag_ids=tuple(tag_id for tag_id, _pose in candidate_new_tags),
        )
        if not accepted_detections:
            return self._failure(
                frame_id,
                "all observations exceed reprojection limit",
                diagnostics=diagnostics,
            )
        if not any(
            detection.tag_id in self._known_tag_poses
            for detection in accepted_detections
        ):
            return self._failure(
                frame_id,
                "all mapped observations exceed reprojection limit",
                diagnostics=diagnostics,
            )
        accepted_ids = {detection.tag_id for detection in accepted_detections}
        new_tag_poses = [item for item in candidate_new_tags if item[0] in accepted_ids]
        if self.config.map_mode == "discover":
            candidate_tag_poses = {
                tag_id: pose
                for tag_id, pose in candidate_tag_poses.items()
                if tag_id in accepted_ids or tag_id == self.tag_map.reference_tag_id
            }

        camera_key = GtsamGraph.camera_key(frame_id)
        constraints = [
            ProjectionConstraint(
                camera_key=camera_key,
                tag_key=GtsamGraph.tag_key(detection.tag_id),
                tag_size=self.tag_map.tag_sizes[detection.tag_id],
                image_corners=detection.corners,
            )
            for detection in accepted_detections
            if detection.tag_id in candidate_tag_poses
        ]

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
            return self._recover_graph_failure(frame_id, f"GTSAM update failed: {exc}")

        try:
            optimized_camera = self._graph.pose_for_key(camera_key)
            pose_covariance = self._graph.pose_covariance_for_key(camera_key)
            optimized_tags = dict(self._known_tag_poses)
            for tag_id in candidate_tag_poses:
                optimized_tags[tag_id] = self._graph.pose_for_key(
                    GtsamGraph.tag_key(tag_id)
                )
        except GtsamGraphError as exc:
            if "covariance" in str(exc):
                pose_covariance = None
                logger.debug("camera covariance unavailable: %s", exc)
                optimized_camera = self._graph.pose_for_key(camera_key)
                optimized_tags = dict(self._known_tag_poses)
                for tag_id in candidate_tag_poses:
                    optimized_tags[tag_id] = self._graph.pose_for_key(
                        GtsamGraph.tag_key(tag_id)
                    )
            else:
                return self._recover_graph_failure(
                    frame_id, f"GTSAM result read failed: {exc}"
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
            return self._recover_graph_failure(
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
            pose_covariance=pose_covariance,
            diagnostics=LocalizationDiagnostics(
                detected_count=len(detections),
                configured_count=len(detections),
                accepted_count=len(accepted_detections),
                used_tag_ids=used_ids,
                rejected_tag_ids=tuple(
                    detection.tag_id
                    for detection in detections
                    if detection not in accepted_detections
                ),
                new_tag_ids=(
                    tuple(tag_id for tag_id, _pose in new_tag_poses)
                    if self.config.map_mode == "discover"
                    else ()
                ),
            ),
        )
