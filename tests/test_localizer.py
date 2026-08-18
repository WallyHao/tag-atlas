from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tagatlas.localizer as localizer_module
from tagatlas import CameraModel, Detection, Localizer, LocalizerConfig, Pose, TagMap
from tagatlas.geometry import inverse_pose, tag_object_points, transform_points
from tagatlas.pose_graph import GtsamGraphError


class SequenceDetector:
    def __init__(self, frames: list[list[Detection]]) -> None:
        self._frames = iter(frames)

    def detect(self, image: np.ndarray) -> list[Detection]:
        del image
        return next(self._frames)


def make_frame(
    camera: CameraModel,
    camera_pose: Pose,
    tag_poses: dict[int, Pose],
    visible_ids: list[int],
) -> list[Detection]:
    camera_from_world = inverse_pose(camera_pose)
    result = []
    for tag_id in visible_ids:
        world_points = transform_points(tag_poses[tag_id], tag_object_points(0.12))
        camera_points = transform_points(camera_from_world, world_points)
        result.append(
            Detection(
                tag_id=tag_id,
                corners=camera.project(camera_points),
                tag_family="tag36h11",
                decision_margin=50.0,
            )
        )
    return result


def test_localizer_discovers_tag_and_optimizes_multiple_frames() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    camera_rotation = np.diag([1.0, -1.0, -1.0])
    tag_poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0])),
    }
    frames = [
        make_frame(
            camera,
            Pose(camera_rotation, np.array([0.0, 0.0, 1.5])),
            tag_poses,
            [0, 1],
        ),
        make_frame(
            camera,
            Pose(camera_rotation, np.array([0.15, 0.0, 1.5])),
            tag_poses,
            [0, 1],
        ),
    ]
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12, 1: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector(frames),
    )

    first = localizer.locate(np.zeros((480, 640), dtype=np.uint8))
    second = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert first.success
    assert second.success
    assert set(second.tag_poses) == {0, 1}
    assert second.camera_pose is not None
    np.testing.assert_allclose(
        second.camera_pose.translation, [0.15, 0.0, 1.5], atol=1e-6
    )
    assert second.reprojection_rmse is not None
    assert second.reprojection_rmse < 1e-6
    assert localizer.factor_count == 4


def test_fixed_map_does_not_discover_new_tags() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0])),
    }
    frame = make_frame(
        camera,
        Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5])),
        tag_poses,
        [0, 1],
    )
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(
            reference_tag_id=0,
            tag_sizes={0: 0.12, 1: 0.12},
            tag_poses={1: tag_poses[1]},
        ),
        config=LocalizerConfig(map_mode="fixed"),
        detector=SequenceDetector([frame]),
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert result.success
    assert result.diagnostics.new_tag_ids == ()
    assert set(result.tag_poses) == {0, 1}


def test_noisy_synthetic_sequence_stays_within_quality_gate() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0])),
    }
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.1, 0.0, 1.5]))
    rng = np.random.default_rng(7)
    frame = make_frame(camera, camera_pose, tag_poses, [0, 1])
    noisy_frame = [
        Detection(
            tag_id=detection.tag_id,
            corners=detection.corners + rng.normal(0.0, 0.2, (4, 2)),
            tag_family=detection.tag_family,
            decision_margin=detection.decision_margin,
        )
        for detection in frame
    ]
    localizer = Localizer(
        config=LocalizerConfig(
            max_reprojection_error=8.0,
            robust_loss="huber",
        ),
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12, 1: 0.12}),
        detector=SequenceDetector([noisy_frame]),
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert result.success
    assert result.camera_pose is not None
    assert result.reprojection_rmse is not None
    assert result.reprojection_rmse < 1.0
    np.testing.assert_allclose(
        result.camera_pose.translation,
        camera_pose.translation,
        atol=0.12,
    )


def test_unconnected_tag_is_not_added_to_graph() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0]))}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    frame = make_frame(camera, camera_pose, tag_poses, [1])
    detector = SequenceDetector([frame])
    localizer = Localizer.from_config(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12, 1: 0.12}),
        config={},
        detector=detector,
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason == "observations are not connected to the current tag map"
    assert set(localizer.tag_poses) == {0}
    assert localizer.factor_count == 0


def test_rejected_frame_does_not_change_graph() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0])),
    }
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    good_frame = make_frame(camera, camera_pose, tag_poses, [0, 1])
    bad_frame = make_frame(camera, camera_pose, tag_poses, [0, 1])
    bad_frame[1] = Detection(
        tag_id=1,
        corners=bad_frame[1].corners + np.array([100.0, 0.0]),
        tag_family="tag36h11",
        decision_margin=50.0,
    )
    localizer = Localizer.from_config(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12, 1: 0.12}),
        config={"max_reprojection_error": 1.0},
        detector=SequenceDetector([good_frame, bad_frame]),
    )

    first = localizer.locate(np.zeros((480, 640), dtype=np.uint8))
    before = localizer.tag_poses
    second = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert first.success
    assert not second.success
    assert localizer.factor_count == 2
    np.testing.assert_allclose(localizer.tag_poses[1].matrix(), before[1].matrix())


def test_detector_adapter_accepts_pupil_style_objects() -> None:
    value = SimpleNamespace(
        tag_id=4,
        corners=np.zeros((4, 2)),
        tag_family=b"tag36h11",
        decision_margin=12.0,
        hamming=1,
    )
    from tagatlas.apriltag_detector import normalize_detection

    detection = normalize_detection(value)
    assert detection.tag_id == 4
    assert detection.tag_family == "tag36h11"
    assert detection.hamming == 1


def test_empty_frame_is_rejected() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector([[]]),
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.frame_id == 0
    assert result.reason == "no configured AprilTags detected"


def test_camera_initialization_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    frame = [
        Detection(
            0,
            np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
            "tag36h11",
            decision_margin=50.0,
        )
    ]
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector([frame]),
    )
    monkeypatch.setattr(
        localizer_module, "seed_camera_pose", lambda *_args, **_kwargs: None
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason == "unable to initialize camera pose"


def test_initial_reprojection_failure_does_not_update_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    frame = [
        Detection(
            0,
            np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
            "tag36h11",
            decision_margin=50.0,
        )
    ]
    localizer = Localizer(
        config=LocalizerConfig(
            max_reprojection_error=1.0,
        ),
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        detector=SequenceDetector([frame]),
    )
    monkeypatch.setattr(
        localizer_module,
        "_detection_reprojection_rmses",
        lambda *_args, **_kwargs: {0: 2.0},
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason == "all observations exceed reprojection limit"
    assert result.used_tag_ids == ()
    assert localizer.factor_count == 0


def test_graph_update_failure_is_rolled_back(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {0: Pose.identity()}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector([[*make_frame(camera, camera_pose, tag_poses, [0])]]),
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise GtsamGraphError("synthetic graph failure")

    monkeypatch.setattr(localizer._graph, "update", fail)
    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason == "GTSAM update failed: synthetic graph failure"
    assert localizer.factor_count == 0


def test_graph_rebuild_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {0: Pose.identity()}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector([make_frame(camera, camera_pose, tag_poses, [0])]),
    )

    monkeypatch.setattr(
        localizer._graph,
        "update",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GtsamGraphError("synthetic graph failure")
        ),
    )
    monkeypatch.setattr(
        localizer,
        "_rebuild_graph",
        lambda: (_ for _ in ()).throw(GtsamGraphError("synthetic rebuild failure")),
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason is not None
    assert "graph rebuild failed: synthetic rebuild failure" in result.reason


def test_graph_result_read_failure_is_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {0: Pose.identity()}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector([make_frame(camera, camera_pose, tag_poses, [0])]),
    )

    monkeypatch.setattr(
        localizer._graph,
        "pose_for_key",
        lambda _key: (_ for _ in ()).throw(
            GtsamGraphError("synthetic pose read failure")
        ),
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason is not None
    assert "result read failed" in result.reason
    assert localizer.factor_count == 0


def test_reprojection_failure_after_optimization_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {0: Pose.identity()}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    localizer = Localizer(
        config=LocalizerConfig(
            max_reprojection_error=1.0,
        ),
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12}),
        detector=SequenceDetector([[*make_frame(camera, camera_pose, tag_poses, [0])]]),
    )
    values = iter([2.0, 2.0])
    monkeypatch.setattr(
        localizer_module, "reprojection_rmse", lambda *_args: next(values)
    )

    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))

    assert not result.success
    assert result.reason == "reprojection error exceeds configured limit"
    assert localizer.factor_count == 0


def test_reset_clears_map_and_rebuilds_accepted_graph() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {0: Pose.identity(), 1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0]))}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    frame = make_frame(camera, camera_pose, tag_poses, [0, 1])
    localizer = Localizer(
        camera=camera,
        tag_map=TagMap(reference_tag_id=0, tag_sizes={0: 0.12, 1: 0.12}),
        config=LocalizerConfig(),
        detector=SequenceDetector([frame, frame]),
    )

    assert localizer.tag_sizes == {0: 0.12, 1: 0.12}
    assert localizer.locate(np.zeros((480, 640), dtype=np.uint8)).success
    localizer._rebuild_graph()
    localizer.reset()

    assert set(localizer.tag_poses) == {0}
    assert localizer.factor_count == 0
    result = localizer.locate(np.zeros((480, 640), dtype=np.uint8))
    assert result.success
    assert result.frame_id == 0
