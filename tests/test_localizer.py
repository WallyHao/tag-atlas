from types import SimpleNamespace

import numpy as np

from tagatlas import CameraModel, Detection, Localizer, Pose
from tagatlas.geometry import inverse_pose, tag_object_points, transform_points


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
        {0: 0.12, 1: 0.12},
        camera.matrix,
        camera.distortion_coefficients,
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


def test_unconnected_tag_is_not_added_to_graph() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    tag_poses = {1: Pose(np.eye(3), np.array([0.4, 0.0, 0.0]))}
    camera_pose = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 1.5]))
    frame = make_frame(camera, camera_pose, tag_poses, [1])
    detector = SequenceDetector([frame])
    localizer = Localizer(
        {0: 0.12, 1: 0.12},
        camera.matrix,
        camera.distortion_coefficients,
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
    localizer = Localizer(
        {0: 0.12, 1: 0.12},
        camera.matrix,
        camera.distortion_coefficients,
        max_reprojection_error=1.0,
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
    from tagatlas.detector import normalize_detection

    detection = normalize_detection(value)
    assert detection.tag_id == 4
    assert detection.tag_family == "tag36h11"
    assert detection.hamming == 1
