import numpy as np

import tagatlas.initialization as initialization
from tagatlas import CameraModel, Detection, Pose
from tagatlas.metrics import reprojection_rmse


def make_camera() -> CameraModel:
    return CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )


def make_detection(tag_id: int = 0) -> Detection:
    return Detection(tag_id, np.zeros((4, 2)), "tag36h11")


def test_seed_camera_pose_returns_none_without_mapped_tags() -> None:
    result = initialization.seed_camera_pose(
        [make_detection()], {}, {0: 0.12}, make_camera()
    )

    assert result is None


def test_seed_camera_pose_uses_fallback_when_pnp_fails(
    monkeypatch,
) -> None:
    fallback = Pose.identity()
    monkeypatch.setattr(initialization, "pose_from_pnp", lambda *_args: None)

    result = initialization.seed_camera_pose(
        [make_detection()], {0: Pose.identity()}, {0: 0.12}, make_camera(), fallback
    )

    assert result is not None
    np.testing.assert_allclose(result.matrix(), fallback.matrix())


def test_seed_tag_pose_returns_none_when_pnp_fails(monkeypatch) -> None:
    monkeypatch.setattr(initialization, "pose_from_pnp", lambda *_args: None)

    result = initialization.seed_tag_pose(
        make_detection(), Pose.identity(), {0: 0.12}, make_camera()
    )

    assert result is None


def test_reprojection_rmse_is_infinite_without_detections() -> None:
    result = reprojection_rmse(Pose.identity(), [], {}, {0: 0.12}, make_camera())

    assert result == float("inf")
