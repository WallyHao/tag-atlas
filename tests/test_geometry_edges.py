import cv2
import numpy as np
import pytest

import tagatlas.geometry as geometry
from tagatlas import Pose


def test_geometry_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        geometry.tag_object_points(0.0)
    with pytest.raises(ValueError, match="shape"):
        geometry.pose_from_matrix(np.eye(3))
    with pytest.raises(ValueError, match="homogeneous"):
        geometry.pose_from_matrix(np.eye(4) * 2.0)
    with pytest.raises(ValueError, match="points"):
        geometry.transform_points(Pose.identity(), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="at least four"):
        geometry.pose_from_pnp(
            np.zeros((3, 3)),
            np.zeros((3, 2)),
            np.eye(3),
            np.zeros(5),
        )


def test_pose_from_matrix_converts_homogeneous_transform() -> None:
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    pose = geometry.pose_from_matrix(matrix)

    np.testing.assert_allclose(pose.translation, [1.0, 2.0, 3.0])


def test_pnp_returns_none_when_opencv_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise cv2.error("synthetic failure")

    monkeypatch.setattr(geometry.cv2, "solvePnP", fail)

    result = geometry.pose_from_pnp(
        np.zeros((4, 3)),
        np.zeros((4, 2)),
        np.eye(3),
        np.zeros(5),
    )

    assert result is None


def test_pnp_returns_none_when_both_methods_report_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def report_failure(
        *_args: object, **_kwargs: object
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        return False, np.zeros((3, 1)), np.zeros((3, 1))

    monkeypatch.setattr(geometry.cv2, "solvePnP", report_failure)

    result = geometry.pose_from_pnp(
        np.zeros((4, 3)),
        np.zeros((4, 2)),
        np.eye(3),
        np.zeros(5),
    )

    assert result is None
