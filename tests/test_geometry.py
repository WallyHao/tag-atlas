import numpy as np

from tagatlas import Pose
from tagatlas.geometry import (
    compose_pose,
    inverse_pose,
    tag_object_points,
    transform_points,
)


def test_pose_round_trip() -> None:
    pose = Pose(
        np.diag([1.0, -1.0, -1.0]),
        np.array([0.2, -0.4, 1.5]),
    )
    point = np.array([[0.1, 0.2, 0.3]])
    recovered = transform_points(inverse_pose(pose), transform_points(pose, point))
    np.testing.assert_allclose(recovered, point, atol=1e-12)


def test_pose_composition_matches_matrix_product() -> None:
    first = Pose(np.eye(3), np.array([1.0, 2.0, 3.0]))
    second = Pose(np.diag([1.0, -1.0, -1.0]), np.array([0.5, 0.0, 0.0]))
    composed = compose_pose(first, second)
    np.testing.assert_allclose(composed.matrix(), first.matrix() @ second.matrix())


def test_tag_corners_are_centered_and_counter_clockwise() -> None:
    corners = tag_object_points(0.2)
    np.testing.assert_allclose(corners.mean(axis=0), np.zeros(3))
    assert np.cross(corners[1] - corners[0], corners[2] - corners[1])[2] > 0.0
