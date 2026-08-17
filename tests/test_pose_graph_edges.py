import numpy as np
import pytest

import tagatlas.pose_graph as pose_graph
from tagatlas import CameraModel, Pose


def test_projection_constraint_validates_corners() -> None:
    with pytest.raises(ValueError, match="image_corners"):
        pose_graph.ProjectionConstraint(1, 2, 0.12, np.zeros((3, 2)))


def test_graph_requires_gtsam_and_exposes_fallback_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pose_graph, "gtsam", None)

    with pytest.raises(RuntimeError, match="GTSAM is required"):
        pose_graph.GtsamGraph(0, 1.0, CameraModel(np.eye(3)))
    assert pose_graph.GtsamGraph.camera_key(3) == 3
    assert pose_graph.GtsamGraph.tag_key(4) == 4


def test_graph_rejects_factor_operations_without_gtsam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pose_graph, "gtsam", None)
    graph = pose_graph.GtsamGraph.__new__(pose_graph.GtsamGraph)
    constraint = pose_graph.ProjectionConstraint(1, 2, 0.12, np.zeros((4, 2)))

    with pytest.raises(RuntimeError, match="unavailable"):
        graph._make_factor(constraint)
    with pytest.raises(RuntimeError, match="unavailable"):
        graph.update(1, Pose.identity(), [], [])


def test_graph_skips_already_known_tag() -> None:
    graph = pose_graph.GtsamGraph(0, 1.0, CameraModel(np.eye(3)))
    reference_key = pose_graph.GtsamGraph.tag_key(0)

    graph.update(
        pose_graph.GtsamGraph.camera_key(0),
        Pose.identity(),
        [(0, Pose.identity())],
        [],
    )

    assert reference_key in graph.known_keys
    assert graph.factor_count == 0
