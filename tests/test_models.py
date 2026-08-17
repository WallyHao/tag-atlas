import numpy as np
import pytest

from tagatlas import Detection, LocalizerConfig, Pose


def test_pose_validates_shapes_and_exposes_identity_matrix() -> None:
    with pytest.raises(ValueError, match="rotation"):
        Pose(np.eye(2), np.zeros(3))
    with pytest.raises(ValueError, match="translation"):
        Pose(np.eye(3), np.zeros(2))

    identity = Pose.identity()
    np.testing.assert_allclose(identity.matrix(), np.eye(4))


def test_detection_validates_corner_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        Detection(1, np.zeros((3, 2)), "tag36h11")
    with pytest.raises(ValueError, match="finite"):
        Detection(1, np.full((4, 2), np.nan), "tag36h11")


def test_config_normalizes_mapping_values() -> None:
    config = LocalizerConfig.from_mapping(
        {
            "reference_tag_id": "1",
            "tag_sizes": {"1": "0.12", "2": 0.15},
            "tag_family": "tag25h9",
            "pixel_noise": "2.0",
            "max_reprojection_error": "4.0",
            "min_decision_margin": "10.0",
        }
    )

    assert config.reference_tag_id == 1
    assert config.tag_sizes == {1: 0.12, 2: 0.15}
    assert config.tag_family == "tag25h9"
    assert config.pixel_noise == 2.0
    assert config.max_reprojection_error == 4.0
    assert config.min_decision_margin == 10.0


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="tag_sizes"):
        LocalizerConfig.from_mapping({})
    with pytest.raises(ValueError, match="reference_tag_id"):
        LocalizerConfig(reference_tag_id=0, tag_sizes={1: 0.12})
    with pytest.raises(ValueError, match="positive meters"):
        LocalizerConfig(reference_tag_id=0, tag_sizes={0: 0.0})
    with pytest.raises(ValueError, match="pixel_noise"):
        LocalizerConfig(reference_tag_id=0, tag_sizes={0: 0.12}, pixel_noise=0.0)
    with pytest.raises(ValueError, match="max_reprojection_error"):
        LocalizerConfig(
            reference_tag_id=0,
            tag_sizes={0: 0.12},
            max_reprojection_error=0.0,
        )
