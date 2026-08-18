import numpy as np
import pytest

from tagatlas import (
    Detection,
    DetectorConfig,
    LocalizationResult,
    LocalizerConfig,
    Pose,
    TagMap,
)


def test_pose_validates_shapes_and_exposes_identity_matrix() -> None:
    with pytest.raises(ValueError, match="rotation"):
        Pose(np.eye(2), np.zeros(3))
    with pytest.raises(ValueError, match="translation"):
        Pose(np.eye(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        Pose(np.full((3, 3), np.nan), np.zeros(3))
    with pytest.raises(ValueError, match="orthonormal"):
        Pose(np.diag([2.0, 1.0, 1.0]), np.zeros(3))
    with pytest.raises(ValueError, match="determinant"):
        Pose(np.diag([-1.0, 1.0, 1.0]), np.zeros(3))

    identity = Pose.identity()
    np.testing.assert_allclose(identity.matrix(), np.eye(4))


def test_detection_validates_corner_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        Detection(1, np.zeros((3, 2)), "tag36h11")
    with pytest.raises(ValueError, match="finite"):
        Detection(1, np.full((4, 2), np.nan), "tag36h11")
    with pytest.raises(ValueError, match="integer"):
        Detection(1.5, np.zeros((4, 2)), "tag36h11")
    with pytest.raises(ValueError, match="finite"):
        Detection(1, np.zeros((4, 2)), "tag36h11", decision_margin=float("nan"))


def test_public_models_are_deeply_immutable() -> None:
    pose = Pose.identity()
    tag_map = TagMap(reference_tag_id=0, tag_sizes={0: 0.12})

    with pytest.raises(ValueError):
        pose.rotation[0, 0] = 0.0
    with pytest.raises(TypeError):
        tag_map.tag_sizes[1] = 0.12  # type: ignore[index]


def test_localization_result_is_deeply_immutable() -> None:
    result = LocalizationResult(
        success=True,
        frame_id=0,
        camera_pose=Pose.identity(),
        tag_poses={0: Pose.identity()},
        used_tag_ids=[0],
        reprojection_rmse=0.0,
    )

    with pytest.raises(TypeError):
        result.tag_poses[1] = Pose.identity()  # type: ignore[index]
    with pytest.raises(ValueError, match="camera_pose"):
        LocalizationResult(
            success=True,
            frame_id=0,
            camera_pose=None,
            tag_poses={},
            used_tag_ids=(),
            reprojection_rmse=0.0,
        )

    result_with_covariance = LocalizationResult(
        success=True,
        frame_id=0,
        camera_pose=Pose.identity(),
        tag_poses={0: Pose.identity()},
        used_tag_ids=(0,),
        reprojection_rmse=0.0,
        pose_covariance=np.eye(6),
    )
    with pytest.raises(ValueError):
        result_with_covariance.pose_covariance[0, 0] = 0.0  # type: ignore[index]


def test_config_normalizes_mapping_values() -> None:
    tag_map = TagMap.from_mapping(
        {
            "reference_tag_id": "1",
            "tag_sizes": {"1": "0.12", "2": 0.15},
        }
    )
    config = LocalizerConfig.from_mapping(
        {
            "tag_family": "tag25h9",
            "pixel_noise": "2.0",
            "max_reprojection_error": "4.0",
            "min_decision_margin": "10.0",
        }
    )

    assert tag_map.reference_tag_id == 1
    assert tag_map.tag_sizes == {1: 0.12, 2: 0.15}
    assert config.tag_family == "tag25h9"
    assert config.pixel_noise == 2.0
    assert config.max_reprojection_error == 4.0
    assert config.min_decision_margin == 10.0


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="tag_sizes"):
        TagMap.from_mapping({})
    with pytest.raises(ValueError, match="reference_tag_id"):
        TagMap(reference_tag_id=0, tag_sizes={1: 0.12})
    with pytest.raises(ValueError, match="positive meters"):
        TagMap(reference_tag_id=0, tag_sizes={0: 0.0})
    with pytest.raises(ValueError, match="positive meters"):
        TagMap(reference_tag_id=0, tag_sizes={0: float("nan")})
    with pytest.raises(ValueError, match="pixel_noise"):
        LocalizerConfig.from_mapping({"pixel_noise": 0.0})
    with pytest.raises(ValueError, match="max_reprojection_error"):
        LocalizerConfig(max_reprojection_error=0.0)
    with pytest.raises(ValueError, match="min_decision_margin"):
        LocalizerConfig(min_decision_margin=-1.0)
    with pytest.raises(ValueError, match="min_tag_area"):
        LocalizerConfig(min_tag_area=0.0)
    with pytest.raises(ValueError, match="robust_loss"):
        LocalizerConfig(robust_loss="invalid")
    with pytest.raises(ValueError, match="robust_scale"):
        LocalizerConfig(robust_scale=0.0)
    with pytest.raises(ValueError, match="finite"):
        LocalizerConfig(pixel_noise=float("inf"))
    with pytest.raises(ValueError, match="unknown"):
        LocalizerConfig.from_mapping({"pixel_niose": 1.0})


def test_detector_config_is_loaded_from_mapping() -> None:
    config = LocalizerConfig.from_mapping(
        {"detector": {"nthreads": 2, "quad_decimate": 2.0}}
    )

    assert config.detector == DetectorConfig(nthreads=2, quad_decimate=2.0)


def test_detector_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="nthreads"):
        DetectorConfig(nthreads=0)
    with pytest.raises(ValueError, match="quad_decimate"):
        DetectorConfig(quad_decimate=0.0)
