import numpy as np
import pytest

from tagatlas import CameraModel


def test_pinhole_projection_without_distortion() -> None:
    camera = CameraModel(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    )
    pixels = camera.project(np.array([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0]]))
    np.testing.assert_allclose(pixels, [[320.0, 240.0], [570.0, 115.0]])


def test_camera_rejects_invalid_calibration() -> None:
    with pytest.raises(ValueError, match="shape"):
        CameraModel(np.eye(2))
    with pytest.raises(ValueError, match="distortion"):
        CameraModel(np.eye(3), np.zeros(6))
