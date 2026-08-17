"""Camera calibration and projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera with radtan or equidistant distortion."""

    matrix: FloatArray
    distortion_coefficients: FloatArray | None = None
    distortion_model: str = "radtan"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("camera matrix must have shape (3, 3)")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        distortion = np.asarray(
            self.distortion_coefficients
            if self.distortion_coefficients is not None
            else np.empty(0),
            dtype=np.float64,
        ).reshape(-1)
        model = self.distortion_model.lower()
        if model not in {"radtan", "plumb_bob", "equidistant", "fisheye"}:
            raise ValueError(f"unsupported distortion model: {self.distortion_model}")
        if model in {"equidistant", "fisheye"} and distortion.size > 4:
            raise ValueError("equidistant distortion accepts at most four coefficients")
        if model in {"radtan", "plumb_bob"} and distortion.size > 5:
            raise ValueError("radtan distortion accepts at most five coefficients")
        if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
            raise ValueError("camera calibration must contain finite values")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "distortion_coefficients", distortion.copy())
        object.__setattr__(self, "distortion_model", model)

    @property
    def fx(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.matrix[1, 2])

    def project(self, points_camera: FloatArray) -> FloatArray:
        """Project camera-frame 3D points into pixels."""

        points = np.asarray(points_camera, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_camera must have shape (N, 3)")
        z = points[:, 2]
        z_safe = np.where(z > 1e-9, z, 1e-9)
        x = points[:, 0] / z_safe
        y = points[:, 1] / z_safe
        assert self.distortion_coefficients is not None
        coeffs = np.zeros(5, dtype=np.float64)
        coeffs[: self.distortion_coefficients.size] = self.distortion_coefficients

        if self.distortion_model in {"equidistant", "fisheye"}:
            radius = np.sqrt(x * x + y * y)
            theta = np.arctan(radius)
            theta2 = theta * theta
            theta_distorted = theta * (
                1.0
                + coeffs[0] * theta2
                + coeffs[1] * theta2**2
                + coeffs[2] * theta2**3
                + coeffs[3] * theta2**4
            )
            scale = np.ones_like(radius)
            nonzero = radius > 1e-12
            scale[nonzero] = theta_distorted[nonzero] / radius[nonzero]
            xd = x * scale
            yd = y * scale
        else:
            radius2 = x * x + y * y
            radial = 1.0 + coeffs[0] * radius2 + coeffs[1] * radius2**2
            radial += coeffs[4] * radius2**3
            xd = (
                x * radial
                + 2.0 * coeffs[2] * x * y
                + coeffs[3] * (radius2 + 2.0 * x * x)
            )
            yd = (
                y * radial
                + coeffs[2] * (radius2 + 2.0 * y * y)
                + 2.0 * coeffs[3] * x * y
            )
        return np.column_stack((self.fx * xd + self.cx, self.fy * yd + self.cy))
