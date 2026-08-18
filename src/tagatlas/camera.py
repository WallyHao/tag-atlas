"""Camera calibration and projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera with OpenCV radtan distortion."""

    matrix: FloatArray
    distortion_coefficients: FloatArray | None = None

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        if matrix.shape != (3, 3):
            raise ValueError("camera matrix must have shape (3, 3)")
        distortion = np.array(
            self.distortion_coefficients
            if self.distortion_coefficients is not None
            else np.empty(0),
            dtype=np.float64,
            copy=True,
        ).reshape(-1)
        if distortion.size > 5:
            raise ValueError("radtan distortion accepts at most five coefficients")
        if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
            raise ValueError("camera calibration must contain finite values")
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        matrix.setflags(write=False)
        distortion.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)

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
        return self.project_with_jacobian(points)[0]

    def project_with_jacobian(
        self,
        points_camera: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Project points and return the pixel Jacobian for each point."""

        points = np.asarray(points_camera, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_camera must have shape (N, 3)")
        z = points[:, 2]
        if not np.isfinite(points).all():
            raise ValueError("points_camera must contain finite values")
        if np.any(z <= 1e-9):
            raise ValueError("points_camera must have positive depth")
        x = points[:, 0] / z
        y = points[:, 1] / z
        assert self.distortion_coefficients is not None
        coeffs = np.zeros(5, dtype=np.float64)
        coeffs[: self.distortion_coefficients.size] = self.distortion_coefficients

        radius2 = x * x + y * y
        radial = 1.0 + coeffs[0] * radius2 + coeffs[1] * radius2**2
        radial += coeffs[4] * radius2**3
        radial_gradient = coeffs[0] + 2.0 * coeffs[1] * radius2
        radial_gradient += 3.0 * coeffs[4] * radius2**2
        xd = x * radial + 2.0 * coeffs[2] * x * y + coeffs[3] * (radius2 + 2.0 * x * x)
        yd = y * radial + coeffs[2] * (radius2 + 2.0 * y * y) + 2.0 * coeffs[3] * x * y
        pixels = np.column_stack((self.fx * xd + self.cx, self.fy * yd + self.cy))

        dxd_dx = radial + 2.0 * x * x * radial_gradient
        dxd_dx += 2.0 * coeffs[2] * y + 6.0 * coeffs[3] * x
        dxd_dy = 2.0 * x * y * radial_gradient
        dxd_dy += 2.0 * coeffs[2] * x + 2.0 * coeffs[3] * y
        dyd_dx = 2.0 * x * y * radial_gradient
        dyd_dx += 2.0 * coeffs[2] * x + 2.0 * coeffs[3] * y
        dyd_dy = radial + 2.0 * y * y * radial_gradient
        dyd_dy += 6.0 * coeffs[2] * y + 2.0 * coeffs[3] * x

        normalized_jacobian = np.zeros((points.shape[0], 2, 3), dtype=np.float64)
        normalized_jacobian[:, 0, 0] = 1.0 / z
        normalized_jacobian[:, 0, 2] = -x / z
        normalized_jacobian[:, 1, 1] = 1.0 / z
        normalized_jacobian[:, 1, 2] = -y / z
        projection_jacobian = np.empty_like(normalized_jacobian)
        projection_jacobian[:, 0, :] = (
            self.fx * dxd_dx[:, None] * normalized_jacobian[:, 0, :]
            + self.fx * dxd_dy[:, None] * normalized_jacobian[:, 1, :]
        )
        projection_jacobian[:, 1, :] = (
            self.fy * dyd_dx[:, None] * normalized_jacobian[:, 0, :]
            + self.fy * dyd_dy[:, None] * normalized_jacobian[:, 1, :]
        )
        return pixels, projection_jacobian
