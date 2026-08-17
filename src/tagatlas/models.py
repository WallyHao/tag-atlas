"""Public data types used by TagAtlas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _readonly_array(value: object) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _tag_id(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer")
    return int(value)


@dataclass(frozen=True)
class Pose:
    """Rigid transform represented by a rotation matrix and translation."""

    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("pose must contain finite values")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("rotation must have determinant 1")
        object.__setattr__(self, "rotation", _readonly_array(rotation))
        object.__setattr__(self, "translation", _readonly_array(translation))

    @classmethod
    def identity(cls) -> Pose:
        """Return the identity transform."""

        return cls(np.eye(3), np.zeros(3))

    def matrix(self) -> FloatArray:
        """Return the homogeneous 4x4 transform matrix."""

        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.translation
        return result


@dataclass(frozen=True)
class Detection:
    """Normalized output from an AprilTag detector."""

    tag_id: int
    corners: FloatArray
    tag_family: str
    decision_margin: float = 0.0
    hamming: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag_id", _tag_id(self.tag_id, "tag_id"))
        decision_margin = float(self.decision_margin)
        if not np.isfinite(decision_margin):
            raise ValueError("decision_margin must be finite")
        hamming = _tag_id(self.hamming, "hamming")
        if hamming < 0:
            raise ValueError("hamming must not be negative")
        corners = np.asarray(self.corners, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError("corners must have shape (4, 2)")
        if not np.isfinite(corners).all():
            raise ValueError("corners must contain finite values")
        object.__setattr__(self, "decision_margin", decision_margin)
        object.__setattr__(self, "hamming", hamming)
        object.__setattr__(self, "corners", _readonly_array(corners))


@dataclass(frozen=True)
class LocalizationResult:
    """Result of processing one image."""

    success: bool
    frame_id: int
    camera_pose: Pose | None
    tag_poses: Mapping[int, Pose]
    used_tag_ids: tuple[int, ...]
    reprojection_rmse: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.reprojection_rmse is not None and not np.isfinite(
            self.reprojection_rmse
        ):
            raise ValueError("reprojection_rmse must be finite when provided")
        object.__setattr__(self, "tag_poses", MappingProxyType(dict(self.tag_poses)))
        object.__setattr__(self, "used_tag_ids", tuple(self.used_tag_ids))


@dataclass(frozen=True)
class LocalizerConfig:
    """Configuration for map discovery and graph updates."""

    tag_family: str = "tag36h11"
    pixel_noise: float = 1.0
    max_reprojection_error: float = 8.0
    min_decision_margin: float = 0.0
    min_tag_area: float = 16.0
    robust_loss: str = "huber"
    robust_scale: float = 1.345

    def __post_init__(self) -> None:
        finite_positive = (
            ("pixel_noise", self.pixel_noise),
            ("max_reprojection_error", self.max_reprojection_error),
            ("robust_scale", self.robust_scale),
        )
        for name, value in finite_positive:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.min_decision_margin) or self.min_decision_margin < 0.0:
            raise ValueError("min_decision_margin must not be negative")
        if not np.isfinite(self.min_tag_area) or self.min_tag_area <= 0.0:
            raise ValueError("min_tag_area must be positive and finite")
        if self.robust_loss not in {"none", "huber", "cauchy"}:
            raise ValueError("robust_loss must be none, huber, or cauchy")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> LocalizerConfig:
        """Create configuration from a YAML/JSON-like mapping."""

        family = cast(str, config.get("tag_family", "tag36h11"))
        pixel_noise = cast(float | str, config.get("pixel_noise", 1.0))
        max_error = cast(float | str, config.get("max_reprojection_error", 8.0))
        min_margin = cast(float | str, config.get("min_decision_margin", 0.0))
        min_area = cast(float | str, config.get("min_tag_area", 16.0))
        robust_loss = cast(str, config.get("robust_loss", "huber"))
        robust_scale = cast(float | str, config.get("robust_scale", 1.345))
        return cls(
            tag_family=str(family),
            pixel_noise=float(pixel_noise),
            max_reprojection_error=float(max_error),
            min_decision_margin=float(min_margin),
            min_tag_area=float(min_area),
            robust_loss=str(robust_loss),
            robust_scale=float(robust_scale),
        )
