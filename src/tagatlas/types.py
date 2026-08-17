"""Public data types used by TagAtlas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


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
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation", translation.copy())

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
        corners = np.asarray(self.corners, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError("corners must have shape (4, 2)")
        if not np.isfinite(corners).all():
            raise ValueError("corners must contain finite values")
        object.__setattr__(self, "corners", corners.copy())


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


@dataclass(frozen=True)
class LocalizerConfig:
    """Configuration for map discovery and graph updates."""

    reference_tag_id: int
    tag_sizes: Mapping[int, float]
    tag_family: str = "tag36h11"
    pixel_noise: float = 1.0
    max_reprojection_error: float = 8.0
    min_decision_margin: float = 0.0

    def __post_init__(self) -> None:
        sizes = {int(tag_id): float(size) for tag_id, size in self.tag_sizes.items()}
        if self.reference_tag_id not in sizes:
            raise ValueError("reference_tag_id must be present in tag_sizes")
        if any(size <= 0.0 for size in sizes.values()):
            raise ValueError("all tag sizes must be positive meters")
        if self.pixel_noise <= 0.0:
            raise ValueError("pixel_noise must be positive")
        if self.max_reprojection_error <= 0.0:
            raise ValueError("max_reprojection_error must be positive")
        object.__setattr__(self, "tag_sizes", sizes)

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> LocalizerConfig:
        """Create configuration from a YAML/JSON-like mapping."""

        raw_sizes = config.get("tag_sizes")
        if not isinstance(raw_sizes, Mapping):
            raise ValueError("config.tag_sizes must be a mapping")
        reference = cast(int | str, config.get("reference_tag_id", 0))
        family = cast(str, config.get("tag_family", "tag36h11"))
        pixel_noise = cast(float | str, config.get("pixel_noise", 1.0))
        max_error = cast(float | str, config.get("max_reprojection_error", 8.0))
        min_margin = cast(float | str, config.get("min_decision_margin", 0.0))
        return cls(
            reference_tag_id=int(reference),
            tag_sizes={int(key): float(value) for key, value in raw_sizes.items()},
            tag_family=str(family),
            pixel_noise=float(pixel_noise),
            max_reprojection_error=float(max_error),
            min_decision_margin=float(min_margin),
        )
