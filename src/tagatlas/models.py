"""Public data types used by TagAtlas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


def _parse_tag_id(value: object, field_name: str) -> int:
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    return _tag_id(value, field_name)


def _float_value(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        return float(cast(float | int | str, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc


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
class LocalizationDiagnostics:
    """Structured feedback about one localization attempt."""

    detected_count: int = 0
    configured_count: int = 0
    accepted_count: int = 0
    used_tag_ids: tuple[int, ...] = ()
    rejected_tag_ids: tuple[int, ...] = ()
    new_tag_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        counts = (
            ("detected_count", self.detected_count),
            ("configured_count", self.configured_count),
            ("accepted_count", self.accepted_count),
        )
        for name, value in counts:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a non-negative integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.configured_count > self.detected_count:
            raise ValueError("configured_count cannot exceed detected_count")
        if self.accepted_count > self.configured_count:
            raise ValueError("accepted_count cannot exceed configured_count")
        object.__setattr__(self, "detected_count", int(self.detected_count))
        object.__setattr__(self, "configured_count", int(self.configured_count))
        object.__setattr__(self, "accepted_count", int(self.accepted_count))
        object.__setattr__(self, "used_tag_ids", tuple(self.used_tag_ids))
        object.__setattr__(self, "rejected_tag_ids", tuple(self.rejected_tag_ids))
        object.__setattr__(self, "new_tag_ids", tuple(self.new_tag_ids))


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
    diagnostics: LocalizationDiagnostics = LocalizationDiagnostics()
    pose_covariance: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.success and self.camera_pose is None:
            raise ValueError("successful results must include camera_pose")
        if not self.success and self.camera_pose is not None:
            raise ValueError("failed results must not include camera_pose")
        if self.success and self.reprojection_rmse is None:
            raise ValueError("successful results must include reprojection_rmse")
        if not self.success and not self.reason:
            raise ValueError("failed results must include a reason")
        if self.reprojection_rmse is not None and not np.isfinite(
            self.reprojection_rmse
        ):
            raise ValueError("reprojection_rmse must be finite when provided")
        if self.pose_covariance is not None:
            covariance = np.asarray(self.pose_covariance, dtype=np.float64)
            if covariance.shape != (6, 6):
                raise ValueError("pose_covariance must have shape (6, 6)")
            if not np.isfinite(covariance).all():
                raise ValueError("pose_covariance must contain finite values")
            object.__setattr__(self, "pose_covariance", _readonly_array(covariance))
        object.__setattr__(self, "tag_poses", MappingProxyType(dict(self.tag_poses)))
        object.__setattr__(self, "used_tag_ids", tuple(self.used_tag_ids))


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration passed to the default pupil-apriltags detector."""

    nthreads: int = 1
    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: int = 1
    decode_sharpening: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.nthreads, bool) or self.nthreads < 1:
            raise ValueError("nthreads must be a positive integer")
        if isinstance(self.refine_edges, bool) or self.refine_edges not in {0, 1}:
            raise ValueError("refine_edges must be 0 or 1")
        values = (
            ("quad_decimate", self.quad_decimate, 0.0),
            ("quad_sigma", self.quad_sigma, -np.inf),
            ("decode_sharpening", self.decode_sharpening, -np.inf),
        )
        for name, value, lower_bound in values:
            normalized = _float_value(value, name)
            if not np.isfinite(normalized) or normalized <= lower_bound:
                raise ValueError(
                    f"{name} must be finite and greater than {lower_bound}"
                )
            object.__setattr__(self, name, normalized)
        object.__setattr__(self, "nthreads", int(self.nthreads))
        object.__setattr__(self, "refine_edges", int(self.refine_edges))


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
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    map_mode: str = "discover"

    def __post_init__(self) -> None:
        if not isinstance(self.tag_family, str) or not self.tag_family:
            raise ValueError("tag_family must be a non-empty string")
        pixel_noise = _float_value(self.pixel_noise, "pixel_noise")
        max_error = _float_value(self.max_reprojection_error, "max_reprojection_error")
        min_margin = _float_value(self.min_decision_margin, "min_decision_margin")
        min_area = _float_value(self.min_tag_area, "min_tag_area")
        robust_scale = _float_value(self.robust_scale, "robust_scale")
        finite_positive = (
            ("pixel_noise", pixel_noise),
            ("max_reprojection_error", max_error),
            ("robust_scale", robust_scale),
        )
        for name, value in finite_positive:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(min_margin) or min_margin < 0.0:
            raise ValueError("min_decision_margin must not be negative")
        if not np.isfinite(min_area) or min_area <= 0.0:
            raise ValueError("min_tag_area must be positive and finite")
        if not isinstance(self.robust_loss, str):
            raise ValueError("robust_loss must be a string")
        if self.robust_loss not in {"none", "huber", "cauchy"}:
            raise ValueError("robust_loss must be none, huber, or cauchy")
        if not isinstance(self.map_mode, str) or self.map_mode not in {
            "discover",
            "fixed",
        }:
            raise ValueError("map_mode must be discover or fixed")
        object.__setattr__(self, "pixel_noise", pixel_noise)
        object.__setattr__(self, "max_reprojection_error", max_error)
        object.__setattr__(self, "min_decision_margin", min_margin)
        object.__setattr__(self, "min_tag_area", min_area)
        object.__setattr__(self, "robust_scale", robust_scale)
        if not isinstance(self.detector, DetectorConfig):
            raise ValueError("detector must be a DetectorConfig")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> LocalizerConfig:
        """Create configuration from a YAML/JSON-like mapping."""

        allowed = {
            "tag_family",
            "pixel_noise",
            "max_reprojection_error",
            "min_decision_margin",
            "min_tag_area",
            "robust_loss",
            "robust_scale",
            "detector",
            "map_mode",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"unknown localizer configuration fields: {sorted(unknown)}"
            )
        family = config.get("tag_family", "tag36h11")
        robust_loss = config.get("robust_loss", "huber")
        detector = config.get("detector", {})
        if not isinstance(family, str) or not family:
            raise ValueError("tag_family must be a non-empty string")
        if not isinstance(robust_loss, str):
            raise ValueError("robust_loss must be a string")
        if not isinstance(detector, Mapping):
            raise ValueError("detector must be a mapping")
        detector_config = DetectorConfig(**detector)
        map_mode = config.get("map_mode", "discover")
        if not isinstance(map_mode, str):
            raise ValueError("map_mode must be discover or fixed")
        return cls(
            tag_family=family,
            pixel_noise=_float_value(config.get("pixel_noise", 1.0), "pixel_noise"),
            max_reprojection_error=_float_value(
                config.get("max_reprojection_error", 8.0),
                "max_reprojection_error",
            ),
            min_decision_margin=_float_value(
                config.get("min_decision_margin", 0.0),
                "min_decision_margin",
            ),
            min_tag_area=_float_value(config.get("min_tag_area", 16.0), "min_tag_area"),
            robust_loss=robust_loss,
            robust_scale=_float_value(
                config.get("robust_scale", 1.345), "robust_scale"
            ),
            detector=detector_config,
            map_mode=map_mode,
        )
