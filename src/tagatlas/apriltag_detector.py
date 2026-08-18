"""AprilTag detector adapter."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import Detection

logger = logging.getLogger(__name__)


def _is_valid_quad(corners: NDArray[np.float64], min_area: float) -> bool:
    edges = np.roll(corners, -1, axis=0) - corners
    next_edges = np.roll(edges, -1, axis=0)
    cross_products = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    area = 0.5 * abs(
        float(np.dot(corners[:, 0], np.roll(corners[:, 1], -1)))
        - float(np.dot(corners[:, 1], np.roll(corners[:, 0], -1)))
    )
    same_orientation = bool(
        np.all(cross_products > 1e-9) or np.all(cross_products < -1e-9)
    )
    return bool(area >= min_area and same_orientation)


class DetectorProtocol(Protocol):
    """Protocol implemented by the detector used by :class:`Localizer`."""

    def detect(self, image: np.ndarray[Any, Any]) -> Sequence[Detection]:
        """Detect tags in a grayscale or color image."""


def _family_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def normalize_detection(value: Any) -> Detection:
    """Convert a pupil-apriltags detection or test double."""

    return Detection(
        tag_id=int(value.tag_id),
        corners=np.asarray(value.corners, dtype=np.float64),
        tag_family=_family_name(value.tag_family),
        decision_margin=float(getattr(value, "decision_margin", 0.0)),
        hamming=int(getattr(value, "hamming", 0)),
    )


def filter_detections(
    values: Sequence[Any],
    *,
    tag_family: str,
    tag_sizes: Mapping[int, float],
    min_decision_margin: float,
    min_tag_area: float = 16.0,
) -> tuple[Detection, ...]:
    """Normalize and filter detections accepted by the localizer."""

    selected: dict[int, Detection] = {}
    for value in values:
        detection = (
            value if isinstance(value, Detection) else normalize_detection(value)
        )
        if detection.tag_family != tag_family:
            continue
        if detection.tag_id not in tag_sizes:
            continue
        if detection.decision_margin < min_decision_margin:
            continue
        if not _is_valid_quad(detection.corners, min_tag_area):
            continue
        previous = selected.get(detection.tag_id)
        if previous is None or detection.decision_margin > previous.decision_margin:
            selected[detection.tag_id] = detection
    result = tuple(selected.values())
    logger.debug(
        "filtered AprilTag detections input=%d accepted=%d",
        len(values),
        len(result),
    )
    return result


class PupilAprilTagDetector:
    """Thin adapter around ``pupil_apriltags.Detector``."""

    def __init__(
        self,
        families: str = "tag36h11",
        nthreads: int = 1,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ) -> None:
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:
            raise RuntimeError(
                "pupil-apriltags is required to construct the default detector"
            ) from exc
        self._detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=0,
        )

    def detect(self, image: np.ndarray[Any, Any]) -> Sequence[Detection]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty NumPy array")
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("image must contain finite numeric values")
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in {3, 4}:
            code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            gray = cv2.cvtColor(image, code)
        else:
            raise ValueError("image must be grayscale, BGR, or BGRA")
        if gray.dtype != np.uint8:
            low = float(np.min(gray))
            high = float(np.max(gray))
            if high > low:
                gray = ((gray - low) * (255.0 / (high - low))).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)
        raw = self._detector.detect(gray, estimate_tag_pose=False)
        return tuple(normalize_detection(item) for item in raw)
