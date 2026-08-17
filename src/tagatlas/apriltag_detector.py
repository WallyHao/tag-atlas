"""AprilTag detector adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import cv2
import numpy as np

from .models import Detection


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
        previous = selected.get(detection.tag_id)
        if previous is None or detection.decision_margin > previous.decision_margin:
            selected[detection.tag_id] = detection
    return tuple(selected.values())


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
