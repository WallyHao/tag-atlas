import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tagatlas import Detection
from tagatlas.apriltag_detector import (
    PupilAprilTagDetector,
    filter_detections,
)


def make_raw_detection(
    tag_id: int,
    family: str = "tag36h11",
    margin: float = 10.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        tag_id=tag_id,
        corners=np.zeros((4, 2)),
        tag_family=family,
        decision_margin=margin,
    )


def test_filter_detections_keeps_only_best_valid_detection() -> None:
    values = [
        make_raw_detection(1, margin=5.0),
        make_raw_detection(1, margin=20.0),
        make_raw_detection(2, family="tag25h9"),
        make_raw_detection(3),
        Detection(4, np.zeros((4, 2)), "tag36h11", decision_margin=2.0),
    ]

    selected = filter_detections(
        values,
        tag_family="tag36h11",
        tag_sizes={1: 0.12, 4: 0.12},
        min_decision_margin=10.0,
    )

    assert [detection.tag_id for detection in selected] == [1]
    assert selected[0].decision_margin == 20.0


def test_default_detector_normalizes_image_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDetector:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.images: list[np.ndarray] = []

        def detect(
            self, image: np.ndarray, *, estimate_tag_pose: bool
        ) -> list[SimpleNamespace]:
            assert not estimate_tag_pose
            self.images.append(image)
            return [make_raw_detection(1)]

    monkeypatch.setitem(
        sys.modules,
        "pupil_apriltags",
        SimpleNamespace(Detector=FakeDetector),
    )
    detector = PupilAprilTagDetector()

    grayscale = np.zeros((4, 4), dtype=np.uint8)
    bgr = np.zeros((4, 4, 3), dtype=np.float32)
    bgr[0, 0, 0] = 2.0
    bgra = np.ones((4, 4, 4), dtype=np.float32)

    assert len(detector.detect(grayscale)) == 1
    assert len(detector.detect(bgr)) == 1
    assert len(detector.detect(bgra)) == 1
    assert all(image.dtype == np.uint8 for image in detector._detector.images)


def test_default_detector_rejects_invalid_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "pupil_apriltags",
        SimpleNamespace(
            Detector=lambda **_: SimpleNamespace(detect=lambda *_args, **_kwargs: [])
        ),
    )
    detector = PupilAprilTagDetector()

    with pytest.raises(ValueError, match="image"):
        detector.detect(np.zeros((4, 4, 2), dtype=np.uint8))


def test_default_detector_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pupil_apriltags", None)

    with pytest.raises(RuntimeError, match="pupil-apriltags"):
        PupilAprilTagDetector()
