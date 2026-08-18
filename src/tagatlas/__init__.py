"""TagAtlas: multi-AprilTag spatial localization."""

import logging

from ._version import __version__
from .camera import CameraModel
from .localizer import Localizer
from .logging_utils import configure_logging
from .models import (
    Detection,
    DetectorConfig,
    LocalizationDiagnostics,
    LocalizationResult,
    LocalizerConfig,
    Pose,
)
from .tag_map import TagMap

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CameraModel",
    "configure_logging",
    "Detection",
    "DetectorConfig",
    "LocalizationDiagnostics",
    "Localizer",
    "LocalizerConfig",
    "LocalizationResult",
    "Pose",
    "TagMap",
    "__version__",
]
