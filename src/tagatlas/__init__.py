"""TagAtlas: multi-AprilTag spatial localization."""

import logging

from .camera import CameraModel
from .localizer import Localizer
from .logging_utils import configure_logging
from .models import Detection, LocalizationResult, LocalizerConfig, Pose
from .tag_map import TagMap

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "CameraModel",
    "configure_logging",
    "Detection",
    "Localizer",
    "LocalizerConfig",
    "LocalizationResult",
    "Pose",
    "TagMap",
]
__version__ = "0.1.0"
