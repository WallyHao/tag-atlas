"""TagAtlas: multi-AprilTag spatial localization."""

from .camera import CameraModel
from .localizer import Localizer
from .types import Detection, LocalizationResult, LocalizerConfig, Pose

__all__ = [
    "CameraModel",
    "Detection",
    "Localizer",
    "LocalizerConfig",
    "LocalizationResult",
    "Pose",
]
__version__ = "0.1.0"
