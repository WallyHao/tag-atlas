# Usage

## Install

```bash
uv sync
```

The default detector is provided by `pupil-apriltags`. A custom detector can
be injected for a different detector implementation or for testing.

## Build The Camera Model

`CameraModel` expects a 3x3 pinhole intrinsic matrix and zero to five OpenCV
radtan coefficients in this order:

```text
k1, k2, p1, p2, k3
```

Equidistant/fisheye calibration is not supported. Convert the calibration to
the radtan model before constructing the camera model.

```python
import numpy as np

from tagatlas import CameraModel

camera = CameraModel(
    matrix=np.array(
        [[603.9, 0.0, 665.0], [0.0, 603.2, 554.3], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    distortion_coefficients=np.array([-0.01, 0.001, 0.0, 0.0, 0.0], dtype=np.float64),
)
```

## Configure The Map

The reference tag is the world-frame origin. All tags that may be added to the
map must be listed with their physical size in meters.

```python
from tagatlas import LocalizerConfig

config = LocalizerConfig(
    reference_tag_id=0,
    tag_sizes={0: 0.12, 1: 0.12, 2: 0.15},
    tag_family="tag36h11",
    pixel_noise=1.0,
    max_reprojection_error=8.0,
)
```

For YAML/JSON-like mappings, use `Localizer.from_config()` instead:

```python
from tagatlas import Localizer

localizer = Localizer.from_config(
    {
        "reference_tag_id": 0,
        "tag_sizes": {0: 0.12, 1: 0.12, 2: 0.15},
    },
    camera,
)
```

For the typed form, pass the `LocalizerConfig` and `CameraModel` directly:

```python
from tagatlas import Localizer

localizer = Localizer(config, camera)
```

## Process Frames

`locate()` accepts a grayscale, BGR, or BGRA NumPy image. It returns a
`LocalizationResult` for every frame, including rejected frames.

```python
import cv2

image = cv2.imread("frame.png")
if image is None:
    raise RuntimeError("unable to read frame.png")

result = localizer.locate(image)
if result.success:
    assert result.camera_pose is not None
    print("camera pose:", result.camera_pose)
    print("mapped tags:", result.tag_poses.keys())
    print("used tags:", result.used_tag_ids)
    print("pixel RMSE:", result.reprojection_rmse)
else:
    print("rejected:", result.reason)
```

Call `locate()` repeatedly on sequential frames. The camera pose is expressed
in the reference-tag world frame, and `tag_poses` contains the latest map.
Call `reset()` to clear the trajectory and map while keeping calibration and
configuration.
