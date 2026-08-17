# Usage

## Install

```bash
uv sync
```

The default detector is provided by `pupil-apriltags`. A custom detector can
be injected for a different implementation or for testing.

## Configure The Camera

`CameraModel` expects a 3x3 pinhole intrinsic matrix and zero to five OpenCV
radtan coefficients:

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

Equidistant and fisheye calibration models are not supported.

## Configure The Tag Map

`TagMap` explicitly identifies the world-frame reference Tag and the physical
size of every Tag that may be observed:

```python
from tagatlas import TagMap

tag_map = TagMap(
    reference_tag_id=0,
    tag_sizes={0: 0.12, 1: 0.12, 2: 0.15},
)
```

For YAML/JSON-like data, use:

```python
tag_map = TagMap.from_mapping(
    {
        "reference_tag_id": 0,
        "tag_sizes": {0: 0.12, 1: 0.12, 2: 0.15},
    }
)
```

## Configure The Localizer

`LocalizerConfig` contains algorithm parameters only:

```python
from tagatlas import LocalizerConfig

config = LocalizerConfig(
    tag_family="tag36h11",
    pixel_noise=1.0,
    max_reprojection_error=8.0,
    min_tag_area=16.0,
    robust_loss="huber",
    robust_scale=1.345,
)
```

It can also be loaded from a mapping with `LocalizerConfig.from_mapping()`.

## Process Frames

Construct the stateful runtime explicitly:

```python
from tagatlas import Localizer

localizer = Localizer(
    camera=camera,
    tag_map=tag_map,
    config=config,
)
```

`locate()` accepts grayscale, BGR, or BGRA NumPy images and returns a result for
every frame, including rejected frames:

```python
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

The camera pose is expressed in the reference-Tag world frame. Call `reset()`
to clear the trajectory and map while keeping calibration and configuration.

## Logging

TagAtlas uses the standard library `logging` package and does not emit output
unless configured by the application:

```python
import logging

from tagatlas import configure_logging

configure_logging(logging.INFO)
```

Applications that already manage handlers can configure the `tagatlas` logger
directly.
