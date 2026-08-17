# TagAtlas

TagAtlas is a Python library for multi-AprilTag spatial localization, coordinate
transforms, and pose fusion.

## Development Setup

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Usage

The public API uses explicit configuration objects. `TagMap` describes the
static Tag map, `LocalizerConfig` describes algorithm parameters, and
`Localizer` owns the stateful runtime graph.

```python
import cv2
import numpy as np

from tagatlas import CameraModel, Localizer, LocalizerConfig, TagMap

camera = CameraModel(
    matrix=np.array(
        [[603.9, 0.0, 665.0], [0.0, 603.2, 554.3], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    distortion_coefficients=np.array([-0.01, 0.001, 0.0, 0.0, 0.0], dtype=np.float64),
)

tag_map = TagMap(
    reference_tag_id=0,
    tag_sizes={0: 0.12, 1: 0.12, 2: 0.15},
)

localizer = Localizer(
    camera=camera,
    tag_map=tag_map,
    config=LocalizerConfig(robust_loss="huber"),
)

image = cv2.imread("frame.png")
if image is None:
    raise RuntimeError("unable to read frame.png")
result = localizer.locate(image)
if result.success:
    print(result.camera_pose)
    print(result.tag_poses)
else:
    print(f"frame rejected: {result.reason}")
```

The fixed reference Tag defines the world-frame origin. All declared Tag sizes
are in meters. Call `locate()` on sequential frames and `reset()` to clear the
trajectory and discovered map.

Calls on one `Localizer` instance are serialized. Use separate instances for
independent concurrent pipelines.

`CameraModel` supports a 3x3 pinhole matrix and zero to five OpenCV radtan
distortion coefficients in the order `k1, k2, p1, p2, k3`.

The default detector uses `pupil-apriltags`; a custom implementation can be
injected through the `detector` argument when constructing `Localizer`.

## Package Layout

```text
src/tagatlas/
  models.py              Public data models and algorithm configuration
  tag_map.py             Static Tag map configuration
  camera.py              Camera calibration and projection
  geometry.py            SE(3), tag geometry, and PnP
  apriltag_detector.py   AprilTag detector adapter and filtering
  initialization.py      PnP pose seeding
  metrics.py             Reprojection quality metrics
  pose_graph.py          GTSAM incremental graph
  logging_utils.py       Application logging configuration
  localizer.py           Stateful localization pipeline
tests/                   Automated tests
```

See [`docs/usage.md`](docs/usage.md) for the complete workflow.
