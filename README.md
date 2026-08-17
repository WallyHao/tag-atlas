# TagAtlas

TagAtlas is a Python library for multi-AprilTag spatial localization, coordinate
transforms, and pose fusion.

## Development setup

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy src
```

## Configuration

The reference tag is fixed at the origin of the world frame. Every tag that
may be discovered must have a size in meters:

```python
config = {
    "reference_tag_id": 0,
    "tag_sizes": {0: 0.12, 1: 0.12, 2: 0.15},
}
```

Create a `CameraModel` from the calibrated intrinsic matrix and OpenCV radtan
distortion coefficients, then construct the localizer from the configuration:

```python
import cv2
import numpy as np

from tagatlas import CameraModel, Localizer

camera = CameraModel(
    matrix=np.array(
        [[603.9, 0.0, 665.0], [0.0, 603.2, 554.3], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    ),
    distortion_coefficients=np.array([-0.01, 0.001, 0.0, 0.0, 0.0], dtype=np.float64),
)
localizer = Localizer.from_config(config, camera)

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

For typed configuration, use `LocalizerConfig` directly with
`Localizer(config, camera)`. `Localizer` uses the `pupil-apriltags` detector and
a GTSAM factor graph. A new tag is added only after it is connected to the
reference tag or an already-mapped tag. Observations that cannot initialize a
connected subgraph are held out of the graph.

## Current features

- Per-tag size configuration
- Camera pose expressed in the reference-tag coordinate frame
- Multi-frame tag map discovery
- GTSAM iSAM2 incremental optimization
- Reprojection-error reporting

## Package layout

```text
src/tagatlas/
  models.py              Public data models
  camera.py              Camera calibration and projection
  geometry.py            SE(3), tag geometry, and PnP
  apriltag_detector.py   AprilTag detector adapter and filtering
  initialization.py      PnP pose seeding
  metrics.py             Reprojection quality metrics
  pose_graph.py          GTSAM incremental graph
  localizer.py           Localization pipeline facade
tests/                   Automated tests
docs/                    Architecture and verification notes
```

See [`docs/usage.md`](docs/usage.md) for the complete API workflow.

Large external datasets are not stored in the repository. Reproducible
regressions should use small fixtures under `tests/data` and document their
source and checksum.
