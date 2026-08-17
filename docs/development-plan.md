# TagAtlas Development Plan

## Scope

TagAtlas is a multi-frame AprilTag localization library. The first version
uses `pupil-apriltags` for corner detections and GTSAM for the nonlinear
factor graph. The configured reference tag is fixed at the world-frame
origin. Other tags are static variables that can be discovered and optimized
as observations connect them to the existing map.

## Completed Foundation

- The project uses `uv` and stores the resolved environment in `uv.lock`.
- Runtime dependencies include GTSAM, OpenCV, NumPy, and pupil-apriltags.
- Development checks are Ruff, Pytest, and strict Mypy.
- `Pose`, `Detection`, `CameraModel`, and `LocalizerConfig` are typed public
  data structures.
- Camera projection supports OpenCV-style radtan distortion.
- The GTSAM graph uses `Pose3` values, a fixed reference-tag prior, and
  incremental `ISAM2` updates.
- A tag observation is represented by one eight-dimensional custom factor for
  its four pixel corners. Jacobians are calculated in Pose3 local coordinates.
- A tag is not added when the current frame has no connection to the known
  map.

## Implementation Stages

1. Freeze the transform convention and configuration schema.
2. Validate camera calibration and tag geometry independently of GTSAM.
3. Validate the pupil-apriltags adapter with detector doubles and real images.
4. Add single-reference-tag PnP initialization.
5. Add connected unknown-tag initialization and graph insertion.
6. Add multi-frame camera variables and iSAM2 updates.
7. Add observation quality gates, robust noise, and graph diagnostics.
8. Add optional full-graph optimization and map serialization.
9. Run synthetic and TagSLAM regression datasets.

## Configuration

The minimum mapping configuration is:

```python
config = {
    "reference_tag_id": 0,
    "tag_sizes": {0: 0.12, 1: 0.12, 2: 0.15},
    "tag_family": "tag36h11",
    "pixel_noise": 1.0,
    "max_reprojection_error": 8.0,
}
```

The camera calibration is passed as a `CameraModel`. Tag dimensions are in
meters and image coordinates are in pixels.

## Acceptance Criteria

- A reference tag alone initializes a camera pose.
- A frame containing a known tag and a new configured tag adds the new tag to
  the graph.
- A frame containing only unconnected unknown tags is rejected without adding
  factors.
- A noise-free synthetic sequence recovers camera and tag poses to numerical
  precision.
- Pixel RMSE, used tag IDs, and the optimized map are returned to callers.
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .`, and
  `uv run mypy src` all pass.
