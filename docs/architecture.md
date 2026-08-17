# TagAtlas Architecture

## Coordinate Convention

TagAtlas names a transform `T_dst_src` when it converts coordinates from
`src` to `dst`:

```text
x_dst = T_dst_src x_src
```

The fixed reference tag is the world frame. `T_world_tag` is a tag-to-world
pose, `T_world_camera` is the returned camera pose, and a detector measurement
is `T_camera_tag`. The projection chain is:

```text
T_camera_tag = inverse(T_world_camera) @ T_world_tag
```

## Module Boundaries

- `models.py` contains public immutable data models and shared array types.
- `tag_map.py` contains the immutable static Tag map configuration.
- `camera.py` owns calibration validation and radtan projection.
- `geometry.py` owns rigid transforms, tag corner geometry, and OpenCV PnP.
- `apriltag_detector.py` adapts and filters detector output.
- `initialization.py` creates camera and new-tag pose seeds.
- `metrics.py` computes localization quality metrics.
- `pose_graph.py` owns GTSAM variables, factors, and incremental updates.
- `localizer.py` coordinates the stateful processing pipeline and exposes the public API.
- `logging_utils.py` provides opt-in package logging configuration.

Each tag has local corners at `(+/-size/2, +/-size/2, 0)`. The corner order
matches pupil-apriltags' counter-clockwise output.

## GTSAM Variables

- `t<tag_id>` is a static `gtsam.Pose3` tag pose.
- `c<frame_id>` is a per-frame `gtsam.Pose3` camera pose.
- The reference tag has a near-zero covariance identity prior.

Each image detection adds an eight-dimensional `gtsam.CustomFactor`. Its
residual is the four predicted pixel corners minus the four detected corners.
The factor connects one camera variable and one tag variable. Its Jacobians
are finite differences in the six-dimensional Pose3 tangent space, which keeps
the first implementation independent of distortion-specific analytic
derivatives.

## Frame Processing

1. Detect tags and normalize the detector output.
2. Drop tags with the wrong family, missing size, low decision margin, or
   duplicate IDs.
3. Use already mapped tags to initialize the current camera pose with PnP.
4. Initialize each connected new tag with tag-to-camera PnP and compose it
   with the camera world pose.
5. Add the camera value, new tag values, and projection factors to iSAM2.
6. Read optimized poses and compute pixel RMSE.

If a frame contains no mapped tag, it cannot determine both the camera pose and
an entirely new tag pose without an odometry or temporal motion factor. The
frame is therefore rejected rather than introducing an unobservable component
into the graph.

## Relation To TagSLAM

This design follows TagSLAM's `TagProjection` and `GraphUpdater` principles:
four corner projection constraints, one fixed gauge, PnP-based initialization,
and delayed insertion of disconnected measurements. The first Python version
uses a simpler two-pose factor (`camera`, `tag`) because the requested
configuration has one camera and one static world frame. Body poses,
multi-camera extrinsics, odometry, subgraph enumeration, and robust graph
rejection are later extensions.
