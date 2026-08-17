# Performance

The GTSAM custom factor uses analytic Pose3 Jacobians in GTSAM's right-retraction
local coordinates. This avoids six residual evaluations per pose and keeps the
projection derivative consistent with the configured radtan camera model.

Performance measurements should report:

- Python version and dependency versions.
- Image dimensions and detector implementation.
- Number of frames and configured/discovered Tags.
- Per-frame latency percentiles, not only the average.
- Peak memory and final factor count.
- Reprojection RMSE and pose error alongside timing.

Benchmarks should use deterministic synthetic detections first, then a small
versioned real-image fixture. They must not be part of the coverage-gated unit
test command.
