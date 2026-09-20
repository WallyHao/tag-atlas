# Verification

## Local Checks

Run the complete local validation with:

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

`uv run pytest` also enforces branch coverage with a minimum total threshold
of 90%.

The same checks run in `.github/workflows/ci.yml` for Python 3.11 and 3.12.

The current tests cover SE(3) round trips, camera projection, calibration
validation, detector normalization, connected new-tag discovery, and
unconnected-observation rejection.

## Synthetic Data

Synthetic tests should generate static tag poses and camera poses, project tag
corners through the configured camera model, and optionally add Gaussian pixel
noise or outliers. The expected metrics are camera translation error,
rotation error, tag translation/rotation error, and pixel RMSE.

The synthetic test is the authoritative test for transform direction and
factor construction because it has a known ground truth and does not depend on
the detector's native library.

## TagSLAM Data

The official TagSLAM root repository includes `example/example.bag` for its
quick test. The `tagslam_test` repository contains reference tests 1 through
21 and downloadable reference bags. The most relevant cases are:

- Test 1: 269-frame single-camera smoke test.
- Test 12: 4499 frames with one fixed tag and the remaining tags discovered.
- Tests 16 and 17: localization accuracy demonstrations.
- Test 18: moving-block multi-frame state estimation.

These bags use ROS topics and the original MIT detector, while the library uses
pupil-apriltags. They should therefore be evaluated in two stages: first use
the recorded tag corner messages to validate the optimizer, then extract image
frames and rerun pupil-apriltags for an end-to-end detector regression.

Full bags should remain external test data because of their size. A small,
versioned subset of images, detections, calibration, and expected poses should
be stored under `tests/data` with source URL, commit/version, checksum, and
license information. The repository currently has no checked-in or local
external dataset dependency; the default test suite is synthetic and self-
contained.

## Quality Gates

- Noisy synthetic sequences must remain below a documented pose-error and
  reprojection-error threshold.
- `uv run pytest` must maintain at least 90% source coverage.
- Rejected frames must not change the public map.
- A new tag must be added only after a connected initialization succeeds.
- Repeated optimization must not change the fixed reference tag pose.
- Dataset regressions must record frame count, detected tag count, factor
  count, RMSE, and final pose errors.
