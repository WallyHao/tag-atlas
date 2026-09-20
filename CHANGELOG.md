# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- PEP 561 type information via `py.typed`, so downstream type checkers can use
  the package's inline annotations.
- Project metadata: homepage, repository, issue tracker, documentation and
  changelog URLs, plus keywords and PyPI classifiers.
- `README`, `CONTRIBUTING` and `CHANGELOG` documentation.

## [0.1.0] - 2026-08-18

### Added

- Initial release: multi-AprilTag 6-DoF localization with GTSAM `ISAM2`,
  analytic `Pose3` projection Jacobians, PnP initialization, online map
  discovery and fixed-map localization.
- Configurable Huber and Cauchy robust losses.
- Immutable, validated public data models with JSON map persistence.
- Synthetic and unit test suite with a 90% branch-coverage gate and CI on
  Python 3.11 and 3.12.
