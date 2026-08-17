# Real-Image Fixtures

Real detector regression fixtures belong in this directory. Each fixture must
include:

- Source URL or dataset identifier.
- License information.
- Camera calibration and Tag sizes.
- Expected detector family and Tag IDs.
- SHA-256 checksum.
- Expected localization metrics and tolerances.

Large datasets remain external. The default test suite currently uses synthetic
detections and does not claim end-to-end detector coverage.
