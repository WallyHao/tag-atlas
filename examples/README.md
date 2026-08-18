# Examples

## Live Mapping

`live_mapping.py` demonstrates online map discovery with four AprilTags. The
relative positions of Tags 1, 2, and 3 are intentionally unknown. Tag 0 must be
visible in the first useful frame and defines the world-frame origin.

The example assumes:

- Tag IDs are `0`, `1`, `2`, and `3`;
- every Tag has an edge length of `0.10 m`;
- the detector family is `tag36h11`;
- the camera has no lens distortion;
- the default camera resolution is `640x480`;
- the default pinhole intrinsics are `fx=fy=500`, `cx=320`, and `cy=240`.

Replace the default intrinsics with the actual calibrated values before using
the pose numerically. The example does not require camera extrinsics: they are
estimated from the observed Tag map.

Run it with:

```bash
uv run python examples/live_mapping.py --camera-index 0
```

The example requires a GUI-capable Matplotlib backend. On Ubuntu/Debian, install
Tk support if Matplotlib reports that it is using the non-interactive `Agg`
backend:

```bash
sudo apt install python3-tk
uv sync
```

Optional calibration arguments:

```bash
uv run python examples/live_mapping.py \
  --camera-index 0 \
  --width 1280 \
  --height 720 \
  --fx 900 \
  --fy 900 \
  --cx 640 \
  --cy 360
```

The Matplotlib window contains two synchronized views:

- left: the camera image, detected Tag corners, IDs, localization status, and
  reprojection RMSE;
- right: a 3D world-coordinate scene with dynamically discovered Tag planes,
  numbered labels, camera axes, viewing frustum, position, and trajectory.

The 3D axes use the Tag 0 world frame. The camera red, green, and blue arrows
show its local X, Y, and Z axes; the red frustum shows the viewing direction.

Controls:

- `q` or `Esc`: quit;
- `r`: reset the trajectory and discovered map;
- `s`: save the current discovered map as `tag-map.json`.

If no Tag is visible, the example displays feedback in the window and keeps
running. If Tag 0 is not visible at startup, move the camera until Tag 0 is
detected before expecting a world-frame pose.
