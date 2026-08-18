"""Live AprilTag mapping with a camera view and a 3D world view.

Start with Tag 0 visible. The first connected observation defines the world
frame, and Tags 1, 2, and 3 are discovered as the camera moves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib


def configure_interactive_backend() -> None:
    """Select a GUI backend before importing pyplot."""

    backend = str(matplotlib.get_backend()).lower()
    if backend not in {"agg", "template"}:
        return
    try:
        matplotlib.use("TkAgg", force=True)
    except ImportError as exc:
        raise RuntimeError(
            "The live 3D example needs an interactive Matplotlib backend. "
            "On Ubuntu/Debian install it with `sudo apt install python3-tk`, "
            "then recreate the uv environment with `uv sync`."
        ) from exc


import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from tagatlas import (  # noqa: E402
    CameraModel,
    Detection,
    Localizer,
    LocalizerConfig,
    TagMap,
)
from tagatlas.apriltag_detector import PupilAprilTagDetector  # noqa: E402

configure_interactive_backend()
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
TAG_IDS = (0, 1, 2, 3)
TAG_SIZE = 0.10


class RecordingDetector:
    """Keep detector output so the example can draw image-space corners."""

    def __init__(self, family: str) -> None:
        self._detector = PupilAprilTagDetector(families=family)
        self.last_detections: tuple[Detection, ...] = ()

    def detect(self, image: NDArray[Any]) -> tuple[Detection, ...]:
        self.last_detections = tuple(self._detector.detect(image))
        return self.last_detections


class LivePlot:
    """Matplotlib view combining the camera image and a live 3D map."""

    def __init__(self, width: int, height: int) -> None:
        self.figure = plt.figure(figsize=(15, 7), constrained_layout=True)
        grid = self.figure.add_gridspec(1, 2, width_ratios=(1, 1.05))
        self.image_axes: Axes = self.figure.add_subplot(grid[0, 0])
        self.world_axes = self.figure.add_subplot(grid[0, 1], projection="3d")
        self.image_artist = self.image_axes.imshow(
            np.zeros((height, width, 3), dtype=np.uint8)
        )
        self.image_axes.set_title("Camera view")
        self.image_axes.axis("off")
        self.figure.canvas.manager.set_window_title("TagAtlas live mapping")
        self.controls = {"quit": False, "reset": False, "save": False}
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.world_axes.set_title("World coordinate system")
        self.world_axes.set_xlabel("World X (m)")
        self.world_axes.set_ylabel("World Y (m)")
        self.world_axes.set_zlabel("World Z (m)")
        self.world_axes.view_init(elev=25, azim=-60)
        self._origin: Any = None
        self._tag_polygons: dict[int, Any] = {}
        self._tag_labels: dict[int, Any] = {}
        self._trajectory_line: Any = None
        self._camera_artists: list[Any] = []

    def _on_key(self, event: Any) -> None:
        if event.key in {"q", "escape"}:
            self.controls["quit"] = True
        elif event.key == "r":
            self.controls["reset"] = True
        elif event.key == "s":
            self.controls["save"] = True

    def consume(self, name: str) -> bool:
        value = self.controls[name]
        self.controls[name] = False
        return value

    def update_image(
        self,
        frame: NDArray[np.uint8],
        detections: tuple[Detection, ...],
        used_ids: tuple[int, ...],
        result: Any,
    ) -> None:
        image = draw_detections(frame, detections, used_ids)
        status = "LOCALIZED" if result.success else f"FEEDBACK: {result.reason}"
        diagnostics = result.diagnostics
        lines = [
            status,
            f"detected={diagnostics.detected_count}  "
            f"accepted={diagnostics.accepted_count}",
            f"mapped={len(result.tag_poses)}  used={result.used_tag_ids}",
        ]
        if result.reprojection_rmse is not None:
            lines.append(f"RMSE={result.reprojection_rmse:.2f}px")
        self.image_artist.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.image_axes.set_title("Camera view | " + " | ".join(lines))

    def update_world(
        self,
        tag_poses: dict[int, Any],
        trajectory: list[NDArray[np.float64]],
        camera_pose: Any | None,
    ) -> None:
        axes = self.world_axes
        if self._origin is None:
            origin = axes.plot([0.0], [0.0], [0.0], marker="+", c="black", ms=10)
            self._origin = origin[0]

        geometry: list[NDArray[np.float64]] = []
        for pose in tag_poses.values():
            geometry.append(pose.translation)
        geometry.extend(trajectory)
        if camera_pose is not None:
            geometry.append(camera_pose.translation)
        points = np.asarray(geometry or [np.zeros(3)], dtype=np.float64)
        span = np.maximum(np.ptp(points, axis=0), 0.25)
        center = np.mean(points, axis=0)
        axes.set_xlim(center[0] - span[0], center[0] + span[0])
        axes.set_ylim(center[1] - span[1], center[1] + span[1])
        axes.set_zlim(center[2] - span[2], center[2] + span[2])
        axes.set_box_aspect((span[0], span[1], span[2]))

        for tag_id in set(self._tag_polygons) - set(tag_poses):
            self._tag_polygons.pop(tag_id).remove()
            self._tag_labels.pop(tag_id).remove()
        for tag_id, pose in sorted(tag_poses.items()):
            if tag_id not in self._tag_polygons:
                color = "#ffb000" if tag_id == 0 else "#36c275"
                polygon = Poly3DCollection(
                    [np.zeros((4, 3))], alpha=0.85, facecolor=color
                )
                polygon.set_edgecolor("white")
                polygon.set_linewidth(1.5)
                axes.add_collection3d(polygon)
                self._tag_polygons[tag_id] = polygon
                self._tag_labels[tag_id] = axes.text(
                    0.0,
                    0.0,
                    0.0,
                    f"Tag {tag_id}",
                    color="black",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
                )
            self._tag_polygons[tag_id].set_verts([tag_corners(pose, TAG_SIZE)])
            position = pose.translation
            self._tag_labels[tag_id].set_position(
                (position[0], position[1], position[2] + TAG_SIZE * 0.35)
            )

        if trajectory:
            path = np.asarray(trajectory)
            if self._trajectory_line is None:
                self._trajectory_line = axes.plot([], [], [], color="#357edd", lw=2)[0]
            self._trajectory_line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
            self._trajectory_line.set_visible(True)
        elif self._trajectory_line is not None:
            self._trajectory_line.set_visible(False)

        if camera_pose is not None:
            scale = max(float(np.max(span)) * 0.12, 0.08)
            self._update_camera(axes, camera_pose, scale)
        else:
            for artist in self._camera_artists:
                artist.set_visible(False)
        self.figure.canvas.draw_idle()

    def _update_camera(self, axes: Axes, pose: Any, scale: float) -> None:
        """Reuse camera line artists and update their endpoints each frame."""

        center = pose.translation
        rotation = pose.rotation
        depth = scale * 2.0
        half_width = scale * 0.8
        half_height = scale * 0.6
        image_plane = np.array(
            [
                [-half_width, -half_height, depth],
                [half_width, -half_height, depth],
                [half_width, half_height, depth],
                [-half_width, half_height, depth],
            ],
            dtype=np.float64,
        )
        plane_world = image_plane @ rotation.T + center
        if not self._camera_artists:
            axis_colors = ["#e63946", "#2a9d8f", "#457b9d"]
            for color in axis_colors:
                self._camera_artists.append(axes.plot([], [], [], color=color, lw=2)[0])
            for _index in range(4):
                self._camera_artists.append(
                    axes.plot([], [], [], color="#e63946", lw=0.8)[0]
                )
            self._camera_artists.append(
                axes.plot([], [], [], color="#e63946", lw=1.2)[0]
            )
            self._camera_artists.append(
                axes.plot([], [], [], marker="o", color="#e63946", ms=5)[0]
            )
        artists = iter(self._camera_artists)
        for index in range(3):
            endpoint = center + rotation[:, index] * scale
            line = next(artists)
            line.set_data_3d(
                [center[0], endpoint[0]],
                [center[1], endpoint[1]],
                [center[2], endpoint[2]],
            )
            line.set_visible(True)
        for corner in plane_world:
            line = next(artists)
            line.set_data_3d(
                [center[0], corner[0]],
                [center[1], corner[1]],
                [center[2], corner[2]],
            )
            line.set_visible(True)
        outline = next(artists)
        closed = np.vstack([plane_world, plane_world[0]])
        outline.set_data_3d(closed[:, 0], closed[:, 1], closed[:, 2])
        outline.set_visible(True)
        center_marker = next(artists)
        center_marker.set_data([center[0]], [center[1]])
        center_marker.set_3d_properties([center[2]])
        center_marker.set_visible(True)

    def show(self) -> None:
        plt.pause(0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--output-map", type=Path, default=Path("tag-map.json"))
    return parser.parse_args()


def make_camera(args: argparse.Namespace) -> CameraModel:
    cx = args.width / 2.0 if args.cx is None else args.cx
    cy = args.height / 2.0 if args.cy is None else args.cy
    return CameraModel(
        matrix=np.array(
            [[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        distortion_coefficients=np.empty(0, dtype=np.float64),
    )


def draw_detections(
    frame: NDArray[np.uint8],
    detections: tuple[Detection, ...],
    used_ids: tuple[int, ...],
) -> NDArray[np.uint8]:
    result = frame.copy()
    used = set(used_ids)
    for detection in detections:
        corners = np.round(detection.corners).astype(np.int32)
        color = (0, 220, 0) if detection.tag_id in used else (0, 165, 255)
        cv2.polylines(result, [corners], True, color, 2, cv2.LINE_AA)
        center = tuple(np.mean(corners, axis=0).astype(int))
        cv2.putText(
            result,
            f"Tag {detection.tag_id}",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def tag_corners(pose: Any, size: float) -> NDArray[np.float64]:
    local_corners = np.array(
        [
            [-size / 2, -size / 2, 0.0],
            [size / 2, -size / 2, 0.0],
            [size / 2, size / 2, 0.0],
            [-size / 2, size / 2, 0.0],
        ],
        dtype=np.float64,
    )
    return local_corners @ pose.rotation.T + pose.translation


def main() -> None:
    args = parse_args()
    camera = make_camera(args)
    tag_sizes = {tag_id: TAG_SIZE for tag_id in TAG_IDS}
    tag_map = TagMap(reference_tag_id=0, tag_sizes=tag_sizes)
    detector = RecordingDetector("tag36h11")
    localizer = Localizer(
        camera=camera,
        tag_map=tag_map,
        config=LocalizerConfig(map_mode="discover"),
        detector=detector,
    )
    capture = cv2.VideoCapture(args.camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError(f"unable to open camera {args.camera_index}")

    plot = LivePlot(args.width, args.height)
    trajectory: list[NDArray[np.float64]] = []
    last_result: Any | None = None
    try:
        while not plot.controls["quit"]:
            ok, frame = capture.read()
            if not ok:
                print("camera frame could not be read")
                break
            result = localizer.locate(frame)
            last_result = result
            if result.success and result.camera_pose is not None:
                trajectory.append(result.camera_pose.translation.copy())
            if plot.consume("reset"):
                localizer.reset()
                trajectory.clear()
            if plot.consume("save"):
                saved_map = TagMap(
                    reference_tag_id=0,
                    tag_sizes=tag_sizes,
                    tag_poses=dict(result.tag_poses),
                )
                saved_map.to_json(args.output_map)
                print(f"saved discovered map to {args.output_map}")
            plot.update_image(
                frame, detector.last_detections, result.used_tag_ids, result
            )
            plot.update_world(dict(result.tag_poses), trajectory, result.camera_pose)
            plot.show()
            if last_result is not None and not plt.fignum_exists(plot.figure.number):
                break
    finally:
        capture.release()
        plt.close(plot.figure)


if __name__ == "__main__":
    main()
