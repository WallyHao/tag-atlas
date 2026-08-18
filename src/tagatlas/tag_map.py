"""Static AprilTag map configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .models import Pose, _float_value, _parse_tag_id


@dataclass(frozen=True)
class TagMap:
    """Validated static Tag sizes and the world-frame reference Tag."""

    reference_tag_id: int
    tag_sizes: Mapping[int, float]
    tag_poses: Mapping[int, Pose] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        reference_tag_id = _parse_tag_id(self.reference_tag_id, "reference_tag_id")
        sizes: dict[int, float] = {}
        for tag_id, size in self.tag_sizes.items():
            normalized_id = _parse_tag_id(tag_id, "tag ID")
            normalized_size = _float_value(size, "tag size")
            if not np.isfinite(normalized_size) or normalized_size <= 0.0:
                raise ValueError("all tag sizes must be positive meters and finite")
            if normalized_id in sizes:
                raise ValueError(f"duplicate tag id: {normalized_id}")
            sizes[normalized_id] = normalized_size
        if reference_tag_id not in sizes:
            raise ValueError("reference_tag_id must be present in tag_sizes")
        raw_poses = self.tag_poses
        poses: dict[int, Pose] = {}
        for tag_id, pose in raw_poses.items():
            normalized_id = _parse_tag_id(tag_id, "tag pose ID")
            if normalized_id not in sizes:
                raise ValueError("tag poses must reference configured tag IDs")
            if normalized_id in poses:
                raise ValueError(f"duplicate tag pose id: {normalized_id}")
            if not isinstance(pose, Pose):
                raise ValueError("tag poses must contain Pose values")
            poses[normalized_id] = pose
        if reference_tag_id in poses and not np.allclose(
            poses[reference_tag_id].matrix(), np.eye(4), atol=1e-6
        ):
            raise ValueError("reference tag pose must be identity")
        object.__setattr__(self, "reference_tag_id", reference_tag_id)
        object.__setattr__(self, "tag_sizes", MappingProxyType(sizes))
        object.__setattr__(self, "tag_poses", MappingProxyType(poses))

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> TagMap:
        """Create a TagMap from a YAML/JSON-like mapping."""

        raw_sizes = config.get("tag_sizes")
        if not isinstance(raw_sizes, Mapping):
            raise ValueError("tag_sizes must be a mapping")
        allowed = {"reference_tag_id", "tag_sizes"}
        allowed.add("tag_poses")
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown TagMap fields: {sorted(unknown)}")
        reference = _parse_tag_id(config.get("reference_tag_id", 0), "reference_tag_id")
        sizes = {
            _parse_tag_id(key, "tag ID"): _float_value(value, "tag size")
            for key, value in raw_sizes.items()
        }
        raw_poses = config.get("tag_poses", {})
        if not isinstance(raw_poses, Mapping):
            raise ValueError("tag_poses must be a mapping")
        poses: dict[int, Pose] = {}
        for key, value in raw_poses.items():
            tag_id = _parse_tag_id(key, "tag pose ID")
            if tag_id in poses:
                raise ValueError(f"duplicate tag pose id: {tag_id}")
            poses[tag_id] = _pose_from_value(value)
        return cls(
            reference_tag_id=reference,
            tag_sizes=sizes,
            tag_poses=poses,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-compatible mapping representation of the map."""

        return {
            "schema_version": 1,
            "reference_tag_id": self.reference_tag_id,
            "tag_sizes": dict(self.tag_sizes),
            "tag_poses": {
                tag_id: pose.matrix().tolist()
                for tag_id, pose in self.tag_poses.items()
            },
        }

    def to_json(self, path: str | Path) -> None:
        """Write the map to a UTF-8 JSON file."""

        Path(path).write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str | Path) -> TagMap:
        """Load a map from a JSON file written by :meth:`to_json`."""

        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read TagMap JSON: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError("TagMap JSON root must be an object")
        schema_version = value.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError(f"unsupported TagMap schema version: {schema_version}")
        config = dict(value)
        config.pop("schema_version", None)
        return cls.from_mapping(config)


def _pose_from_value(value: object) -> Pose:
    """Parse a homogeneous matrix from a JSON-like mapping value."""

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("tag poses must contain 4x4 matrices") from exc
    if matrix.shape != (4, 4):
        raise ValueError("tag poses must contain 4x4 matrices")
    return Pose(matrix[:3, :3], matrix[:3, 3])


def _invalid_pose() -> Pose:
    raise ValueError("tag poses must contain 4x4 matrices")
