"""Static AprilTag map configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .models import _float_value, _parse_tag_id


@dataclass(frozen=True)
class TagMap:
    """Validated static Tag sizes and the world-frame reference Tag."""

    reference_tag_id: int
    tag_sizes: Mapping[int, float]

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
        object.__setattr__(self, "reference_tag_id", reference_tag_id)
        object.__setattr__(self, "tag_sizes", MappingProxyType(sizes))

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> TagMap:
        """Create a TagMap from a YAML/JSON-like mapping."""

        raw_sizes = config.get("tag_sizes")
        if not isinstance(raw_sizes, Mapping):
            raise ValueError("tag_sizes must be a mapping")
        allowed = {"reference_tag_id", "tag_sizes"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown TagMap fields: {sorted(unknown)}")
        reference = _parse_tag_id(config.get("reference_tag_id", 0), "reference_tag_id")
        sizes = {
            _parse_tag_id(key, "tag ID"): _float_value(value, "tag size")
            for key, value in raw_sizes.items()
        }
        return cls(
            reference_tag_id=reference,
            tag_sizes=sizes,
        )
