import pytest

from tagatlas import TagMap


def test_tag_map_validates_and_normalizes_mapping() -> None:
    tag_map = TagMap.from_mapping(
        {
            "reference_tag_id": "1",
            "tag_sizes": {"1": "0.12", "2": 0.15},
        }
    )

    assert tag_map.reference_tag_id == 1
    assert tag_map.tag_sizes == {1: 0.12, 2: 0.15}


def test_tag_map_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="tag_sizes"):
        TagMap.from_mapping({})
    with pytest.raises(ValueError, match="reference_tag_id"):
        TagMap(reference_tag_id=0, tag_sizes={1: 0.12})
    with pytest.raises(ValueError, match="positive meters"):
        TagMap(reference_tag_id=0, tag_sizes={0: 0.0})
    with pytest.raises(ValueError, match="finite"):
        TagMap(reference_tag_id=0, tag_sizes={0: float("nan")})
