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
    with pytest.raises(ValueError, match="integer"):
        TagMap.from_mapping({"tag_sizes": {1.5: 0.12}})
    with pytest.raises(ValueError, match="unknown"):
        TagMap.from_mapping({"tag_sizes": {0: 0.12}, "extra": True})


def test_tag_map_round_trips_json(tmp_path) -> None:
    tag_map = TagMap(reference_tag_id=1, tag_sizes={1: 0.12, 4: 0.2})
    path = tmp_path / "map.json"

    tag_map.to_json(path)
    loaded = TagMap.from_json(path)

    assert loaded.reference_tag_id == tag_map.reference_tag_id
    assert loaded.tag_sizes == tag_map.tag_sizes


def test_tag_map_rejects_unknown_json_schema(tmp_path) -> None:
    path = tmp_path / "map.json"
    path.write_text(
        '{"schema_version": 2, "reference_tag_id": 0, "tag_sizes": {"0": 0.1}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema version"):
        TagMap.from_json(path)
