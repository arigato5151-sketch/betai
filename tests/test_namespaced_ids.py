from __future__ import annotations

import pytest

from app.core.namespaced_ids import (
    FIXTURE_DOWNLOAD_TEAM_OFFSET,
    FOOTBALL_DATA_FIXTURE_OFFSET,
    FOOTBALL_DATA_TEAM_OFFSET,
    MAX_PROVIDER_ID,
    OPENFOOTBALL_FIXTURE_OFFSET,
    OPENFOOTBALL_TEAM_OFFSET,
    SOURCE_ID_OFFSETS,
    STATSBOMB_FIXTURE_OFFSET,
    STATSBOMB_PLAYER_OFFSET,
    STATSBOMB_TEAM_OFFSET,
    hashed_id,
    namespaced_id,
)

_HASH_OFFSETS = {
    FOOTBALL_DATA_FIXTURE_OFFSET,
    FOOTBALL_DATA_TEAM_OFFSET,
    OPENFOOTBALL_FIXTURE_OFFSET,
    OPENFOOTBALL_TEAM_OFFSET,
    FIXTURE_DOWNLOAD_TEAM_OFFSET,
    STATSBOMB_FIXTURE_OFFSET,
    STATSBOMB_TEAM_OFFSET,
    STATSBOMB_PLAYER_OFFSET,
}


def test_namespaced_id_maps_raw_provider_id_into_positive_range() -> None:
    offset = SOURCE_ID_OFFSETS["fixture_download"]
    assert namespaced_id("fixture_download", 12345) == offset + 12345
    assert offset + 12345 > 0


@pytest.mark.parametrize(
    "value", [0, -5, True, 1_000_000_000, "not-an-int", None]
)
def test_namespaced_id_rejects_invalid_raw_ids(value: object) -> None:
    assert namespaced_id("thesportsdb", value) is None


def test_hashed_id_is_deterministic_and_bounded() -> None:
    natural_key = "203:2025:2025-08-08T18:30:00+00:00:england-man-utd:england-liverpool"
    first = hashed_id("football-data-fixture", natural_key, FOOTBALL_DATA_FIXTURE_OFFSET)
    second = hashed_id("football-data-fixture", natural_key, FOOTBALL_DATA_FIXTURE_OFFSET)
    assert first == second
    assert FOOTBALL_DATA_FIXTURE_OFFSET < first <= FOOTBALL_DATA_FIXTURE_OFFSET + MAX_PROVIDER_ID


def test_hashed_id_changes_with_natural_key() -> None:
    a = hashed_id("football-data-fixture", "match-one", FOOTBALL_DATA_FIXTURE_OFFSET)
    b = hashed_id("football-data-fixture", "match-two", FOOTBALL_DATA_FIXTURE_OFFSET)
    assert a != b


def test_hashed_id_namespace_keeps_sources_disjoint() -> None:
    key = "a-team-name"
    values = {
        hashed_id(f"source-{namespace}", key, offset)
        for namespace, offset in enumerate(_HASH_OFFSETS, start=1)
    }
    assert len(values) == len(_HASH_OFFSETS)
    for value in values:
        assert value > 0


def test_all_offsets_share_colliding_digit_space() -> None:
    """No two offsets may claim the same numeric range for one source id."""
    key = "2025:08-08:team-a:team-b"
    fixture = hashed_id("fixture-download-fixture", key, 1_750_000_000)
    assert fixture == SOURCE_ID_OFFSETS["fixture_download"] + (
        fixture - SOURCE_ID_OFFSETS["fixture_download"]
    )