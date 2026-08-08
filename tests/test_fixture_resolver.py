from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fixture_resolver import FixtureResolver

MOCK_FIXTURES = {
    "response": [
        {
            "fixture": {"id": 9876, "date": "2025-08-16T14:00:00+00:00"},
            "teams": {
                "home": {"name": "Manchester City"},
                "away": {"name": "FC Internazionale"},
            },
        }
    ]
}


def test_parses_football_data_date_variants() -> None:
    assert FixtureResolver.parse_football_data_date("16/08/2025").isoformat() == "2025-08-16"  # type: ignore[union-attr]
    assert FixtureResolver.parse_football_data_date("16/08/25").isoformat() == "2025-08-16"  # type: ignore[union-attr]
    assert FixtureResolver.parse_football_data_date("2025-08-16") is None


@patch("fixture_resolver.requests.get")
def test_resolves_fuzzy_names_and_writes_fixture_id(
    mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "test-key")
    response = MagicMock()
    response.json.return_value = MOCK_FIXTURES
    mock_get.return_value = response
    source = pd.DataFrame(
        [{"Date": "16/08/2025", "HomeTeam": "Man City", "AwayTeam": "Inter"}]
    )

    resolved = FixtureResolver(
        cache_path=tmp_path / "fixture_cache.json", date_tolerance_days=0
    ).resolve_dataframe(source)

    assert resolved.loc[0, "fixture_id"] == 9876
    mock_get.assert_called_once_with(
        "https://v3.football.api-sports.io/fixtures",
        headers={"x-apisports-key": "test-key"},
        params={"date": "2025-08-16"},
        timeout=20.0,
    )


@patch("fixture_resolver.requests.get")
def test_uses_persisted_cache_without_a_network_request(
    mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "test-key")
    response = MagicMock()
    response.json.return_value = MOCK_FIXTURES
    mock_get.return_value = response
    cache_path = tmp_path / "fixture_cache.json"
    resolver = FixtureResolver(cache_path=cache_path, date_tolerance_days=0)
    assert (
        resolver.resolve_fixture_id(
            FixtureResolver.parse_football_data_date("16/08/2025"), home_team="Man City", away_team="Inter"  # type: ignore[arg-type]
        )
        == 9876
    )

    mock_get.reset_mock()
    cached_resolver = FixtureResolver(cache_path=cache_path, date_tolerance_days=0)
    assert (
        cached_resolver.resolve_fixture_id(
            FixtureResolver.parse_football_data_date("16/08/25"), home_team="Man City", away_team="Inter"  # type: ignore[arg-type]
        )
        == 9876
    )
    mock_get.assert_not_called()


@patch("fixture_resolver.requests.get")
def test_tolerance_queries_adjacent_day_and_rejects_weak_match(
    mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "test-key")
    response = MagicMock()
    response.json.return_value = MOCK_FIXTURES
    mock_get.return_value = response
    resolver = FixtureResolver(
        cache_path=tmp_path / "fixture_cache.json",
        date_tolerance_days=1,
        min_request_interval_seconds=0,
    )

    assert (
        resolver.resolve_fixture_id(
            FixtureResolver.parse_football_data_date("17/08/2025"), home_team="Man City", away_team="Inter"  # type: ignore[arg-type]
        )
        == 9876
    )
    assert mock_get.call_count == 3
    assert resolver.name_similarity("Arsenal", "Manchester City") < resolver.threshold
