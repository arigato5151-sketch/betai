"""Verify Football-Data rows resolve to fixture IDs before statistics enrichment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_enricher import FootballDataEnricher
from fixture_resolver import FixtureResolver

FIXTURES_PAYLOAD = {
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

STATISTICS_PAYLOAD = {
    "response": [
        {
            "team": {"id": 1},
            "statistics": [
                {"type": "Total Shots", "value": 18},
                {"type": "Shots on Goal", "value": 7},
                {"type": "Corner Kicks", "value": 6},
                {"type": "Fouls", "value": 9},
            ],
        },
        {
            "team": {"id": 2},
            "statistics": [
                {"type": "Total Shots", "value": 12},
                {"type": "Shots on Goal", "value": 4},
                {"type": "Corner Kicks", "value": 3},
                {"type": "Fouls", "value": 11},
            ],
        },
    ]
}


def _statistics_frame() -> pd.DataFrame:
    columns = ("Date", "HomeTeam", "AwayTeam") + FootballDataEnricher._TARGET_COLUMNS
    payload = {
        "Date": ["16/08/2025"],
        "HomeTeam": ["Man City"],
        "AwayTeam": ["Inter"],
    }
    payload.update(
        {column: [float("nan")] for column in FootballDataEnricher._TARGET_COLUMNS}
    )
    return pd.DataFrame(payload, columns=columns)


@pytest.mark.parametrize("resolver_prime", [False, True])
@patch("requests.get")
def test_resolve_then_enrich_fills_statistics(
    mock_get: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    resolver_prime: bool,
) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "test-key")

    fixture_response = MagicMock()
    fixture_response.json.return_value = FIXTURES_PAYLOAD
    statistics_response = MagicMock()
    statistics_response.json.return_value = STATISTICS_PAYLOAD

    def respond(url: str, **_kwargs):
        return statistics_response if "statistics" in url else fixture_response

    mock_get.side_effect = respond

    resolver = FixtureResolver(
        cache_path=tmp_path / "fixture_cache.json", date_tolerance_days=0
    )
    resolved = resolver.resolve_dataframe(_statistics_frame())
    if resolver_prime:
        # Re-running from a fresh resolver must hit the persisted cache only.
        mock_get.reset_mock()
        resolver = FixtureResolver(
            cache_path=tmp_path / "fixture_cache.json", date_tolerance_days=0
        )
        resolver.resolve_dataframe(resolved)

    enriched = FootballDataEnricher().enrich_missing_statistics(resolved)

    assert int(enriched["fixture_id"].iloc[0]) == 9876
    fixture_calls = [
        call for call in mock_get.call_args_list if "statistics" not in call.args[0]
    ]
    statistics_calls = [
        call for call in mock_get.call_args_list if "statistics" in call.args[0]
    ]
    assert {call.args[0] for call in statistics_calls} == {
        "https://v3.football.api-sports.io/fixtures/statistics"
    }
    assert len(statistics_calls) == 1
    assert len(fixture_calls) == (0 if resolver_prime else 1)
    assert enriched.loc[0, "HS"] == 18
    assert enriched.loc[0, "AS"] == 12
    assert enriched.loc[0, "HST"] == 7
    assert enriched.loc[0, "AST"] == 4
    assert enriched.loc[0, "HC"] == 6
    assert enriched.loc[0, "AC"] == 3
    assert enriched.loc[0, "HF"] == 9
    assert enriched.loc[0, "AF"] == 11
