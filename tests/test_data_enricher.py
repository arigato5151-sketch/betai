from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_enricher import FootballAPIConfigurationError, FootballDataEnricher

MOCK_RESPONSE = {
    "response": [
        {
            "team": {"id": 10},
            "statistics": [
                {"type": "Total Shots", "value": 15},
                {"type": "Shots on Goal", "value": 6},
                {"type": "Corner Kicks", "value": 7},
                {"type": "Fouls", "value": 9},
            ],
        },
        {
            "team": {"id": 20},
            "statistics": [
                {"type": "Total Shots", "value": 8},
                {"type": "Shots on Goal", "value": 2},
                {"type": "Corner Kicks", "value": 3},
                {"type": "Fouls", "value": 13},
            ],
        },
    ]
}


@patch("data_enricher.requests.get")
def test_enriches_only_missing_statistics(
    mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "test-key")
    response = MagicMock()
    response.json.return_value = MOCK_RESPONSE
    mock_get.return_value = response
    source = pd.DataFrame(
        [
            {
                "fixture_id": 123,
                "HS": np.nan,
                "AS": np.nan,
                "HST": np.nan,
                "AST": np.nan,
                "HC": np.nan,
                "AC": np.nan,
                "HF": np.nan,
                "AF": np.nan,
            },
            {
                "fixture_id": 456,
                "HS": 10,
                "AS": 6,
                "HST": 4,
                "AST": 1,
                "HC": 5,
                "AC": 2,
                "HF": 8,
                "AF": 11,
            },
        ]
    )

    enriched = FootballDataEnricher(
        min_request_interval_seconds=0
    ).enrich_missing_statistics(source)

    assert enriched.loc[
        0, ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF"]
    ].tolist() == [15, 8, 6, 2, 7, 3, 9, 13]
    assert enriched.loc[1, "HS"] == 10
    mock_get.assert_called_once_with(
        "https://v3.football.api-sports.io/fixtures/statistics",
        headers={"x-apisports-key": "test-key"},
        params={"fixture": 123},
        timeout=20.0,
    )
    response.raise_for_status.assert_called_once()


def test_does_not_call_api_for_complete_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    source = pd.DataFrame([{"fixture_id": 123, "HS": 10, "HC": 5}])

    assert FootballDataEnricher().enrich_missing_statistics(source).equals(source)


@patch("data_enricher.requests.get")
def test_rate_limit_waits_between_requests(
    mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOOTBALL_API_KEY", "test-key")
    response = MagicMock()
    response.json.return_value = MOCK_RESPONSE
    mock_get.return_value = response
    sleeps: list[float] = []
    clock_values = iter([10.0, 10.2, 10.2, 11.2])
    enricher = FootballDataEnricher(
        min_request_interval_seconds=1.0,
        clock=lambda: next(clock_values),
        sleep=sleeps.append,
    )
    source = pd.DataFrame(
        [
            {"fixture_id": 1, "HS": np.nan, "HC": np.nan},
            {"fixture_id": 2, "HS": np.nan, "HC": np.nan},
        ]
    )

    enricher.enrich_missing_statistics(source)

    assert sleeps == [pytest.approx(0.8)]
    assert mock_get.call_count == 2


def test_missing_api_key_fails_only_when_enrichment_is_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    source = pd.DataFrame([{"fixture_id": 123, "HS": np.nan, "HC": 5}])

    with pytest.raises(FootballAPIConfigurationError, match="FOOTBALL_API_KEY"):
        FootballDataEnricher().enrich_missing_statistics(source)
