from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.prediction.ml.model_router import TieredModelArtifactStore
from app.prediction.ml.train_tiered_models import (
    normalize_pipeline_row,
    train_tiered_models,
)

COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
)


class FakeFootballDataFetcher:
    """Deterministic ``data_pipeline``-compatible fetcher (no network)."""

    def get_league_data(self, season: str, league_key: str) -> pd.DataFrame:
        assert season == "2425"
        outcomes = ("H", "D", "A")
        teams = ("Alpha", "Beta", "Gamma", "Delta")
        rows = []
        for index in range(36):
            home = teams[index % len(teams)]
            away = teams[(index + 1) % len(teams)]
            rows.append(
                {
                    "Date": "11/08/2024",
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": index % 3,
                    "FTAG": (index + 1) % 3,
                    "FTR": outcomes[index % 3],
                    "HS": 10 + index % 5,
                    "AS": 8,
                    "HST": 4,
                    "AST": 3,
                    "HC": 5,
                    "AC": 4,
                    "HF": 9,
                    "AF": 11,
                }
            )
        return pd.DataFrame(rows)


def _alternating_odds_provider():
    counter = {"value": 0}

    def provider(fixture: dict[str, object]) -> dict[str, object]:
        counter["value"] += 1
        if counter["value"] % 2 == 0:
            return {}
        return {
            "opening_home_odd": 1.9,
            "opening_draw_odd": 3.4,
            "opening_away_odd": 4.3,
            "closing_home_odd": 1.8,
            "closing_draw_odd": 3.5,
            "closing_away_odd": 4.5,
        }

    return provider


def _fixture(index: int, *, rich: bool) -> dict[str, object]:
    outcomes = ("AWAY_WIN", "DRAW", "HOME_WIN")
    fixture: dict[str, object] = {
        "kickoff": datetime(2024, 8, 1, tzinfo=UTC) + timedelta(days=index),
        "league_id": 39 if rich else 2,
        "home_team": f"Home {index % 4}",
        "away_team": f"Away {index % 4}",
        "home_goals": index % 3,
        "away_goals": (index + 1) % 3,
        "actual_result": outcomes[index % 3],
    }
    if rich:
        fixture.update(
            {
                "home_shots": 10 + index % 5,
                "away_shots": 8,
                "home_shots_on_target": 4,
                "away_shots_on_target": 3,
                "home_corners": 5,
                "away_corners": 4,
                "home_fouls": 9,
                "away_fouls": 11,
                "opening_home_odd": 1.9,
                "opening_draw_odd": 3.4,
                "opening_away_odd": 4.3,
                "closing_home_odd": 1.8,
                "closing_draw_odd": 3.5,
                "closing_away_odd": 4.5,
            }
        )
    return fixture


def test_training_smoke_exports_signed_tiered_artifact(tmp_path) -> None:
    fixtures = [
        *(_fixture(index, rich=True) for index in range(18)),
        *(_fixture(index + 40, rich=False) for index in range(18)),
    ]
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)

    result = train_tiered_models(fixtures, artifact_store=store, backend="sklearn")

    assert store.active_path.is_file()
    assert store.verify(store.active_path) is True
    assert store.load_active() is not None
    assert result["tier1_metrics"]["samples"] >= 3
    assert result["tier2_metrics"]["samples"] >= 3
    assert result["metadata"]["training_source"] == "historical_fixtures"


def test_pipeline_source_training_smoke_exports_signed_artifact(tmp_path) -> None:
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)

    result = train_tiered_models(
        seasons=["2425"],
        leagues=["Premier_League", "La_Liga"],
        pipeline_fetcher=FakeFootballDataFetcher(),
        enrich_odds=_alternating_odds_provider(),
        artifact_store=store,
        backend="sklearn",
    )

    assert store.active_path.is_file()
    assert store.verify(store.active_path) is True
    assert store.load_active() is not None
    assert result["metadata"]["training_source"] == "pipeline"
    assert result["metadata"]["training_backend"] == "sklearn"
    assert result["tier1_metrics"]["samples"] >= 3
    assert result["tier2_metrics"]["samples"] >= 3


def test_normalize_pipeline_row_maps_columns_and_results() -> None:
    row = pd.Series(
        {
            "Date": "11/08/2024",
            "HomeTeam": "Burnley",
            "AwayTeam": "Man City",
            "FTHG": 0,
            "FTAG": 3,
            "FTR": "A",
            "HS": 6,
            "AS": 17,
            "HST": 1,
            "AST": 8,
            "HC": 6,
            "AC": 5,
            "HF": 11,
            "AF": 8,
        }
    )

    fixture = normalize_pipeline_row(row, league_id=39)

    assert fixture["league_id"] == 39
    assert fixture["home_team"] == "Burnley"
    assert fixture["away_team"] == "Man City"
    assert fixture["home_goals"] == 0
    assert fixture["away_goals"] == 3
    assert fixture["actual_result"] == "AWAY_WIN"
    assert fixture["home_shots"] == 6
    assert fixture["away_fouls"] == 8


def test_training_rejects_insufficient_tier_data(tmp_path) -> None:
    fixtures = [_fixture(index, rich=True) for index in range(3)]

    with pytest.raises(ValueError, match="required per tier"):
        train_tiered_models(
            fixtures,
            artifact_store=TieredModelArtifactStore(artifacts_dir=tmp_path),
            backend="sklearn",
        )
