from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.prediction.ml.model_router import TieredModelArtifactStore
from app.prediction.ml.train_tiered_models import (
    ModelPromotionRejected,
    _champion_benchmark_metrics,
    _market_benchmark_metrics,
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

    result = train_tiered_models(
        fixtures,
        artifact_store=store,
        backend="sklearn",
        require_market_superiority=False,
    )

    assert store.active_path.is_file()
    assert store.verify(store.active_path) is True
    assert store.load_active() is not None
    assert result["tier1_metrics"]["samples"] >= 3
    assert isinstance(result["tier1_metrics"]["beats_opening_market"], bool)
    assert "market_log_loss" in result["tier1_metrics"]
    assert "market_brier_score" in result["tier1_metrics"]
    assert "log_loss_improvement_vs_market" in result["tier1_metrics"]
    assert result["tier2_metrics"]["samples"] >= 3
    assert result["metadata"]["training_source"] == "historical_fixtures"
    assert result["metadata"]["tier1_features"] == list(
        store.load_active().tier1_model.FEATURES
    )
    assert all(
        not name.startswith("closing_") for name in result["metadata"]["tier1_features"]
    )


def test_pipeline_source_training_smoke_exports_signed_artifact(tmp_path) -> None:
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)

    result = train_tiered_models(
        seasons=["2425"],
        leagues=["Premier_League", "La_Liga"],
        pipeline_fetcher=FakeFootballDataFetcher(),
        enrich_odds=_alternating_odds_provider(),
        artifact_store=store,
        backend="sklearn",
        require_market_superiority=False,
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


def test_candidate_that_does_not_beat_market_is_not_promoted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = [
        *(_fixture(index, rich=True) for index in range(18)),
        *(_fixture(index + 40, rich=False) for index in range(18)),
    ]
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)
    monkeypatch.setattr(
        "app.prediction.ml.train_tiered_models._market_benchmark_metrics",
        lambda *_args, **_kwargs: {
            "market_accuracy": 0.6,
            "market_log_loss": 0.9,
            "market_brier_score": 0.55,
            "log_loss_improvement_vs_market": -0.1,
            "brier_improvement_vs_market": -0.05,
            "beats_opening_market": False,
        },
    )

    with pytest.raises(ModelPromotionRejected) as error:
        train_tiered_models(fixtures, artifact_store=store, backend="sklearn")

    assert error.value.tier1_metrics["beats_opening_market"] is False
    assert store.active_path.exists() is False


def test_market_gate_requires_sample_size_and_positive_bootstrap_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_count = 60
    actual = np.arange(sample_count, dtype=int) % 3
    features = pd.DataFrame(
        {
            "opening_away_odd": np.full(sample_count, 3.0),
            "opening_draw_odd": np.full(sample_count, 3.0),
            "opening_home_odd": np.full(sample_count, 3.0),
        }
    )
    strong = np.full((sample_count, 3), 0.05)
    strong[np.arange(sample_count), actual] = 0.90
    monkeypatch.setattr(
        "app.prediction.ml.train_tiered_models.settings.TIERED_PROMOTION_MIN_HOLDOUT_SAMPLES",
        30,
    )

    metrics = _market_benchmark_metrics(
        features,
        pd.Series(actual),
        model_probabilities=strong,
    )

    assert metrics["promotion_sample_sufficient"] is True
    assert metrics["market_log_loss_improvement_lower_bound"] > 0
    assert metrics["market_brier_improvement_lower_bound"] > 0
    assert metrics["beats_opening_market"] is True

    too_small = _market_benchmark_metrics(
        features.iloc[:12],
        pd.Series(actual[:12]),
        model_probabilities=strong[:12],
    )
    assert too_small["promotion_sample_sufficient"] is False
    assert too_small["beats_opening_market"] is False


def test_champion_gate_rejects_noise_and_accepts_material_improvement() -> None:
    noisy_candidate = {"log_loss": 0.9995, "brier_score": 0.5998}
    champion = {"log_loss": 1.0, "brier_score": 0.6}

    rejected = _champion_benchmark_metrics(noisy_candidate, champion)
    accepted = _champion_benchmark_metrics(
        {"log_loss": 0.99, "brier_score": 0.595}, champion
    )

    assert rejected["beats_active_champion"] is False
    assert accepted["beats_active_champion"] is True


def test_equal_candidate_does_not_replace_signed_active_champion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = [
        *(_fixture(index, rich=True) for index in range(18)),
        *(_fixture(index + 40, rich=False) for index in range(18)),
    ]
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)
    first = train_tiered_models(
        fixtures,
        artifact_store=store,
        backend="sklearn",
        require_market_superiority=False,
    )
    active_version = first["artifact_version"]
    monkeypatch.setattr(
        "app.prediction.ml.train_tiered_models._market_benchmark_metrics",
        lambda *_args, **_kwargs: {"beats_opening_market": True},
    )

    with pytest.raises(ModelPromotionRejected) as error:
        train_tiered_models(fixtures, artifact_store=store, backend="sklearn")

    assert "active_champion" in error.value.tier1_metrics["promotion_failures"]
    assert store.load_active().artifact_version == active_version


def test_market_benchmark_prefers_closing_line_when_available() -> None:
    sample_count = 60
    actual = np.arange(sample_count, dtype=int) % 3
    features = pd.DataFrame(
        {
            "opening_home_odd": np.full(sample_count, 2.0),
            "opening_draw_odd": np.full(sample_count, 3.0),
            "opening_away_odd": np.full(sample_count, 4.0),
            "closing_home_odd": np.full(sample_count, 3.0),
            "closing_draw_odd": np.full(sample_count, 3.0),
            "closing_away_odd": np.full(sample_count, 3.0),
        }
    )
    strong = np.full((sample_count, 3), 0.05)
    strong[np.arange(sample_count), actual] = 0.90

    metrics = _market_benchmark_metrics(
        features, pd.Series(actual), model_probabilities=strong
    )

    # Closing 3.0/3.0/3.0 is an even market; only the opening line differs.
    assert metrics["market_line"] == "closing"
    assert metrics["market_accuracy"] == pytest.approx(1 / 3)

    opening_only = _market_benchmark_metrics(
        features.drop(
            columns=["closing_home_odd", "closing_draw_odd", "closing_away_odd"]
        ),
        pd.Series(actual),
        model_probabilities=strong,
    )
    assert opening_only["market_line"] == "opening"


def test_opening_only_benchmark_blocks_promotion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = [
        *(_fixture(index, rich=True) for index in range(18)),
        *(_fixture(index + 40, rich=False) for index in range(18)),
    ]
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)
    monkeypatch.setattr(
        "app.prediction.ml.train_tiered_models._market_benchmark_metrics",
        lambda *_args, **_kwargs: {
            "beats_opening_market": True,
            "market_line": "opening",
        },
    )

    with pytest.raises(ModelPromotionRejected) as error:
        train_tiered_models(fixtures, artifact_store=store, backend="sklearn")

    assert "opening_only_benchmark" in error.value.tier1_metrics["promotion_failures"]


def test_tier2_gate_requires_samples_and_class_balance(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "TIERED_MIN_TIER2_SAMPLES", 500)
    monkeypatch.setattr(settings, "TIERED_MIN_TIER2_SAMPLES_PER_CLASS", 20)
    target = pd.Series(["HOME_WIN"] * 10 + ["AWAY_WIN"] * 3 + ["DRAW"] * 1)

    from app.prediction.ml.train_tiered_models import _tier2_gate

    gate = _tier2_gate(target)

    assert gate["passed"] is False
    assert gate["reasons"] == ["insufficient_tier2_samples", "tier2_class_imbalance"]
    assert gate["training_samples"] == 14
    assert gate["min_per_class"] == 1

    monkeypatch.setattr(settings, "TIERED_MIN_TIER2_SAMPLES", 10)
    monkeypatch.setattr(settings, "TIERED_MIN_TIER2_SAMPLES_PER_CLASS", 1)
    healthy = _tier2_gate(target)
    assert healthy["passed"] is True
    assert healthy["reasons"] == []
