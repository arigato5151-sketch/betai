import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.prediction.ensemble_weights import EnsembleWeightManager

OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


def labeled_rows(count: int, *, league_id: int | None = None) -> list[SimpleNamespace]:
    rows = []
    for index in range(count):
        actual_index = index % len(OUTCOMES)
        wrong_index = (actual_index + 1) % len(OUTCOMES)

        def probabilities(actual: float, wrong: float) -> dict[str, float]:
            values = [round((100.0 - actual - wrong), 4)] * len(OUTCOMES)
            values[actual_index] = actual
            values[wrong_index] = wrong
            remaining_index = 3 - actual_index - wrong_index
            values[remaining_index] = round(100.0 - actual - wrong, 4)
            return dict(zip(OUTCOMES, values, strict=True))

        rows.append(
            SimpleNamespace(
                id=index + 1,
                created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
                league_id=league_id,
                actual_result=OUTCOMES[actual_index],
                probability_components={
                    "components": {
                        "stats": probabilities(10.0, 80.0),
                        "ml": probabilities(50.0, 25.0),
                        "market": probabilities(80.0, 10.0),
                    }
                },
            )
        )
    return rows


@pytest.fixture
def calibration_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> EnsembleWeightManager:
    monkeypatch.setattr(
        settings, "ENSEMBLE_WEIGHTS_PATH", str(tmp_path / "ensemble_weights.json")
    )
    monkeypatch.setattr(settings, "MIN_ENSEMBLE_CALIBRATION_SAMPLES", 30)
    monkeypatch.setattr(settings, "ENSEMBLE_HOLDOUT_FRACTION", 0.2)
    monkeypatch.setattr(settings, "ENSEMBLE_MIN_SOURCE_WEIGHT", 0.05)
    monkeypatch.setattr(settings, "ENSEMBLE_MIN_LOG_LOSS_IMPROVEMENT", 0.001)
    monkeypatch.setattr(settings, "ENSEMBLE_STATS_WEIGHT", 0.4)
    monkeypatch.setattr(settings, "ENSEMBLE_ML_WEIGHT", 0.2)
    monkeypatch.setattr(settings, "ENSEMBLE_MARKET_WEIGHT", 0.4)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES", 12)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_PRIOR_STRENGTH", 20.0)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_HALF_LIFE_DAYS", 180.0)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_MIN_DATA_QUALITY_SCORE", 0.0)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_MAX_BRIER_REGRESSION", 0.01)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_STATS_LOW_DATA_BOOST", 1.5)
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_ML_HIGH_QUALITY_BOOST", 1.5)
    return EnsembleWeightManager()


def test_optimizer_activates_only_after_holdout_improvement(
    calibration_settings: EnsembleWeightManager,
) -> None:
    result = calibration_settings.optimize_and_activate(labeled_rows(90))
    weights, metadata = calibration_settings.get_active_weights()

    assert result["status"] == "activated"
    assert result["validation_log_loss"] < result["baseline_log_loss"]
    assert weights["market"] > weights["ml"]
    assert weights["market"] > weights["stats"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert metadata["source"] == "global_bma"
    assert metadata["artifact_version"] == result["artifact_version"]


def test_insufficient_complete_samples_keep_configured_weights(
    calibration_settings: EnsembleWeightManager,
) -> None:
    rows = labeled_rows(10)
    rows[0].probability_components["components"].pop("ml")

    result = calibration_settings.optimize_and_activate(rows)
    weights, metadata = calibration_settings.get_active_weights()

    assert result == {
        "status": "insufficient_data",
        "samples": 10,
        "required_samples": 30,
    }
    assert weights == {"stats": 0.4, "ml": 0.2, "market": 0.4}
    assert metadata["source"] == "configured"


def test_candidate_without_holdout_improvement_is_rejected(
    calibration_settings: EnsembleWeightManager,
) -> None:
    rows = labeled_rows(60)
    for row in rows:
        components = row.probability_components["components"]
        components["stats"] = dict(components["market"])
        components["ml"] = dict(components["market"])

    result = calibration_settings.optimize_and_activate(rows)

    assert result["status"] == "rejected"
    assert result["improvement"] == pytest.approx(0.0)
    assert not Path(settings.ENSEMBLE_WEIGHTS_PATH).exists()


def test_artifact_write_failure_does_not_activate_candidate(
    calibration_settings: EnsembleWeightManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_write(_artifact: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(calibration_settings, "_write_artifact", fail_to_write)

    result = calibration_settings.optimize_and_activate(labeled_rows(60))

    assert result == {
        "status": "artifact_write_failed",
        "samples": 60,
        "error": "OSError",
    }


def test_invalid_artifact_falls_back_to_configured_weights(
    calibration_settings: EnsembleWeightManager,
) -> None:
    artifact_path = settings.ENSEMBLE_WEIGHTS_PATH
    with open(artifact_path, "w", encoding="utf-8") as artifact_file:
        json.dump({"schema_version": 999, "weights": {}}, artifact_file)

    weights, metadata = calibration_settings.get_active_weights()

    assert weights == {"stats": 0.4, "ml": 0.2, "market": 0.4}
    assert metadata == {
        "source": "configured",
        "league_id": None,
        "source_set": "stats+ml+market",
        "artifact_version": None,
    }


def _outcome_probabilities(
    actual_index: int, actual_probability: float
) -> dict[str, float]:
    remainder = (1.0 - actual_probability) / 2.0
    values = [remainder, remainder, remainder]
    values[actual_index] = actual_probability
    return {
        outcome: round(values[index] * 100.0, 6)
        for index, outcome in enumerate(OUTCOMES)
    }


def league_rows(count: int) -> list[SimpleNamespace]:
    rows: list[SimpleNamespace] = []
    for index in range(count):
        league_id = 203 if index % 2 == 0 else 39
        actual_index = index % len(OUTCOMES)
        stats_probability = 0.45 if league_id == 203 else 0.85
        ml_probability = 0.85 if league_id == 203 else 0.35
        rows.append(
            SimpleNamespace(
                id=index + 1,
                created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
                league_id=league_id,
                actual_result=OUTCOMES[actual_index],
                data_quality={"score": 95.0},
                probability_components={
                    "components": {
                        "stats": _outcome_probabilities(
                            actual_index, stats_probability
                        ),
                        "ml": _outcome_probabilities(actual_index, ml_probability),
                        "market": _outcome_probabilities(actual_index, 0.55),
                    }
                },
            )
        )
    return rows


def test_league_bma_learns_opposite_weights_by_league(
    calibration_settings: EnsembleWeightManager,
) -> None:
    result = calibration_settings.optimize_and_activate(league_rows(180))

    league_203, metadata_203 = calibration_settings.get_active_weights(
        203, ("stats", "ml", "market")
    )
    league_39, metadata_39 = calibration_settings.get_active_weights(
        39, ("stats", "ml", "market")
    )

    assert result["status"] == "activated"
    assert metadata_203["source"] == "league_bma"
    assert metadata_39["source"] == "league_bma"
    assert league_203["ml"] > league_203["stats"]
    assert league_39["stats"] > league_39["ml"]
    assert sum(league_203.values()) == pytest.approx(1.0)
    assert sum(league_39.values()) == pytest.approx(1.0)


def test_unknown_league_uses_stats_heavy_low_data_prior(
    calibration_settings: EnsembleWeightManager,
) -> None:
    weights, metadata = calibration_settings.get_active_weights(999, ("stats", "ml"))

    assert weights["stats"] > 2 / 3
    assert weights["stats"] > weights["ml"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert metadata["source"] == "low_data_prior"
    assert metadata["league_id"] == 999
    assert metadata["source_set"] == "stats+ml"


def test_schema_v1_global_weights_remain_compatible(
    calibration_settings: EnsembleWeightManager,
) -> None:
    artifact_path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_version": "legacy-1",
                "validation_log_loss": 0.9,
                "samples": 120,
                "weights": {"stats": 0.2, "ml": 0.3, "market": 0.5},
            }
        ),
        encoding="utf-8",
    )

    weights, metadata = calibration_settings.get_active_weights(
        203, ("stats", "ml", "market")
    )

    assert weights == {"stats": 0.2, "ml": 0.3, "market": 0.5}
    assert metadata["source"] == "learned_v1"
    assert metadata["artifact_version"] == "legacy-1"


def test_tampered_bma_artifact_is_rejected(
    calibration_settings: EnsembleWeightManager,
) -> None:
    result = calibration_settings.optimize_and_activate(labeled_rows(90))
    assert result["status"] == "activated"
    artifact_path = Path(settings.ENSEMBLE_WEIGHTS_PATH)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["global"]["stats+ml+market"]["weights"]["market"] = 0.99
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    fresh_manager = EnsembleWeightManager()
    weights, metadata = fresh_manager.get_active_weights()

    assert weights == {"stats": 0.4, "ml": 0.2, "market": 0.4}
    assert metadata["source"] == "configured"


def test_recent_evidence_outweighs_old_evidence_after_decay(
    calibration_settings: EnsembleWeightManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ENSEMBLE_BMA_HALF_LIFE_DAYS", 3.0)
    rows: list[SimpleNamespace] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(60):
        actual_index = index % len(OUTCOMES)
        recent_period = index >= 30
        rows.append(
            SimpleNamespace(
                created_at=start + timedelta(days=index),
                league_id=203,
                actual_result=OUTCOMES[actual_index],
                probability_components={
                    "components": {
                        "stats": _outcome_probabilities(
                            actual_index, 0.9 if recent_period else 0.4
                        ),
                        "ml": _outcome_probabilities(
                            actual_index, 0.4 if recent_period else 0.9
                        ),
                    }
                },
            )
        )

    profile = calibration_settings._posterior_profile(
        calibration_settings._extract_samples(rows)
    )

    assert profile is not None
    assert profile["weights"]["stats"] > profile["weights"]["ml"]


def test_post_kickoff_predictions_are_excluded_from_bma_evidence(
    calibration_settings: EnsembleWeightManager,
) -> None:
    rows = labeled_rows(30, league_id=203)
    for row in rows:
        row.kickoff = row.created_at
        row.analyzed_at = row.created_at + timedelta(minutes=1)

    assert calibration_settings._extract_samples(rows) == []
