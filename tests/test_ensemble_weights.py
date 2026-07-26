import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.prediction.ensemble_weights import EnsembleWeightManager

OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


def labeled_rows(count: int) -> list[SimpleNamespace]:
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
                created_at=f"2026-01-{(index % 28) + 1:02d}T00:00:00",
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
    assert metadata["source"] == "learned"
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
        "samples": 9,
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
    assert metadata == {"source": "configured", "artifact_version": None}
