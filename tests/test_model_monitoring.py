from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.api.endpoints import MLStatusResponse
from app.db.models import Base, MatchPrediction
from app.services.model_monitoring import ModelMonitoringService


def _prediction(index: int, *, correct: bool) -> MatchPrediction:
    return MatchPrediction(
        fixture_id=index,
        training_eligible=True,
        result_verification_status="verified",
        actual_result="HOME_WIN",
        # Final ensemble stays good; monitoring must inspect the ML component.
        prob_home=80.0,
        prob_draw=10.0,
        prob_away=10.0,
        probability_components={
            "components": {
                "ml": {
                    "HOME_WIN": 80.0 if correct else 5.0,
                    "DRAW": 10.0,
                    "AWAY_WIN": 10.0 if correct else 85.0,
                }
            }
        },
        model_artifact_version="model-v1",
        kickoff=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
        analyzed_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
    )


def test_monitor_reports_insufficient_data() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_prediction(1, correct=True))
        db.commit()
        result = ModelMonitoringService(db).snapshot("model-v1")
    assert result["status"] == "insufficient_data"
    assert result["drift_detected"] is False


def test_monitor_detects_recent_brier_drift(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MODEL_DRIFT_MIN_SAMPLES", 10)
    monkeypatch.setattr(settings, "MODEL_DRIFT_WINDOW_SIZE", 10)
    monkeypatch.setattr(settings, "MODEL_DRIFT_BRIER_THRESHOLD", 0.04)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([_prediction(index, correct=True) for index in range(1, 11)])
        db.add_all([_prediction(index, correct=False) for index in range(11, 21)])
        db.commit()
        result = ModelMonitoringService(db).snapshot("model-v1")
    assert result["status"] == "drift"
    assert result["drift_detected"] is True
    assert result["recent_brier"] > result["baseline_brier"]
    assert result["brier_delta_lower_bound"] >= result["threshold"]
    assert result["metric_source"] == "ml_component"
    assert result["artifact_version"] == "model-v1"


def test_monitor_excludes_other_artifacts_and_rows_without_ml_component(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "MODEL_DRIFT_MIN_SAMPLES", 10)
    monkeypatch.setattr(settings, "MODEL_DRIFT_WINDOW_SIZE", 10)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    other = [_prediction(index, correct=False) for index in range(1, 21)]
    for row in other:
        row.model_artifact_version = "model-v2"
    missing_component = [_prediction(index + 30, correct=False) for index in range(20)]
    for row in missing_component:
        row.probability_components = None
    with Session(engine) as db:
        db.add_all(other + missing_component)
        db.commit()
        result = ModelMonitoringService(db).snapshot("model-v1")

    assert result["status"] == "insufficient_data"
    assert result["samples"] == 0
    assert result["drift_detected"] is False


def test_monitor_requires_an_active_artifact() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        result = ModelMonitoringService(db).snapshot(None)

    assert result["status"] == "model_unavailable"
    assert result["metric_source"] == "ml_component"
    assert result["window_size"] == 0
    assert result["brier_delta_lower_bound"] is None
    assert result["threshold"] == settings.MODEL_DRIFT_BRIER_THRESHOLD
    assert result["evaluated_at"]


def test_ml_status_contract_accepts_complete_monitoring_snapshot() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        monitoring = ModelMonitoringService(db).snapshot("model-v1")

    response = MLStatusResponse.model_validate(
        {
            "ready": True,
            "model_name": "Test Model",
            "artifact_version": "model-v1",
            "metrics": {"samples": 100, "brier_score": 0.2},
            "runtime": {"inference_success": 5, "inference_failure": 1},
            "rollback_available": True,
            "training_data": {
                "labeled_predictions": 30,
                "historical_fixtures": 100,
                "minimum_samples": 30,
                "historical_minimum_team_matches": 3,
            },
            "monitoring": monitoring,
        }
    )

    assert response.monitoring.status == "insufficient_data"
    assert response.monitoring.artifact_version == response.artifact_version
