from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Base, MatchPrediction
from app.services.model_monitoring import ModelMonitoringService


def _prediction(index: int, *, correct: bool) -> MatchPrediction:
    return MatchPrediction(
        fixture_id=index,
        training_eligible=True,
        result_verification_status="verified",
        actual_result="HOME_WIN",
        prob_home=80.0 if correct else 5.0,
        prob_draw=10.0 if correct else 10.0,
        prob_away=10.0 if correct else 85.0,
    )


def test_monitor_reports_insufficient_data() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_prediction(1, correct=True))
        db.commit()
        result = ModelMonitoringService(db).snapshot()
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
        result = ModelMonitoringService(db).snapshot()
    assert result["status"] == "drift"
    assert result["drift_detected"] is True
    assert result["recent_brier"] > result["baseline_brier"]
