from datetime import UTC, datetime
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, MatchPrediction
from app.tasks.jobs import _sync_completed_matches


def test_composite_prediction_without_provider_id_stays_pending(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(
            MatchPrediction(
                home_team="PEC Zwolle",
                away_team="Ajax",
                league_id=88,
                kickoff=datetime(2026, 8, 12, 19, tzinfo=UTC),
                fixture_id=None,
                fixture_source=None,
                provider_fixture_id=None,
                actual_result=None,
                prediction="AWAY_WIN",
                odd=1.8,
                is_value_bet=1,
                edge=2.4,
                training_eligible=True,
                result_verification_status="pending",
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
        db.commit()

    monkeypatch.setattr("app.tasks.jobs.SessionLocal", session_factory)

    result = _sync_completed_matches(Mock(), Mock())

    assert result["status"] == "ready"
    assert result.get("verified", 0) == 0
    assert result["pending"] == 1

    with session_factory() as db:
        record = db.query(MatchPrediction).one()
        assert record.actual_result is None
        assert record.result_verification_status == "pending"
