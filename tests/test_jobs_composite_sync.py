from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    Base,
    FixtureOddsSnapshot,
    HistoricalFixture,
    MatchPrediction,
)
from app.tasks.jobs import _sync_completed_matches


def _prediction(**overrides) -> MatchPrediction:
    values: dict = {
        "home_team": "PEC Zwolle",
        "away_team": "Ajax",
        "league_id": 88,
        "home_team_id": 645,
        "away_team_id": 11,
        "kickoff": datetime(2026, 8, 12, 19, tzinfo=UTC),
        "fixture_id": None,
        "fixture_source": None,
        "provider_fixture_id": None,
        "actual_result": None,
        "prediction": "AWAY_WIN",
        "odd": 1.8,
        "is_value_bet": 1,
        "edge": 2.4,
        "training_eligible": True,
        "result_verification_status": "pending",
        "created_at": datetime.now(UTC).replace(tzinfo=None),
    }
    values.update(overrides)
    return MatchPrediction(**values)


def test_composite_prediction_without_provider_id_stays_pending(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(_prediction())
        db.commit()

    monkeypatch.setattr("app.tasks.results.SessionLocal", session_factory)

    result = _sync_completed_matches(Mock(), Mock())

    assert result["status"] == "ready"
    assert result.get("verified", 0) == 0
    assert result["pending"] == 1

    with session_factory() as db:
        record = db.query(MatchPrediction).one()
        assert record.actual_result is None
        assert record.result_verification_status == "pending"


def test_composite_prediction_verified_against_local_history(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(_prediction())
        db.add(
            HistoricalFixture(
                fixture_id=9_000_088,
                league_id=88,
                season=2026,
                kickoff=datetime(2026, 8, 12, 19, tzinfo=UTC),
                home_team_id=645,
                away_team_id=11,
                home_team="PEC Zwolle",
                away_team="Ajax",
                home_goals=0,
                away_goals=2,
                actual_result="AWAY_WIN",
                status="FT",
                data_source="api_football",
            )
        )
        db.commit()

    monkeypatch.setattr("app.tasks.results.SessionLocal", session_factory)
    monkeypatch.setattr("app.tasks.results.retrain_ml_model_task.delay", Mock())

    result = _sync_completed_matches(Mock(), Mock())

    assert result.get("verified") == 1
    assert result.get("pending", 0) == 0

    with session_factory() as db:
        record = db.query(MatchPrediction).one()
        assert record.actual_result == "AWAY_WIN"
        assert record.actual_score_home == 0
        assert record.actual_score_away == 2
        assert record.result_verification_status == "verified"


def test_verified_result_uses_timestamped_pre_kickoff_closing_snapshot(
    monkeypatch,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    kickoff = datetime(2026, 8, 12, 19, tzinfo=UTC)
    fixture_id = 9_000_091
    with session_factory() as db:
        db.add(
            _prediction(
                fixture_id=fixture_id,
                fixture_source="api_football",
                provider_fixture_id=str(fixture_id),
                kickoff=kickoff,
            )
        )
        db.add(
            HistoricalFixture(
                fixture_id=fixture_id,
                league_id=88,
                season=2026,
                kickoff=kickoff,
                home_team_id=645,
                away_team_id=11,
                home_team="PEC Zwolle",
                away_team="Ajax",
                home_goals=0,
                away_goals=2,
                actual_result="AWAY_WIN",
                status="FT",
                data_source="api_football",
            )
        )
        db.add(
            FixtureOddsSnapshot(
                fixture_id=fixture_id,
                home_odd=3.1,
                draw_odd=3.4,
                away_odd=2.2,
                source="api_football_odds",
                captured_at=kickoff - timedelta(minutes=15),
            )
        )
        db.commit()

    monkeypatch.setattr("app.tasks.results.SessionLocal", session_factory)
    monkeypatch.setattr("app.tasks.results.retrain_ml_model_task.delay", Mock())
    api_client = Mock()

    result = _sync_completed_matches(api_client, Mock())

    assert result.get("verified") == 1
    api_client.get_fixture_market.assert_not_called()
    with session_factory() as db:
        record = db.query(MatchPrediction).one()
        assert record.closing_odds == 2.2
        assert record.clv == -0.1818


def test_composite_matching_ignores_shifted_kickoff(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(_prediction())
        db.add(
            HistoricalFixture(
                fixture_id=9_000_089,
                league_id=88,
                season=2026,
                kickoff=datetime(2026, 8, 20, 19, tzinfo=UTC),
                home_team_id=645,
                away_team_id=11,
                home_team="PEC Zwolle",
                away_team="Ajax",
                home_goals=3,
                away_goals=1,
                actual_result="HOME_WIN",
                status="FT",
                data_source="api_football",
            )
        )
        db.commit()

    monkeypatch.setattr("app.tasks.results.SessionLocal", session_factory)

    result = _sync_completed_matches(Mock(), Mock())

    assert result.get("verified", 0) == 0
    assert result["pending"] == 1


def test_composite_requires_single_unambiguous_row(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    kickoff = datetime(2026, 8, 12, 19, tzinfo=UTC)
    with session_factory() as db:
        db.add(
            _prediction(
                kickoff=kickoff,
            )
        )
        db.add(
            HistoricalFixture(
                fixture_id=9_000_088,
                league_id=88,
                season=2026,
                kickoff=kickoff,
                home_team_id=645,
                away_team_id=11,
                home_team="PEC Zwolle",
                away_team="Ajax",
                home_goals=0,
                away_goals=2,
                actual_result="AWAY_WIN",
                status="FT",
                data_source="api_football",
            )
        )
        db.commit()
    with session_factory() as db:
        db.add(
            HistoricalFixture(
                fixture_id=9_000_090,
                league_id=88,
                season=2026,
                kickoff=kickoff + timedelta(minutes=5),
                home_team_id=645,
                away_team_id=11,
                home_team="PEC Zwolle",
                away_team="Ajax",
                home_goals=1,
                away_goals=0,
                actual_result="HOME_WIN",
                status="FT",
                data_source="api_football",
            )
        )
        db.commit()

    monkeypatch.setattr("app.tasks.results.SessionLocal", session_factory)

    result = _sync_completed_matches(Mock(), Mock())

    assert result.get("verified", 0) == 0
    assert result["pending"] == 1
    with session_factory() as db:
        record = db.query(MatchPrediction).one()
        assert record.actual_result is None
