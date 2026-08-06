from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base, HistoricalFixture, MatchPrediction
from app.services.data_quality import DataQualityService


def test_data_quality_snapshot_reports_coverage_and_freshness() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime(2026, 7, 23, 12, tzinfo=UTC)

    with Session(engine) as session:
        session.add(
            HistoricalFixture(
                fixture_id=1,
                league_id=203,
                season=2026,
                kickoff=now - timedelta(days=1),
                home_team_id=1,
                away_team_id=2,
                home_team="Home",
                away_team="Away",
                home_goals=2,
                away_goals=1,
                home_starting_xi=list(range(1, 12)),
                away_starting_xi=list(range(12, 23)),
                actual_result="HOME_WIN",
                status="FT",
                updated_at=now - timedelta(hours=2),
            )
        )
        session.add(
            MatchPrediction(
                training_eligible=True,
                result_verification_status="verified",
                prediction="HOME_WIN",
                actual_result="HOME_WIN",
                closing_odds=1.9,
                feature_schema_version="v1",
                ensemble_version="v1",
                analyzed_at=now - timedelta(days=1),
            )
        )
        session.commit()

        snapshot = DataQualityService(session).snapshot(now)

    assert snapshot["historical"]["fixtures"] == 1
    assert snapshot["historical"]["freshness_hours"] == 2.0
    assert snapshot["historical"]["lineup_coverage_pct"] == 100.0
    assert snapshot["historical"]["source_counts"] == {"api_football": 1}
    assert snapshot["historical"]["current_season"] == 2026
    assert snapshot["historical"]["current_season_covered_leagues"] == 1
    assert 203 not in snapshot["historical"]["current_season_missing_league_ids"]
    assert snapshot["predictions"]["labeled_coverage_pct"] == 100.0
    assert snapshot["predictions"]["closing_odds_coverage_pct"] == 100.0
    assert snapshot["predictions"]["provenance_coverage_pct"] == 100.0
    assert snapshot["status"] == "healthy"


def test_sync_run_lifecycle_is_visible_in_snapshot() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        service = DataQualityService(session)
        run = service.start_sync("historical_fixtures", [2025, 2026])
        service.finish_sync(
            run.id,
            processed=12,
            failures=[{"league_id": 39, "season": 2025, "error": "TimeoutError"}],
        )
        snapshot = service.snapshot()

    assert snapshot["latest_sync"]["status"] == "partial"
    assert snapshot["latest_sync"]["fixtures_processed"] == 12
    assert snapshot["latest_sync"]["failures"][0]["error"] == "TimeoutError"


def test_lineup_coverage_accepts_namespaced_ids_and_rejects_json_null() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime(2026, 8, 3, 12, tzinfo=UTC)

    with Session(engine) as session:
        common = {
            "league_id": 179,
            "season": 2026,
            "kickoff": now - timedelta(days=1),
            "home_team_id": -1,
            "away_team_id": -2,
            "home_team": "Home",
            "away_team": "Away",
            "home_goals": 1,
            "away_goals": 0,
            "actual_result": "HOME_WIN",
            "status": "FT",
        }
        session.add(
            HistoricalFixture(
                fixture_id=-1,
                **common,
                home_starting_xi=list(range(-11, 0)),
                away_starting_xi=list(range(-22, -11)),
            )
        )
        session.add(
            HistoricalFixture(
                fixture_id=-2,
                **common,
                home_starting_xi=None,
                away_starting_xi=None,
            )
        )
        session.commit()

        snapshot = DataQualityService(session).snapshot(now)

    assert snapshot["historical"]["lineup_coverage_pct"] == 50.0


def test_current_season_coverage_requires_volume_and_freshness() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    now = datetime(2026, 10, 1, 12, tzinfo=UTC)

    with Session(engine) as session:
        for index in range(30):
            session.add(
                HistoricalFixture(
                    fixture_id=1000 + index,
                    league_id=39,
                    season=2026,
                    kickoff=now - timedelta(days=45, minutes=index),
                    home_team_id=1,
                    away_team_id=2,
                    home_team="Home",
                    away_team="Away",
                    home_goals=1,
                    away_goals=0,
                    actual_result="HOME_WIN",
                    status="FT",
                )
            )
        session.add(
            HistoricalFixture(
                fixture_id=2000,
                league_id=203,
                season=2026,
                kickoff=now - timedelta(days=1),
                home_team_id=3,
                away_team_id=4,
                home_team="Fresh Home",
                away_team="Fresh Away",
                home_goals=1,
                away_goals=1,
                actual_result="DRAW",
                status="FT",
            )
        )
        session.commit()

        coverage = {
            row["league_id"]: row
            for row in DataQualityService(session).snapshot(now)["historical"][
                "current_season_coverage"
            ]
        }

    assert coverage[39]["available"] is True
    assert coverage[39]["fresh"] is False
    assert coverage[39]["covered"] is False
    assert coverage[203]["available"] is True
    assert coverage[203]["fresh"] is True
    assert coverage[203]["fixtures"] < coverage[203]["expected_minimum_fixtures"]
    assert coverage[203]["covered"] is False
