from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import Base
from app.services.fixture_readiness import FixtureReadinessService


def _row(
    fixture_id: int,
    kickoff: datetime,
    *,
    league_id: int,
    home_id: int,
    away_id: int,
    home: str,
    away: str,
) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "league_id": league_id,
        "season": 2025,
        "kickoff": kickoff,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team": home,
        "away_team": away,
        "home_goals": 1,
        "away_goals": 0,
        "actual_result": "HOME_WIN",
        "status": "FT",
        "data_source": "test",
    }


def test_fixture_readiness_counts_cross_competition_history_by_team_name() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2026, 8, 14, 18, 30, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(5):
        rows.append(
            _row(
                100 + index,
                kickoff - timedelta(days=10 + index),
                league_id=203,
                home_id=10,
                away_id=20 + index,
                home="Galatasaray",
                away=f"Rakip {index}",
            )
        )
    rows.append(
        _row(
            200,
            kickoff - timedelta(days=20),
            league_id=204,
            home_id=30,
            away_id=40,
            home="Çorum FK",
            away="Bodrum",
        )
    )

    with Session(engine) as db:
        HistoricalFixtureRepository(db).upsert_many(rows)
        result = FixtureReadinessService(db).annotate(
            [
                {
                    "fixture_id": 999,
                    "league_id": 203,
                    "season": 2026,
                    "kickoff": kickoff.isoformat(),
                    "home_team_id": 2_373_329_775,
                    "away_team_id": 2_285_755_108,
                    "home_team": "Galatasaray",
                    "away_team": "Çorum",
                }
            ]
        )[0]["data_readiness"]

    assert result == {
        "status": "insufficient",
        "home_history_matches": 5,
        "away_history_matches": 1,
        "required_history_matches": 5,
        "reasons": ["away_history_insufficient"],
    }


def test_fixture_readiness_marks_both_complete_histories_sufficient() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2026, 8, 14, 18, 30, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(5):
        rows.extend(
            [
                _row(
                    300 + index,
                    kickoff - timedelta(days=10 + index),
                    league_id=203,
                    home_id=10,
                    away_id=100 + index,
                    home="Ev Takımı",
                    away=f"Ev Rakibi {index}",
                ),
                _row(
                    400 + index,
                    kickoff - timedelta(days=20 + index),
                    league_id=203,
                    home_id=200 + index,
                    away_id=20,
                    home=f"Dep Rakibi {index}",
                    away="Deplasman Takımı",
                ),
            ]
        )

    with Session(engine) as db:
        HistoricalFixtureRepository(db).upsert_many(rows)
        readiness = FixtureReadinessService(db).annotate(
            [
                {
                    "fixture_id": 999,
                    "league_id": 203,
                    "season": 2026,
                    "kickoff": kickoff.isoformat(),
                    "home_team_id": 10,
                    "away_team_id": 20,
                    "home_team": "Ev Takımı",
                    "away_team": "Deplasman Takımı",
                }
            ]
        )[0]["data_readiness"]

    assert readiness["status"] == "sufficient"
    assert readiness["home_history_matches"] == 5
    assert readiness["away_history_matches"] == 5
    assert readiness["reasons"] == []


def test_fixture_readiness_merges_provider_ids_by_canonical_team_name() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2026, 8, 14, 18, 30, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(5):
        rows.extend(
            [
                _row(
                    500 + index,
                    kickoff - timedelta(days=10 + index),
                    league_id=39,
                    home_id=100 if index < 2 else 200,
                    away_id=300 + index,
                    home="Man United" if index < 2 else "Manchester United",
                    away=f"Rakip {index}",
                ),
                _row(
                    600 + index,
                    kickoff - timedelta(days=20 + index),
                    league_id=39,
                    home_id=400 + index,
                    away_id=500,
                    home=f"Diğer Rakip {index}",
                    away="Hull City",
                ),
            ]
        )

    with Session(engine) as db:
        HistoricalFixtureRepository(db).upsert_many(rows)
        readiness = FixtureReadinessService(db).annotate(
            [
                {
                    "fixture_id": 999,
                    "league_id": 39,
                    "season": 2026,
                    "kickoff": kickoff.isoformat(),
                    "home_team_id": 33,
                    "away_team_id": 64,
                    "home_team": "Manchester United",
                    "away_team": "Hull City",
                }
            ]
        )[0]["data_readiness"]

    assert readiness["status"] == "sufficient"
    assert readiness["home_history_matches"] == 5


def test_fixture_readiness_does_not_count_duplicate_provider_rows_twice() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2026, 8, 14, 18, 30, tzinfo=UTC)
    rows = [
        _row(
            700 + index,
            kickoff - timedelta(days=10 + index),
            league_id=39,
            home_id=100,
            away_id=300 + index,
            home="Everton",
            away=f"Rakip {index}",
        )
        for index in range(4)
    ]
    rows.append(
        _row(
            800,
            kickoff - timedelta(days=10),
            league_id=39,
            home_id=200,
            away_id=999,
            home="Everton",
            away="Aynı Maç Rakibi",
        )
    )

    with Session(engine) as db:
        HistoricalFixtureRepository(db).upsert_many(rows)
        readiness = FixtureReadinessService(db).annotate(
            [
                {
                    "fixture_id": 999,
                    "league_id": 39,
                    "season": 2026,
                    "kickoff": kickoff.isoformat(),
                    "home_team_id": 45,
                    "away_team_id": 52,
                    "home_team": "Everton",
                    "away_team": "Unknown Team",
                }
            ]
        )[0]["data_readiness"]

    assert readiness["home_history_matches"] == 4
    assert readiness["status"] == "insufficient"


def test_fixture_readiness_uses_a_bounded_targeted_history_query() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2026, 8, 14, 18, 30, tzinfo=UTC)
    statements: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture_statement(*args) -> None:
        statements.append(str(args[2]))

    with Session(engine) as db:
        HistoricalFixtureRepository(db).upsert_many(
            [
                _row(
                    900 + index,
                    kickoff - timedelta(days=index + 1),
                    league_id=203,
                    home_id=10,
                    away_id=100 + index,
                    home="Target Team",
                    away=f"Opponent {index}",
                )
                for index in range(50)
            ]
        )
        FixtureReadinessService(db).annotate(
            [
                {
                    "fixture_id": 999,
                    "league_id": 203,
                    "kickoff": kickoff.isoformat(),
                    "home_team_id": 10,
                    "away_team_id": 20,
                    "home_team": "Target Team",
                    "away_team": "Other Team",
                }
            ]
        )

    readiness_queries = [
        statement for statement in statements if "historical_fixtures" in statement
    ]
    assert any("LIMIT" in statement.upper() for statement in readiness_queries)
