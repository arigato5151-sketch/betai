from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import Base, HistoricalFixture
from app.prediction.ml.historical import HistoricalFeatureService
from app.tasks.jobs import _current_football_season, sync_historical_fixtures_task


def fixture_row(
    fixture_id: int,
    kickoff: datetime,
    *,
    home_team_id: int = 1,
    away_team_id: int = 2,
    home_goals: int = 2,
    away_goals: int = 1,
    season: int = 2026,
) -> dict:
    if home_goals > away_goals:
        result = "HOME_WIN"
    elif home_goals < away_goals:
        result = "AWAY_WIN"
    else:
        result = "DRAW"
    return {
        "fixture_id": fixture_id,
        "league_id": 203,
        "season": season,
        "kickoff": kickoff,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team": f"Team {home_team_id}",
        "away_team": f"Team {away_team_id}",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "actual_result": result,
        "status": "FT",
    }


@pytest.fixture
def historical_repository() -> HistoricalFixtureRepository:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield HistoricalFixtureRepository(session)


def test_historical_upsert_is_idempotent_and_updates_scores(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    kickoff = datetime(2026, 7, 1, tzinfo=UTC)

    assert historical_repository.upsert_many([fixture_row(100, kickoff)]) == 1
    assert (
        historical_repository.upsert_many(
            [fixture_row(100, kickoff, home_goals=0, away_goals=0)]
        )
        == 1
    )

    stored = historical_repository.get_by_fixture_id(100)
    assert stored is not None
    assert stored.home_goals == 0
    assert stored.actual_result == "DRAW"
    assert historical_repository.db.query(stored.__class__).count() == 1


def test_history_queries_exclude_future_matches(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 7, 10, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            fixture_row(100, cutoff - timedelta(days=2)),
            fixture_row(101, cutoff + timedelta(days=2)),
        ]
    )

    league_history = historical_repository.get_league_history(
        league_id=203, season=2026, before=cutoff
    )
    h2h = historical_repository.get_h2h(home_team_id=1, away_team_id=2, before=cutoff)

    assert [row.fixture_id for row in league_history] == [100]
    assert [row.fixture_id for row in h2h] == [100]
    assert [
        row.fixture_id
        for row in historical_repository.get_team_history(
            team_id=1, league_id=203, before=cutoff
        )
    ] == [100]


def test_historical_context_builds_elo_and_normalizes_reversed_h2h(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 7, 20, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            fixture_row(100, cutoff - timedelta(days=10)),
            fixture_row(
                101,
                cutoff - timedelta(days=5),
                home_team_id=2,
                away_team_id=1,
                home_goals=3,
                away_goals=0,
            ),
        ]
    )

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=1,
        away_team_id=2,
        league_id=203,
        before=cutoff,
    )

    assert context.home_elo < context.away_elo
    assert context.h2h_matches == [
        {"home_goals": 0, "away_goals": 3},
        {"home_goals": 2, "away_goals": 1},
    ]
    assert context.h2h_rates == {
        "home_win_rate": 0.5,
        "draw_rate": 0.0,
        "home_loss_rate": 0.5,
        "source": "historical_fixtures",
    }
    assert context.home_matches_df is not None
    assert context.away_matches_df is not None
    assert context.home_matches_df["result"].tolist() == ["W", "L"]
    assert context.home_matches_df["points"].tolist() == [3.0, 0.0]
    assert context.away_matches_df["result"].tolist() == ["L", "W"]
    assert str(context.home_matches_df["match_date"].dt.tz) == "UTC"


def test_historical_context_carries_elo_across_seasons_with_regression(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 8, 10, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            fixture_row(
                200,
                datetime(2025, 5, 1, tzinfo=UTC),
                season=2024,
            ),
            fixture_row(
                201,
                datetime(2026, 8, 1, tzinfo=UTC),
                home_team_id=3,
                away_team_id=4,
                home_goals=1,
                away_goals=1,
                season=2026,
            ),
        ]
    )

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=1,
        away_team_id=2,
        league_id=203,
        before=cutoff,
        elo_season_regression=0.25,
    )

    assert context.home_elo == pytest.approx(1512.0)
    assert context.away_elo == pytest.approx(1488.0)
    assert [
        row.season
        for row in historical_repository.get_league_history(
            league_id=203, before=cutoff
        )
    ] == [2024, 2026]


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        pytest.param(date(2026, 6, 30), 2025, id="before-season-rollover"),
        pytest.param(date(2026, 7, 1), 2026, id="after-season-rollover"),
    ],
)
def test_current_football_season(today: date, expected: int) -> None:
    assert _current_football_season(today) == expected


def test_historical_sync_task_fetches_then_persists_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tasks import jobs

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2026, 7, 1, tzinfo=UTC)

    class FakeClient:
        async def get_completed_fixtures(
            self, league_id: int, season: int
        ) -> list[dict]:
            assert (league_id, season) == (203, 2026)
            return [fixture_row(100, kickoff), fixture_row(100, kickoff)]

    monkeypatch.setattr(jobs, "ALLOWED_LEAGUE_IDS", {203})
    monkeypatch.setattr(jobs, "APIFootballClient", FakeClient)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: Session(engine))

    result = sync_historical_fixtures_task.run([2026])

    assert result == {
        "seasons": [2026],
        "fixtures_processed": 1,
        "failed_league_seasons": [],
    }
    with Session(engine) as session:
        assert session.query(HistoricalFixture).count() == 1
