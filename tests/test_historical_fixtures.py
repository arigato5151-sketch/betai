from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import Base, HistoricalFixture, HistoricalPlayerPerformance
from app.prediction.ml.historical import HistoricalFeatureService
from app.tasks.jobs import (
    _current_football_season,
    _enrich_historical_player_context,
    sync_football_data_fixtures_task,
    sync_historical_fixtures_task,
)


def fixture_row(
    fixture_id: int,
    kickoff: datetime,
    *,
    home_team_id: int = 1,
    away_team_id: int = 2,
    home_goals: int = 2,
    away_goals: int = 1,
    season: int = 2026,
    league_id: int = 203,
    home_starting_xi: list[int] | None = None,
    away_starting_xi: list[int] | None = None,
) -> dict:
    if home_goals > away_goals:
        result = "HOME_WIN"
    elif home_goals < away_goals:
        result = "AWAY_WIN"
    else:
        result = "DRAW"
    return {
        "fixture_id": fixture_id,
        "league_id": league_id,
        "season": season,
        "kickoff": kickoff,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team": f"Team {home_team_id}",
        "away_team": f"Team {away_team_id}",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_starting_xi": home_starting_xi,
        "away_starting_xi": away_starting_xi,
        "actual_result": result,
        "status": "FT",
    }


def player_context_rows(
    fixture_id: int,
    kickoff: datetime,
    *,
    home_team_id: int = 1,
    away_team_id: int = 2,
) -> list[dict[str, object]]:
    return [
        {
            "fixture_id": fixture_id,
            "league_id": 203,
            "kickoff": kickoff,
            "team_id": team_id,
            "player_id": player_id,
            "started": True,
            "minutes": 90,
            "rating": 8.1,
            "position": "M",
            "goals": 0,
            "assists": 0,
            "source": "api_football_fixture_players",
        }
        for team_id, player_ids in (
            (home_team_id, range(1, 8)),
            (away_team_id, range(20, 27)),
        )
        for player_id in player_ids
    ]


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
            [
                fixture_row(
                    100,
                    kickoff,
                    home_goals=0,
                    away_goals=0,
                    home_starting_xi=list(range(1, 12)),
                    away_starting_xi=list(range(20, 31)),
                )
            ]
        )
        == 1
    )

    stored = historical_repository.get_by_fixture_id(100)
    assert stored is not None
    assert stored.home_goals == 0
    assert stored.actual_result == "DRAW"
    assert stored.home_starting_xi == list(range(1, 12))
    assert stored.away_starting_xi == list(range(20, 31))
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
                home_starting_xi=list(range(20, 31)),
                away_starting_xi=list(range(1, 12)),
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
    assert context.home_previous_starting_xi == list(range(1, 12))
    assert context.away_previous_starting_xi == list(range(20, 31))


def test_latest_starting_xi_can_cross_competitions(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 7, 20, tzinfo=UTC)
    league_lineup = list(range(1, 12))
    cup_lineup = list(range(101, 112))
    historical_repository.upsert_many(
        [
            fixture_row(
                110,
                cutoff - timedelta(days=5),
                league_id=203,
                home_starting_xi=league_lineup,
            ),
            fixture_row(
                111,
                cutoff - timedelta(days=2),
                league_id=39,
                home_starting_xi=cup_lineup,
            ),
        ]
    )

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=1,
        away_team_id=2,
        league_id=203,
        before=cutoff,
    )

    assert context.home_previous_starting_xi == cup_lineup
    assert (
        historical_repository.get_last_starting_xi(
            team_id=1,
            league_id=203,
            before=cutoff,
        )
        == league_lineup
    )


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


def test_historical_context_resolves_external_team_ids_by_name(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 1, 10, tzinfo=UTC)
    row = fixture_row(
        -(1 << 40),
        cutoff - timedelta(days=7),
        home_team_id=-(1 << 41),
        away_team_id=-(1 << 42),
        season=2025,
    )
    row["home_team"] = "Man City"
    row["away_team"] = "Wolves"
    row["data_source"] = "football_data_csv"
    historical_repository.upsert_many([row])

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=50,
        away_team_id=51,
        home_team_name="Manchester City",
        away_team_name="Wolverhampton Wanderers",
        league_id=203,
        before=cutoff,
    )

    assert context.home_elo > 1500.0
    assert context.away_elo < 1500.0
    assert context.home_matches_df is not None
    assert context.home_matches_df["goals_for"].tolist() == [2]
    assert context.away_matches_df is not None
    assert context.away_matches_df["goals_for"].tolist() == [1]


def test_historical_context_resolves_conservative_provider_name_variants(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 8, 1, tzinfo=UTC)
    row = fixture_row(
        -(1 << 43),
        cutoff - timedelta(days=7),
        home_team_id=-(1 << 44),
        away_team_id=-(1 << 45),
        league_id=235,
        season=2026,
    )
    row["home_team"] = "Rodina Moscow"
    row["away_team"] = "FK Rostov"
    row["data_source"] = "football_data_csv"
    historical_repository.upsert_many([row])

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=6822,
        away_team_id=779,
        home_team_name="Rodina Moskva",
        away_team_name="FC Rostov",
        league_id=235,
        before=cutoff,
    )

    assert context.home_matches_df is not None
    assert context.home_matches_df["goals_for"].tolist() == [2]
    assert context.away_matches_df is not None
    assert context.away_matches_df["goals_for"].tolist() == [1]


def test_historical_context_rejects_ambiguous_provider_name_mapping(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 8, 1, tzinfo=UTC)
    first = fixture_row(
        -(1 << 46),
        cutoff - timedelta(days=8),
        home_team_id=-(1 << 47),
        away_team_id=-(1 << 48),
        league_id=235,
        season=2026,
    )
    second = fixture_row(
        -(1 << 49),
        cutoff - timedelta(days=7),
        home_team_id=-(1 << 50),
        away_team_id=-(1 << 51),
        league_id=235,
        season=2026,
    )
    first["home_team"] = "FC Duplicate"
    second["home_team"] = "FK Duplicate"
    historical_repository.upsert_many([first, second])

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=999001,
        away_team_id=999002,
        home_team_name="FC Duplicate",
        away_team_name="Unknown Away",
        league_id=235,
        before=cutoff,
    )

    assert context.home_matches_df is not None
    assert context.home_matches_df.empty
    assert context.home_elo_available is False


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        pytest.param(date(2026, 6, 30), 2025, id="before-season-rollover"),
        pytest.param(date(2026, 7, 1), 2026, id="after-season-rollover"),
    ],
)
def test_current_football_season(today: date, expected: int) -> None:
    assert _current_football_season(today) == expected


@pytest.mark.asyncio
async def test_player_context_enrichment_replaces_incomplete_embedded_rows() -> None:
    kickoff = datetime(2026, 7, 1, tzinfo=UTC)
    row = fixture_row(100, kickoff)
    row["player_performances"] = player_context_rows(100, kickoff)[:1]

    class FakeClient:
        calls = 0

        async def get_fixture_player_context(self, **kwargs) -> dict[str, object]:
            self.calls += 1
            return {
                "home_starting_xi": list(range(1, 12)),
                "away_starting_xi": list(range(20, 31)),
                "player_performances": player_context_rows(100, kickoff),
            }

    client = FakeClient()
    failures = await _enrich_historical_player_context(client, [row], set())

    assert failures == 0
    assert client.calls == 1
    assert len(row["player_performances"]) == 14


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

        async def get_fixture_player_context(self, **kwargs) -> dict[str, object]:
            assert kwargs == {
                "fixture_id": 100,
                "league_id": 203,
                "kickoff": kickoff,
                "home_team_id": 1,
                "away_team_id": 2,
            }
            return {
                "home_starting_xi": list(range(1, 12)),
                "away_starting_xi": list(range(20, 31)),
                "player_performances": player_context_rows(100, kickoff),
            }

    monkeypatch.setattr(jobs, "ALLOWED_LEAGUE_IDS", {2, 3, 203, 848})
    monkeypatch.setattr(jobs, "APIFootballClient", FakeClient)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: Session(engine))

    result = sync_historical_fixtures_task.run([2026], [203])

    assert result == {
        "seasons": [2026],
        "fixtures_processed": 1,
        "player_performances_processed": 14,
        "player_context_failures": 0,
        "failed_league_seasons": [],
    }
    with Session(engine) as session:
        assert session.query(HistoricalFixture).count() == 1
        assert session.query(HistoricalPlayerPerformance).count() == 14


def test_historical_sync_rejects_unsupported_league_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tasks import jobs

    monkeypatch.setattr(jobs, "ALLOWED_LEAGUE_IDS", {2, 3, 848})

    with pytest.raises(ValueError, match=r"Unsupported league_ids: \[999999\]"):
        sync_historical_fixtures_task.run([2026], [999999])


def test_football_data_sync_task_persists_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.football_data_csv import FootballDataImport
    from app.tasks import jobs

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    kickoff = datetime(2025, 8, 8, tzinfo=UTC)
    row = fixture_row(100, kickoff, season=2025)
    row["fixture_id"] = -(1 << 40)
    row["home_team_id"] = -(1 << 41)
    row["away_team_id"] = -(1 << 42)
    row["data_source"] = "football_data_csv"
    row["home_shots"] = 14
    row["away_shots_on_target"] = 3
    row["opening_home_odd"] = 2.0
    row["closing_home_odd"] = 1.8

    class FakeClient:
        supported_league_ids = frozenset({203})

        async def get_completed_fixtures(
            self, league_id: int, season: int
        ) -> FootballDataImport:
            assert (league_id, season) == (203, 2025)
            return FootballDataImport(fixtures=[row, row], skipped_rows=2)

    monkeypatch.setattr(jobs, "FootballDataCSVClient", FakeClient)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: Session(engine))

    result = sync_football_data_fixtures_task.run([2025])

    assert result == {
        "seasons": [2025],
        "fixtures_processed": 1,
        "skipped_incomplete_rows": 2,
        "failed_league_seasons": [],
    }
    with Session(engine) as session:
        stored = session.query(HistoricalFixture).one()
        assert stored.fixture_id == -(1 << 40)
        assert stored.data_source == "football_data_csv"
        assert stored.home_shots == 14
        assert stored.away_shots_on_target == 3
        assert stored.opening_home_odd == 2.0
        assert stored.closing_home_odd == 1.8


def test_football_data_sync_falls_back_until_new_feed_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.football_data_csv import (
        FootballDataDownloadError,
        FootballDataImport,
    )
    from app.tasks import jobs

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    row = fixture_row(
        -(1 << 40),
        datetime(2025, 8, 8, tzinfo=UTC),
        season=2025,
    )
    row["data_source"] = "football_data_csv"

    current_row = fixture_row(
        -(1 << 41),
        datetime(2026, 7, 25, tzinfo=UTC),
        league_id=235,
        season=2026,
    )
    current_row["data_source"] = "football_data_csv"
    calls: list[tuple[int, int]] = []

    class FakeClient:
        supported_league_ids = frozenset({39, 235})

        async def get_completed_fixtures(
            self, league_id: int, season: int
        ) -> FootballDataImport:
            calls.append((league_id, season))
            if league_id == 39 and season == 2026:
                raise FootballDataDownloadError(
                    "not published",
                    status_code=404,
                )
            if league_id == 39 and season == 2025:
                return FootballDataImport(fixtures=[row], skipped_rows=0)
            assert league_id == 235
            assert season == 2026
            return FootballDataImport(fixtures=[current_row], skipped_rows=0)

    monkeypatch.setattr(jobs, "_current_football_season", lambda: 2026)
    monkeypatch.setattr(jobs, "FootballDataCSVClient", FakeClient)
    monkeypatch.setattr(jobs, "SessionLocal", lambda: Session(engine))

    result = sync_football_data_fixtures_task.run()

    assert result == {
        "seasons": [2025, 2026],
        "fixtures_processed": 2,
        "skipped_incomplete_rows": 0,
        "failed_league_seasons": [],
        "league_season_fallbacks": [
            {
                "league_id": 39,
                "from_season": 2026,
                "to_season": 2025,
            }
        ],
    }
    assert calls == [(39, 2026), (39, 2025), (235, 2026)]
    with Session(engine) as session:
        assert session.query(HistoricalFixture).count() == 2
