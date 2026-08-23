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
    _missing_fixture_history_scopes,
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


def test_xg_training_and_backfill_queries_are_bounded(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    kickoff = datetime(2026, 8, 1, 18, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            {
                **fixture_row(300 + index, kickoff + timedelta(days=index)),
                "xg_source": "understat" if index < 2 else None,
                "home_xg": 1.2 if index < 2 else None,
                "away_xg": 0.8 if index < 2 else None,
            }
            for index in range(4)
        ]
    )

    observed = historical_repository.get_recent_observed_xg(1)
    pending = historical_repository.get_missing_xg(1)

    assert [fixture.fixture_id for fixture in observed] == [301]
    assert [fixture.fixture_id for fixture in pending] == [302]


def test_get_by_composite_key_requires_single_row(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    kickoff = datetime(2026, 8, 12, 19, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            fixture_row(100, kickoff, home_team_id=645, away_team_id=11),
            fixture_row(
                101,
                kickoff + timedelta(minutes=5),
                home_team_id=645,
                away_team_id=11,
                home_goals=1,
                away_goals=0,
            ),
        ]
    )

    assert (
        historical_repository.get_by_composite_key(
            league_id=203, home_team_id=645, away_team_id=11, kickoff=kickoff
        )
        is None
    )


def test_get_by_composite_key_matches_window_and_returns_none_for_gaps(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    kickoff = datetime(2026, 8, 12, 19, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            fixture_row(200, kickoff, home_team_id=645, away_team_id=11),
            fixture_row(
                201, kickoff + timedelta(hours=48), home_team_id=645, away_team_id=11
            ),
        ]
    )

    matched = historical_repository.get_by_composite_key(
        league_id=203, home_team_id=645, away_team_id=11, kickoff=kickoff
    )
    assert matched is not None
    assert matched.fixture_id == 200

    assert (
        historical_repository.get_by_composite_key(
            league_id=203,
            home_team_id=645,
            away_team_id=11,
            kickoff=kickoff + timedelta(hours=48),
        )
        is not None
    )
    assert (
        historical_repository.get_by_composite_key(
            league_id=203, home_team_id=645, away_team_id=99, kickoff=kickoff
        )
        is None
    )
    assert (
        historical_repository.get_by_composite_key(
            league_id=203, home_team_id=645, away_team_id=11, kickoff=None
        )
        is None
    )


def test_history_upsert_many_chunks_large_batches_without_sqlite_bind_limit(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    """A bulk import larger than SQLite's 999 bind-variable limit persists fully."""
    kickoff = datetime(2026, 7, 1, tzinfo=UTC)
    first_batch = [
        fixture_row(fixture_id, kickoff + timedelta(minutes=fixture_id))
        for fixture_id in range(1, 1201)
    ]
    assert historical_repository.upsert_many(first_batch) == 1200

    stored = historical_repository.db.query(HistoricalFixture).count()
    assert stored == 1200

    updated_id = 700
    updated_batch = [
        fixture_row(updated_id, kickoff, home_goals=3, away_goals=3),
        fixture_row(12000, kickoff + timedelta(minutes=1)),
    ]
    assert historical_repository.upsert_many(updated_batch) == 2
    result = historical_repository.get_by_fixture_id(updated_id)
    assert result is not None and result.actual_result == "DRAW"
    assert historical_repository.db.query(HistoricalFixture).count() == 1201


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


def test_historical_context_uses_promoted_team_history_from_another_league(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 8, 14, tzinfo=UTC)
    row = fixture_row(
        -(1 << 52),
        cutoff - timedelta(days=30),
        home_team_id=1_000_138_951,
        away_team_id=1_000_138_974,
        league_id=204,
        season=2025,
    )
    row["home_team"] = "Çorum FK"
    row["away_team"] = "Bodrum"
    row["data_source"] = "thesportsdb"
    historical_repository.upsert_many([row])

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=2_373_329_775,
        away_team_id=2_285_755_108,
        home_team_name="Galatasaray",
        away_team_name="Çorum",
        league_id=203,
        before=cutoff,
    )

    assert context.away_matches_df is not None
    assert context.away_matches_df["goals_for"].tolist() == [2]
    assert context.away_elo_available is False


def test_historical_context_merges_recent_form_across_provider_team_ids(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    cutoff = datetime(2026, 8, 23, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for index in range(5):
        row = fixture_row(
            10_000 + index,
            cutoff - timedelta(days=index + 1),
            home_team_id=700 + index,
            away_team_id=100 if index < 2 else 200,
            home_goals=0,
            away_goals=index + 1,
            league_id=135,
        )
        row["home_team"] = f"Rakip {index}"
        row["away_team"] = "Juventus"
        rows.append(row)
    historical_repository.upsert_many(rows)

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=999,
        away_team_id=625,
        home_team_name="Frosinone",
        away_team_name="Juventus",
        league_id=135,
        before=cutoff,
        recent_match_count=5,
    )

    assert context.away_matches_df is not None
    assert len(context.away_matches_df) == 5
    assert context.away_matches_df["goals_for"].tolist() == [5, 4, 3, 2, 1]


def test_cross_competition_fallback_keeps_other_sides_league_resolution(
    historical_repository: HistoricalFixtureRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = datetime(2026, 8, 23, tzinfo=UTC)
    league_rows: list[dict[str, object]] = []
    for index in range(5):
        row = fixture_row(
            11_000 + index,
            cutoff - timedelta(days=30 + index),
            home_team_id=700 + index,
            away_team_id=900,
            league_id=140,
        )
        row["away_team"] = "Racing Santander"
        league_rows.append(row)

    cross_row = fixture_row(
        12_000,
        cutoff - timedelta(days=1),
        home_team_id=111,
        away_team_id=999,
        league_id=136,
    )
    cross_row["home_team"] = "Frosinone"
    cross_row["away_team"] = "Racing Santander"
    historical_repository.upsert_many([*league_rows, cross_row])
    stored_cross_row = historical_repository.get_by_fixture_id(12_000)
    assert stored_cross_row is not None
    monkeypatch.setattr(
        historical_repository,
        "get_recent_before",
        lambda **_kwargs: [stored_cross_row],
    )

    context = HistoricalFeatureService(historical_repository).build_context(
        home_team_id=101,
        away_team_id=202,
        home_team_name="Frosinone",
        away_team_name="Racing Santander",
        league_id=140,
        before=cutoff,
        recent_match_count=5,
    )

    assert context.home_matches_df is not None
    assert len(context.home_matches_df) == 1
    assert context.away_matches_df is not None
    assert len(context.away_matches_df) == 5


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

    class _FakeLock:
        acquired = True
        available = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    monkeypatch.setattr("app.tasks.fixtures_sync.ALLOWED_LEAGUE_IDS", {2, 3, 203, 848})
    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_HISTORICAL_SYNC_ENABLED", True)
    monkeypatch.setattr("app.tasks.fixtures_sync.APIFootballClient", FakeClient)
    monkeypatch.setattr("app.tasks.fixtures_sync.SessionLocal", lambda: Session(engine))
    monkeypatch.setattr(
        "app.tasks.fixtures_sync.DistributedTaskLock",
        lambda name, ttl_seconds=900, **kw: _FakeLock(),
    )

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

    class _FakeLock:
        acquired = True
        available = True

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    monkeypatch.setattr("app.tasks.fixtures_sync.ALLOWED_LEAGUE_IDS", {2, 3, 848})
    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_HISTORICAL_SYNC_ENABLED", True)
    monkeypatch.setattr(
        "app.tasks.fixtures_sync.DistributedTaskLock",
        lambda name, ttl_seconds=900, **kw: _FakeLock(),
    )

    with pytest.raises(ValueError, match=r"Unsupported league_ids: \[999999\]"):
        sync_historical_fixtures_task.run([2026], [999999])


def test_historical_sync_is_disabled_for_free_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.tasks import jobs

    monkeypatch.setattr(jobs.settings, "API_FOOTBALL_HISTORICAL_SYNC_ENABLED", False)

    assert sync_historical_fixtures_task.run([2026], [203]) == {
        "status": "disabled",
        "reason": "api_football_historical_sync_disabled",
    }


def test_football_data_sync_task_persists_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.football_data_csv import FootballDataImport

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

    monkeypatch.setattr("app.tasks.fixtures_sync.FootballDataCSVClient", FakeClient)
    monkeypatch.setattr("app.tasks.fixtures_sync.SessionLocal", lambda: Session(engine))

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

    monkeypatch.setattr(
        "app.tasks.fixtures_sync._current_football_season", lambda: 2026
    )
    monkeypatch.setattr("app.tasks.fixtures_sync.FootballDataCSVClient", FakeClient)
    monkeypatch.setattr("app.tasks.fixtures_sync.SessionLocal", lambda: Session(engine))

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


def test_missing_fixture_history_scopes_use_names_across_provider_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    def session_factory() -> Session:
        return Session(engine)

    with session_factory() as db:
        HistoricalFixtureRepository(db).upsert_many(
            [
                {
                    **fixture_row(
                        91,
                        datetime(2026, 5, 1, tzinfo=UTC),
                        home_team_id=11,
                        away_team_id=12,
                    ),
                    "home_team": "Galatasaray",
                    "away_team": "Çorum FK",
                }
            ]
        )

    monkeypatch.setattr("app.tasks._helpers.SessionLocal", session_factory)
    base = {
        "league_id": 203,
        "season": 2026,
        "kickoff": "2026-08-14T21:30:00+03:00",
        "home_team": "Galatasaray",
    }

    assert _missing_fixture_history_scopes([{**base, "away_team": "Çorum"}]) == []
    assert _missing_fixture_history_scopes([{**base, "away_team": "Yeni Kulüp"}]) == [
        (203, 2025),
        (203, 2026),
    ]


def test_recent_league_history_is_bounded_and_chronological(
    historical_repository: HistoricalFixtureRepository,
) -> None:
    kickoff = datetime(2026, 8, 1, 18, tzinfo=UTC)
    historical_repository.upsert_many(
        [
            fixture_row(950 + index, kickoff + timedelta(days=index))
            for index in range(3)
        ]
    )

    fixtures = historical_repository.get_recent_league_history(
        league_id=203,
        before=kickoff + timedelta(days=10),
        limit=2,
    )

    assert [fixture.fixture_id for fixture in fixtures] == [951, 952]


def test_historical_upsert_accepts_rows_with_different_optional_columns() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    first = fixture_row(301, datetime(2026, 1, 1, tzinfo=UTC))
    first["home_shots"] = 12
    second = fixture_row(302, datetime(2026, 1, 2, tzinfo=UTC))

    with Session(engine) as db:
        processed = HistoricalFixtureRepository(db).upsert_many([first, second])
        rows = db.query(HistoricalFixture).order_by(HistoricalFixture.fixture_id).all()

    assert processed == 2
    assert rows[0].home_shots == 12
    assert rows[1].home_shots is None


def test_missing_api_targets_only_include_insufficient_sides() -> None:
    from app.tasks._helpers import _missing_api_team_targets

    fixtures = [
        {
            "home_team_id": 994,
            "home_team": "Goztepe",
            "away_team_id": 611,
            "away_team": "Fenerbahce",
            "data_readiness": {
                "reasons": ["home_history_insufficient"],
            },
        },
        {
            # Aggregated provider IDs must never be sent to API-Football.
            "home_team_id": 500_000_001,
            "home_team": "Synthetic Team",
            "away_team_id": 611,
            "away_team": "Fenerbahce",
            "data_readiness": {
                "reasons": ["home_history_insufficient"],
            },
        },
    ]

    assert _missing_api_team_targets(fixtures) == [(994, "Goztepe")]


def test_missing_history_sync_classifies_unpublished_feed_as_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.football_data_csv import FootballDataDownloadError
    from app.tasks.jobs import sync_missing_fixture_history_task

    class FakeClient:
        supported_league_ids = frozenset({39})

        async def get_completed_fixtures(self, league_id: int, season: int):
            assert (league_id, season) == (39, 2026)
            raise FootballDataDownloadError("not published", status_code=404)

    class FakeLock:
        acquired = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeRepository:
        def __init__(self, _db) -> None:
            pass

        def upsert_many(self, rows: list[dict[str, object]]) -> int:
            assert rows == []
            return 0

    monkeypatch.setattr(
        "app.tasks.fixtures_sync._missing_fixture_history_scopes",
        lambda _fixtures: [(39, 2026)],
    )
    monkeypatch.setattr(
        "app.tasks.fixtures_sync._missing_fixture_team_targets",
        lambda _fixtures: [],
    )
    monkeypatch.setattr("app.tasks.fixtures_sync.FootballDataCSVClient", FakeClient)
    monkeypatch.setattr(
        "app.tasks.fixtures_sync.DistributedTaskLock",
        lambda *args, **kwargs: FakeLock(),
    )
    monkeypatch.setattr("app.tasks.fixtures_sync.SessionLocal", FakeSession)
    monkeypatch.setattr(
        "app.tasks.fixtures_sync.HistoricalFixtureRepository", FakeRepository
    )

    result = sync_missing_fixture_history_task.run([])

    assert result["status"] == "completed"
    assert result["failures"] == []
    assert result["pending_scopes"] == [
        {
            "league_id": 39,
            "season": 2026,
            "error": "FootballDataDownloadError",
        }
    ]
