from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import Base, HistoricalFixture
from app.db.player_context_repository import PlayerContextRepository
from app.prediction.ml.historical import HistoricalFeatureService


@pytest.fixture
def repositories() -> (
    Iterator[tuple[Session, HistoricalFixtureRepository, PlayerContextRepository]]
):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield (
            session,
            HistoricalFixtureRepository(session),
            PlayerContextRepository(session),
        )


def _add_fixture(
    session: Session,
    *,
    fixture_id: int,
    kickoff: datetime,
    league_id: int = 203,
    home_team_id: int = 1,
    away_team_id: int = 2,
) -> None:
    session.add(
        HistoricalFixture(
            fixture_id=fixture_id,
            league_id=league_id,
            season=2026,
            kickoff=kickoff,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_team=f"Team {home_team_id}",
            away_team=f"Team {away_team_id}",
            home_goals=1,
            away_goals=0,
            home_starting_xi=list(range(1, 12)),
            away_starting_xi=list(range(20, 31)),
            actual_result="HOME_WIN",
            status="FT",
            data_source="api_football",
        )
    )
    session.commit()


def _build_context(
    historical_repository: HistoricalFixtureRepository,
    player_repository: PlayerContextRepository | None,
    cutoff: datetime,
):
    return HistoricalFeatureService(
        historical_repository,
        player_repository,
    ).build_context(
        home_team_id=1,
        away_team_id=2,
        league_id=203,
        before=cutoff,
    )


def test_schedule_window_spans_competitions_and_excludes_cutoff_or_future(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    monkeypatch.setattr(settings, "FATIGUE_LOOKBACK_DAYS", 14)
    _add_fixture(session, fixture_id=100, kickoff=cutoff - timedelta(days=15))
    _add_fixture(session, fixture_id=101, kickoff=cutoff - timedelta(days=10))
    _add_fixture(
        session,
        fixture_id=102,
        kickoff=cutoff - timedelta(days=5),
        league_id=39,
        home_team_id=1,
        away_team_id=3,
    )
    _add_fixture(session, fixture_id=103, kickoff=cutoff)
    _add_fixture(session, fixture_id=104, kickoff=cutoff + timedelta(days=1))

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    assert context.home_schedule_df is not None
    assert context.home_schedule_df["fixture_id"].tolist() == [101, 102]
    assert context.home_schedule_df["league_id"].tolist() == [203, 39]
    assert context.away_schedule_df is not None
    assert context.away_schedule_df["fixture_id"].tolist() == [101]
    assert all(context.home_schedule_df["match_date"] < pd.Timestamp(cutoff))


def test_schedule_horizon_also_covers_ideal_rest_window(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    monkeypatch.setattr(settings, "FATIGUE_LOOKBACK_DAYS", 3)
    monkeypatch.setattr(settings, "FATIGUE_IDEAL_REST_DAYS", 7.0)
    _add_fixture(session, fixture_id=105, kickoff=cutoff - timedelta(days=5))

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    assert context.home_schedule_df is not None
    assert context.home_schedule_df["fixture_id"].tolist() == [105]


def test_player_ratings_use_only_last_observations_before_kickoff(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_LOOKBACK_MATCHES", 3)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_RATING_DECAY", 0.5)
    observations = [
        (201, cutoff - timedelta(days=8), 2.0),
        (202, cutoff - timedelta(days=6), 4.0),
        (203, cutoff - timedelta(days=4), 6.0),
        (204, cutoff - timedelta(days=2), 8.0),
        (205, cutoff, 10.0),
        (206, cutoff + timedelta(days=1), 12.0),
    ]
    for fixture_id, kickoff, _ in observations:
        _add_fixture(session, fixture_id=fixture_id, kickoff=kickoff)
    player_repository.upsert_performances(
        [
            {
                "fixture_id": fixture_id,
                "league_id": 203,
                "kickoff": kickoff,
                "team_id": 1,
                "player_id": 99,
                "started": True,
                "rating": rating,
                "source": "api_football",
            }
            for fixture_id, kickoff, rating in observations
        ]
    )

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    # Only 4, 6 and 8 are retained; newest observation has weight 1.
    expected = (4.0 * 0.25 + 6.0 * 0.5 + 8.0) / 1.75
    assert context.home_player_ratings == {
        99: {
            "minutes": 0.0,
            "appearances": 3.0,
            "goals": 0.0,
            "assists": 0.0,
            "rating": pytest.approx(expected, abs=1e-6),
        }
    }
    assert context.away_player_ratings == {}


def test_player_ratings_exclude_stale_transferred_players(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    monkeypatch.setattr(settings, "HISTORICAL_FORM_MAX_AGE_DAYS", 45)
    stale_kickoff = cutoff - timedelta(days=60)
    current_kickoff = cutoff - timedelta(days=2)
    _add_fixture(session, fixture_id=207, kickoff=stale_kickoff)
    _add_fixture(session, fixture_id=208, kickoff=current_kickoff)
    player_repository.upsert_performances(
        [
            {
                "fixture_id": 207,
                "league_id": 203,
                "kickoff": stale_kickoff,
                "team_id": 1,
                "player_id": 99,
                "started": True,
                "minutes": 900,
                "rating": 9.5,
            },
            {
                "fixture_id": 208,
                "league_id": 203,
                "kickoff": current_kickoff,
                "team_id": 1,
                "player_id": 10,
                "started": True,
                "minutes": 90,
                "rating": 7.0,
            },
        ]
    )

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    assert 99 not in context.home_player_ratings
    assert context.home_player_ratings[10]["rating"] == 7.0


def test_rating_aggregation_is_independent_per_player_and_ignores_invalid_values(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    kickoff = cutoff - timedelta(days=2)
    _add_fixture(session, fixture_id=301, kickoff=kickoff)
    player_repository.upsert_performances(
        [
            {
                "fixture_id": 301,
                "league_id": 203,
                "kickoff": kickoff,
                "team_id": 1,
                "player_id": 10,
                "started": True,
                "rating": 7.5,
            },
            {
                "fixture_id": 301,
                "league_id": 203,
                "kickoff": kickoff,
                "team_id": 1,
                "player_id": 11,
                "started": True,
                "rating": None,
            },
        ]
    )

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    assert context.home_player_ratings == {
        10: {
            "minutes": 0.0,
            "appearances": 1.0,
            "goals": 0.0,
            "assists": 0.0,
            "rating": 7.5,
        }
    }


def test_contribution_only_player_is_kept_when_rating_is_missing(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    kickoff = cutoff - timedelta(days=2)
    _add_fixture(session, fixture_id=302, kickoff=kickoff)
    player_repository.upsert_performances(
        [
            {
                "fixture_id": 302,
                "league_id": 203,
                "kickoff": kickoff,
                "team_id": 1,
                "player_id": 12,
                "started": True,
                "minutes": 90,
                "rating": None,
                "goals": 1,
                "assists": 1,
            }
        ]
    )

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    assert context.home_player_ratings == {
        12: {
            "minutes": 90.0,
            "appearances": 1.0,
            "goals": 1.0,
            "assists": 1.0,
        }
    }


def test_away_travel_distance_uses_team_locations(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    _add_fixture(session, fixture_id=401, kickoff=cutoff - timedelta(days=2))
    player_repository.upsert_team_locations(
        [
            {
                "data_source": "api_football",
                "team_id": 1,
                "name": "Istanbul",
                "latitude": 41.0082,
                "longitude": 28.9784,
            },
            {
                "data_source": "api_football",
                "team_id": 2,
                "name": "Ankara",
                "latitude": 39.9334,
                "longitude": 32.8597,
            },
        ]
    )

    context = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )

    assert context.away_travel_distance_km == pytest.approx(351.0, abs=2.0)


def test_optional_player_repository_and_missing_locations_are_neutral(
    repositories: tuple[
        Session,
        HistoricalFixtureRepository,
        PlayerContextRepository,
    ],
) -> None:
    session, historical_repository, player_repository = repositories
    cutoff = datetime(2026, 7, 20, 18, tzinfo=UTC)
    _add_fixture(session, fixture_id=501, kickoff=cutoff - timedelta(days=2))

    missing_location = _build_context(
        historical_repository,
        player_repository,
        cutoff,
    )
    no_player_repository = _build_context(
        historical_repository,
        None,
        cutoff,
    )

    assert missing_location.away_travel_distance_km == 0.0
    assert no_player_repository.away_travel_distance_km == 0.0
    assert no_player_repository.home_player_ratings == {}
    assert no_player_repository.away_player_ratings == {}
