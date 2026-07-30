from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.db.models import Base, HistoricalFixture
from app.db.player_context_repository import (
    PlayerContextRepository,
    haversine_distance_km,
)


def build_repository() -> tuple[Session, PlayerContextRepository]:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = Session(engine)
    return session, PlayerContextRepository(session)


def add_fixture(
    session: Session,
    *,
    fixture_id: int,
    kickoff: datetime,
    league_id: int = 203,
) -> None:
    session.add(
        HistoricalFixture(
            fixture_id=fixture_id,
            league_id=league_id,
            season=2026,
            kickoff=kickoff,
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
    session.commit()


def performance_row(
    fixture_id: int,
    kickoff: datetime,
    *,
    player_id: int = 10,
    team_id: int = 1,
    league_id: int = 203,
    rating: float | None = 7.0,
) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "league_id": league_id,
        "kickoff": kickoff,
        "team_id": team_id,
        "player_id": player_id,
        "started": True,
        "minutes": 90,
        "rating": rating,
        "position": "Midfielder",
        "goals": 0,
        "assists": 1,
        "source": "api_football",
    }


def test_sqlite_performance_upsert_is_idempotent_and_updates_mutable_fields() -> None:
    session, repository = build_repository()
    kickoff = datetime(2026, 7, 1, 18, tzinfo=UTC)
    add_fixture(session, fixture_id=1001, kickoff=kickoff)
    try:
        created_count = repository.upsert_performances(
            [performance_row(1001, kickoff, rating=6.8)]
        )
        created = repository.get_all_performances()[0]
        ingested_at = created.ingested_at

        updated_count = repository.upsert_performances(
            [
                {
                    **performance_row(1001, kickoff, rating=8.1),
                    "minutes": 75,
                    "goals": 2,
                }
            ]
        )
        session.expire_all()
        persisted = repository.get_all_performances()

        assert created_count == updated_count == 1
        assert len(persisted) == 1
        assert persisted[0].id == created.id
        assert persisted[0].rating == 8.1
        assert persisted[0].minutes == 75
        assert persisted[0].goals == 2
        assert persisted[0].ingested_at == ingested_at
        assert persisted[0].updated_at >= ingested_at
    finally:
        session.close()


def test_performance_upsert_deduplicates_batch_by_fixture_and_player() -> None:
    session, repository = build_repository()
    kickoff = datetime(2026, 7, 1, 18, tzinfo=UTC)
    add_fixture(session, fixture_id=1002, kickoff=kickoff)
    try:
        processed = repository.upsert_performances(
            [
                performance_row(1002, kickoff, rating=6.0),
                performance_row(1002, kickoff, rating=7.5),
            ]
        )

        assert processed == 1
        assert [row.rating for row in repository.get_all_performances()] == [7.5]
    finally:
        session.close()


def test_complete_context_requires_minimum_player_coverage_for_both_teams() -> None:
    session, repository = build_repository()
    kickoff = datetime(2026, 7, 1, 18, tzinfo=UTC)
    add_fixture(session, fixture_id=1003, kickoff=kickoff)
    add_fixture(session, fixture_id=1004, kickoff=kickoff + timedelta(days=1))
    add_fixture(session, fixture_id=1005, kickoff=kickoff + timedelta(days=2))
    try:
        repository.upsert_performances(
            [
                performance_row(1003, kickoff, player_id=10),
                *[
                    performance_row(
                        1004,
                        kickoff + timedelta(days=1),
                        player_id=player_id,
                        team_id=team_id,
                    )
                    for team_id, player_ids in ((1, range(1, 8)), (2, range(20, 27)))
                    for player_id in player_ids
                ],
                *[
                    performance_row(
                        1005,
                        kickoff + timedelta(days=2),
                        player_id=player_id,
                        team_id=1,
                    )
                    for player_id in range(30, 44)
                ],
            ]
        )

        assert repository.get_fixture_ids_with_complete_player_context(
            [1003, 1004, 1005]
        ) == {1004}
        assert repository.get_fixture_ids_with_complete_player_context([]) == set()
    finally:
        session.close()


def test_performance_queries_enforce_strict_point_in_time_cutoff() -> None:
    session, repository = build_repository()
    cutoff = datetime(2026, 7, 10, 18, tzinfo=UTC)
    kickoffs = [cutoff - timedelta(days=2), cutoff, cutoff + timedelta(days=2)]
    for fixture_id, kickoff in zip((1101, 1102, 1103), kickoffs):
        add_fixture(session, fixture_id=fixture_id, kickoff=kickoff)
    try:
        repository.upsert_performances(
            [
                performance_row(fixture_id, kickoff, rating=rating)
                for fixture_id, kickoff, rating in zip(
                    (1101, 1102, 1103),
                    kickoffs,
                    (6.5, 7.0, 8.0),
                )
            ]
        )

        team_rows = repository.get_team_performances_before(1, cutoff)
        player_rows = repository.get_player_performances_before(10, cutoff)

        assert [row.fixture_id for row in team_rows] == [1101]
        assert [row.fixture_id for row in player_rows] == [1101]
        assert repository.get_team_performances_before(1, cutoff, limit=1) == team_rows
    finally:
        session.close()


def test_performance_rows_are_deleted_with_their_fixture() -> None:
    session, repository = build_repository()
    kickoff = datetime(2026, 7, 1, 18, tzinfo=UTC)
    add_fixture(session, fixture_id=1201, kickoff=kickoff)
    try:
        repository.upsert_performances([performance_row(1201, kickoff)])
        fixture = (
            session.query(HistoricalFixture)
            .filter(HistoricalFixture.fixture_id == 1201)
            .one()
        )
        session.delete(fixture)
        session.commit()

        assert repository.get_all_performances() == []
    finally:
        session.close()


def test_sqlite_location_upsert_is_scoped_by_source_and_team() -> None:
    session, repository = build_repository()
    try:
        assert (
            repository.upsert_team_locations(
                [
                    {
                        "data_source": "api_football",
                        "team_id": 1,
                        "name": "Old Name",
                        "latitude": 41.0082,
                        "longitude": 28.9784,
                    }
                ]
            )
            == 1
        )
        original = repository.get_team_location(1)
        assert original is not None
        original_id = original.id

        repository.upsert_team_locations(
            [
                {
                    "data_source": "api_football",
                    "team_id": 1,
                    "name": "Istanbul FC",
                    "latitude": 41.01,
                    "longitude": 28.98,
                },
                {
                    "data_source": "manual",
                    "team_id": 1,
                    "name": "Manual Istanbul FC",
                    "latitude": None,
                    "longitude": None,
                },
            ]
        )
        session.expire_all()

        updated = repository.get_team_location(1)
        manual = repository.get_team_location(1, data_source="manual")
        assert updated is not None
        assert manual is not None
        assert updated.id == original_id
        assert updated.name == "Istanbul FC"
        assert updated.location_source == "manual"
        assert updated.confidence == pytest.approx(1.0)
        assert updated.details is None
        assert len(repository.get_all_team_locations()) == 2
    finally:
        session.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", -90.1),
        ("latitude", 90.1),
        ("latitude", float("nan")),
        ("longitude", -180.1),
        ("longitude", 180.1),
        ("longitude", float("inf")),
        ("longitude", True),
    ],
)
def test_location_upsert_rejects_invalid_coordinates(field: str, value: object) -> None:
    session, repository = build_repository()
    row: dict[str, object] = {
        "data_source": "api_football",
        "team_id": 1,
        "name": "Invalid FC",
        "latitude": 41.0,
        "longitude": 29.0,
    }
    row[field] = value
    try:
        with pytest.raises(ValueError, match=field):
            repository.upsert_team_locations([row])
        assert repository.get_all_team_locations() == []
    finally:
        session.close()


def test_haversine_and_repository_distance_use_kilometres() -> None:
    session, repository = build_repository()
    try:
        repository.upsert_team_locations(
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

        expected = haversine_distance_km(41.0082, 28.9784, 39.9334, 32.8597)
        assert expected == pytest.approx(351.0, abs=2.0)
        assert repository.travel_distance_km(1, 2) == pytest.approx(expected)
        assert repository.travel_distance_km(1, 1) == pytest.approx(0.0)
    finally:
        session.close()


def test_missing_location_or_coordinate_has_neutral_zero_distance() -> None:
    session, repository = build_repository()
    try:
        repository.upsert_team_locations(
            [
                {
                    "data_source": "api_football",
                    "team_id": 1,
                    "name": "Unknown Coordinates",
                    "latitude": None,
                    "longitude": None,
                },
                {
                    "data_source": "api_football",
                    "team_id": 2,
                    "name": "Known Coordinates",
                    "latitude": 39.9334,
                    "longitude": 32.8597,
                },
            ]
        )

        assert repository.travel_distance_km(1, 2) == 0.0
        assert repository.travel_distance_km(2, 999) == 0.0
        assert haversine_distance_km(None, None, 39.9334, 32.8597) == 0.0
    finally:
        session.close()
