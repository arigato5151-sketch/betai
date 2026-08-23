from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from sqlalchemy import or_, select, union_all
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.db.models import (
    HistoricalFixture,
    HistoricalPlayerPerformance,
    TeamLocation,
    utc_now,
)

EARTH_RADIUS_KM = 6371.0088
MINIMUM_PLAYER_CONTEXT_PER_TEAM = 7


def _has_minimum_team_coverage(
    player_counts: Mapping[int, int],
    *,
    home_team_id: int,
    away_team_id: int,
    minimum_players_per_team: int,
) -> bool:
    return (
        home_team_id != away_team_id
        and player_counts.get(home_team_id, 0) >= minimum_players_per_team
        and player_counts.get(away_team_id, 0) >= minimum_players_per_team
    )


def is_fixture_player_context_complete(
    performances: Iterable[Mapping[str, object]],
    *,
    home_team_id: int,
    away_team_id: int,
    minimum_players_per_team: int = MINIMUM_PLAYER_CONTEXT_PER_TEAM,
) -> bool:
    """Require useful, distinct player coverage for both fixture teams."""
    if minimum_players_per_team < 1:
        raise ValueError("minimum_players_per_team must be positive")

    players_by_team: dict[int, set[int]] = {
        home_team_id: set(),
        away_team_id: set(),
    }
    for performance in performances:
        team_id = performance.get("team_id")
        player_id = performance.get("player_id")
        if (
            isinstance(team_id, int)
            and not isinstance(team_id, bool)
            and team_id in players_by_team
            and isinstance(player_id, int)
            and not isinstance(player_id, bool)
            and player_id > 0
        ):
            players_by_team[team_id].add(player_id)

    return _has_minimum_team_coverage(
        {team_id: len(player_ids) for team_id, player_ids in players_by_team.items()},
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        minimum_players_per_team=minimum_players_per_team,
    )


def haversine_distance_km(
    latitude_a: float | None,
    longitude_a: float | None,
    latitude_b: float | None,
    longitude_b: float | None,
) -> float:
    """Return great-circle distance, or neutral zero when coordinates are missing."""
    if None in (latitude_a, longitude_a, latitude_b, longitude_b):
        return 0.0

    lat_a = _validated_coordinate(latitude_a, "latitude", -90.0, 90.0)
    lon_a = _validated_coordinate(longitude_a, "longitude", -180.0, 180.0)
    lat_b = _validated_coordinate(latitude_b, "latitude", -90.0, 90.0)
    lon_b = _validated_coordinate(longitude_b, "longitude", -180.0, 180.0)

    lat_a_rad, lon_a_rad, lat_b_rad, lon_b_rad = map(
        math.radians,
        (lat_a, lon_a, lat_b, lon_b),
    )
    delta_lat = lat_b_rad - lat_a_rad
    delta_lon = lon_b_rad - lon_a_rad
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(delta_lon / 2.0) ** 2
    )
    # Floating point rounding can otherwise push the square-root input over one.
    central_angle = 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))
    return EARTH_RADIUS_KM * central_angle


def _validated_coordinate(
    value: object,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}")
    return numeric


def _positive_identifier(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a positive integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if numeric <= 0 or numeric != value:
        raise ValueError(f"{label} must be a positive integer")
    return numeric


def _nonzero_identifier(value: object, label: str) -> int:
    """Accept provider IDs and deterministic negative IDs used by open feeds."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a non-zero integer")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-zero integer") from exc
    if numeric == 0 or numeric != value:
        raise ValueError(f"{label} must be a non-zero integer")
    return numeric


class PlayerContextRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert_performances(
        self,
        performances: Iterable[Mapping[str, object]],
    ) -> int:
        timestamp = utc_now()
        rows_by_key: dict[tuple[int, int], dict[str, object]] = {}
        allowed_columns = {
            column.name
            for column in HistoricalPlayerPerformance.__table__.columns
            if column.name != "id"
        }

        for raw_row in performances:
            row = dict(raw_row)
            self._reject_unknown_columns(
                row,
                allowed_columns,
                entity="historical player performance",
            )
            fixture_id = _positive_identifier(row.get("fixture_id"), "fixture_id")
            player_id = _positive_identifier(row.get("player_id"), "player_id")
            row["fixture_id"] = fixture_id
            row["league_id"] = _positive_identifier(row.get("league_id"), "league_id")
            row["team_id"] = _positive_identifier(row.get("team_id"), "team_id")
            row["player_id"] = player_id
            if not isinstance(row.get("kickoff"), datetime):
                raise ValueError("kickoff must be a datetime")
            if "started" in row and not isinstance(row["started"], bool):
                raise ValueError("started must be a boolean")
            row.setdefault("started", False)
            for optional_field in (
                "minutes",
                "rating",
                "position",
                "goals",
                "assists",
            ):
                row.setdefault(optional_field, None)
            source = str(row.get("source") or "api_football").strip()
            if not source:
                raise ValueError("source cannot be blank")
            row["source"] = source[:50]
            if row.get("position") is not None:
                row["position"] = str(row["position"]).strip()[:20] or None
            row["ingested_at"] = row.get("ingested_at") or timestamp
            row["updated_at"] = timestamp
            rows_by_key[(fixture_id, player_id)] = row

        rows = list(rows_by_key.values())
        if not rows:
            return 0
        self._upsert_rows(
            HistoricalPlayerPerformance,
            rows,
            conflict_columns=("fixture_id", "player_id"),
        )
        return len(rows)

    def get_team_performances_before(
        self,
        team_id: int,
        before: datetime,
        *,
        league_id: int | None = None,
        limit: int | None = None,
    ) -> list[HistoricalPlayerPerformance]:
        team_id = _positive_identifier(team_id, "team_id")
        if not isinstance(before, datetime):
            raise ValueError("before must be a datetime")
        query = self.db.query(HistoricalPlayerPerformance).filter(
            HistoricalPlayerPerformance.team_id == team_id,
            HistoricalPlayerPerformance.kickoff < before,
        )
        if league_id is not None:
            query = query.filter(
                HistoricalPlayerPerformance.league_id
                == _positive_identifier(league_id, "league_id")
            )
        query = query.order_by(
            HistoricalPlayerPerformance.kickoff.desc(),
            HistoricalPlayerPerformance.fixture_id.desc(),
            HistoricalPlayerPerformance.player_id.asc(),
        )
        if limit is not None:
            query = query.limit(self._validated_limit(limit))
        return query.all()

    def get_player_performances_before(
        self,
        player_id: int,
        before: datetime,
        *,
        limit: int | None = None,
    ) -> list[HistoricalPlayerPerformance]:
        player_id = _positive_identifier(player_id, "player_id")
        if not isinstance(before, datetime):
            raise ValueError("before must be a datetime")
        query = (
            self.db.query(HistoricalPlayerPerformance)
            .filter(
                HistoricalPlayerPerformance.player_id == player_id,
                HistoricalPlayerPerformance.kickoff < before,
            )
            .order_by(
                HistoricalPlayerPerformance.kickoff.desc(),
                HistoricalPlayerPerformance.fixture_id.desc(),
            )
        )
        if limit is not None:
            query = query.limit(self._validated_limit(limit))
        return query.all()

    def get_all_performances(self) -> list[HistoricalPlayerPerformance]:
        return (
            self.db.query(HistoricalPlayerPerformance)
            .order_by(
                HistoricalPlayerPerformance.kickoff.asc(),
                HistoricalPlayerPerformance.fixture_id.asc(),
                HistoricalPlayerPerformance.player_id.asc(),
            )
            .all()
        )

    def get_performances_for_fixture_ids(
        self,
        fixture_ids: Iterable[int],
        *,
        max_results: int,
    ) -> list[HistoricalPlayerPerformance]:
        """Load selected-fixture context without exceeding SQL bind or RAM limits."""
        normalized_ids = list(
            dict.fromkeys(
                _positive_identifier(fixture_id, "fixture_id")
                for fixture_id in fixture_ids
            )
        )
        max_results = self._validated_limit(max_results)
        if not normalized_ids:
            return []

        results: list[HistoricalPlayerPerformance] = []
        # SQLite permits only 999 bind variables; keeping chunks below that also
        # works unchanged on PostgreSQL and bounds each ORM result set.
        for offset in range(0, len(normalized_ids), 500):
            remaining = max_results - len(results)
            if remaining <= 0:
                break
            batch = normalized_ids[offset : offset + 500]
            results.extend(
                self.db.query(HistoricalPlayerPerformance)
                .filter(HistoricalPlayerPerformance.fixture_id.in_(batch))
                .order_by(
                    HistoricalPlayerPerformance.kickoff.asc(),
                    HistoricalPlayerPerformance.fixture_id.asc(),
                    HistoricalPlayerPerformance.player_id.asc(),
                )
                .limit(remaining)
                .all()
            )
        return sorted(
            results,
            key=lambda performance: (
                performance.kickoff,
                performance.fixture_id,
                performance.player_id,
            ),
        )

    def get_fixture_ids_with_complete_player_context(
        self,
        fixture_ids: Iterable[int],
        *,
        minimum_players_per_team: int = MINIMUM_PLAYER_CONTEXT_PER_TEAM,
    ) -> set[int]:
        normalized_ids = {
            _positive_identifier(fixture_id, "fixture_id") for fixture_id in fixture_ids
        }
        if not normalized_ids:
            return set()
        minimum_players_per_team = _positive_identifier(
            minimum_players_per_team,
            "minimum_players_per_team",
        )
        fixtures = (
            self.db.query(
                HistoricalFixture.fixture_id,
                HistoricalFixture.home_team_id,
                HistoricalFixture.away_team_id,
            )
            .filter(HistoricalFixture.fixture_id.in_(normalized_ids))
            .all()
        )
        coverage_rows = (
            self.db.query(
                HistoricalPlayerPerformance.fixture_id,
                HistoricalPlayerPerformance.team_id,
                func.count(func.distinct(HistoricalPlayerPerformance.player_id)),
            )
            .filter(HistoricalPlayerPerformance.fixture_id.in_(normalized_ids))
            .group_by(
                HistoricalPlayerPerformance.fixture_id,
                HistoricalPlayerPerformance.team_id,
            )
            .all()
        )
        coverage_by_fixture: dict[int, dict[int, int]] = {}
        for fixture_id, team_id, player_count in coverage_rows:
            coverage_by_fixture.setdefault(int(fixture_id), {})[int(team_id)] = int(
                player_count
            )

        return {
            int(fixture_id)
            for fixture_id, home_team_id, away_team_id in fixtures
            if _has_minimum_team_coverage(
                coverage_by_fixture.get(int(fixture_id), {}),
                home_team_id=int(home_team_id),
                away_team_id=int(away_team_id),
                minimum_players_per_team=minimum_players_per_team,
            )
        }

    def upsert_team_locations(
        self,
        locations: Iterable[Mapping[str, object]],
    ) -> int:
        timestamp = utc_now()
        rows_by_key: dict[tuple[str, int], dict[str, object]] = {}
        allowed_columns = {
            column.name
            for column in TeamLocation.__table__.columns
            if column.name != "id"
        }

        for raw_row in locations:
            row = dict(raw_row)
            self._reject_unknown_columns(row, allowed_columns, entity="team location")
            data_source = str(row.get("data_source") or "").strip().lower()
            if not data_source:
                raise ValueError("data_source cannot be blank")
            team_id = _nonzero_identifier(row.get("team_id"), "team_id")
            name = str(row.get("name") or "").strip()
            if not name:
                raise ValueError("name cannot be blank")
            latitude = row.get("latitude")
            longitude = row.get("longitude")
            row["data_source"] = data_source[:50]
            row["team_id"] = team_id
            row["name"] = name[:100]
            row["latitude"] = (
                None
                if latitude is None
                else _validated_coordinate(latitude, "latitude", -90.0, 90.0)
            )
            row["longitude"] = (
                None
                if longitude is None
                else _validated_coordinate(longitude, "longitude", -180.0, 180.0)
            )
            location_source = str(row.get("location_source") or "manual").strip()
            if not location_source:
                raise ValueError("location_source cannot be blank")
            confidence = row.get("confidence", 1.0)
            if isinstance(confidence, bool):
                raise ValueError("confidence must be between 0 and 1")
            try:
                numeric_confidence = float(str(confidence))
            except (TypeError, ValueError) as exc:
                raise ValueError("confidence must be between 0 and 1") from exc
            if (
                not math.isfinite(numeric_confidence)
                or not 0.0 <= numeric_confidence <= 1.0
            ):
                raise ValueError("confidence must be between 0 and 1")
            details = row.get("details")
            if details is not None and not isinstance(details, Mapping):
                raise ValueError("details must be an object")
            row["location_source"] = location_source[:50]
            row["confidence"] = numeric_confidence
            row["details"] = dict(details) if isinstance(details, Mapping) else None
            row["ingested_at"] = row.get("ingested_at") or timestamp
            row["updated_at"] = timestamp
            rows_by_key[(data_source[:50], team_id)] = row

        rows = list(rows_by_key.values())
        if not rows:
            return 0
        self._upsert_rows(
            TeamLocation,
            rows,
            conflict_columns=("data_source", "team_id"),
        )
        return len(rows)

    def get_team_location(
        self,
        team_id: int,
        *,
        data_source: str = "api_football",
    ) -> TeamLocation | None:
        team_id = _nonzero_identifier(team_id, "team_id")
        normalized_source = data_source.strip().lower()
        if not normalized_source:
            raise ValueError("data_source cannot be blank")
        return (
            self.db.query(TeamLocation)
            .filter(
                TeamLocation.data_source == normalized_source,
                TeamLocation.team_id == team_id,
            )
            .one_or_none()
        )

    def get_all_team_locations(self) -> list[TeamLocation]:
        return (
            self.db.query(TeamLocation)
            .order_by(TeamLocation.data_source.asc(), TeamLocation.team_id.asc())
            .all()
        )

    def list_missing_team_location_targets(
        self,
        *,
        seasons: Iterable[int],
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        """Return one recent identity per team lacking usable coordinates."""
        normalized_seasons = sorted(
            {_positive_identifier(season, "season") for season in seasons}
        )
        limit = self._validated_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if not normalized_seasons:
            return []

        home = select(
            HistoricalFixture.data_source.label("data_source"),
            HistoricalFixture.home_team_id.label("team_id"),
            HistoricalFixture.home_team.label("team_name"),
            HistoricalFixture.league_id.label("league_id"),
            HistoricalFixture.kickoff.label("kickoff"),
            HistoricalFixture.fixture_id.label("fixture_id"),
        ).where(HistoricalFixture.season.in_(normalized_seasons))
        away = select(
            HistoricalFixture.data_source.label("data_source"),
            HistoricalFixture.away_team_id.label("team_id"),
            HistoricalFixture.away_team.label("team_name"),
            HistoricalFixture.league_id.label("league_id"),
            HistoricalFixture.kickoff.label("kickoff"),
            HistoricalFixture.fixture_id.label("fixture_id"),
        ).where(HistoricalFixture.season.in_(normalized_seasons))
        fixture_teams = union_all(home, away).subquery()
        ranked = select(
            fixture_teams,
            func.row_number()
            .over(
                partition_by=(fixture_teams.c.data_source, fixture_teams.c.team_id),
                order_by=(
                    fixture_teams.c.kickoff.desc(),
                    fixture_teams.c.fixture_id.desc(),
                ),
            )
            .label("team_rank"),
        ).subquery()
        rows = self.db.execute(
            select(
                ranked.c.data_source,
                ranked.c.team_id,
                ranked.c.team_name,
                ranked.c.league_id,
            )
            .outerjoin(
                TeamLocation,
                (TeamLocation.data_source == ranked.c.data_source)
                & (TeamLocation.team_id == ranked.c.team_id),
            )
            .where(
                ranked.c.team_rank == 1,
                or_(
                    TeamLocation.id.is_(None),
                    TeamLocation.latitude.is_(None),
                    TeamLocation.longitude.is_(None),
                ),
            )
            .order_by(ranked.c.data_source.asc(), ranked.c.team_name.asc())
            .offset(offset)
            .limit(limit)
        ).all()
        return [
            {
                "data_source": str(data_source),
                "team_id": int(team_id),
                "name": str(team_name),
                "league_id": int(league_id),
            }
            for data_source, team_id, team_name, league_id in rows
        ]

    def list_team_locations(
        self,
        *,
        data_source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TeamLocation]:
        """Return a bounded deterministic page for administrative inspection."""
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        query = self.db.query(TeamLocation)
        if data_source is not None:
            normalized_source = data_source.strip().lower()
            if not normalized_source:
                raise ValueError("data_source cannot be blank")
            query = query.filter(TeamLocation.data_source == normalized_source)
        return (
            query.order_by(TeamLocation.data_source.asc(), TeamLocation.team_id.asc())
            .offset(offset)
            .limit(self._validated_limit(limit))
            .all()
        )

    def travel_distance_km(
        self,
        origin_team_id: int,
        destination_team_id: int,
        *,
        data_source: str = "api_football",
    ) -> float:
        origin = self.get_team_location(origin_team_id, data_source=data_source)
        destination = self.get_team_location(
            destination_team_id,
            data_source=data_source,
        )
        if origin is None or destination is None:
            return 0.0
        return haversine_distance_km(
            origin.latitude,
            origin.longitude,
            destination.latitude,
            destination.longitude,
        )

    @staticmethod
    def _validated_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return limit

    @staticmethod
    def _reject_unknown_columns(
        row: Mapping[str, object],
        allowed_columns: set[str],
        *,
        entity: str,
    ) -> None:
        unknown = sorted(set(row) - allowed_columns)
        if unknown:
            raise ValueError(f"Unknown {entity} fields: {', '.join(unknown)}")

    def _upsert_rows(
        self,
        model: type[HistoricalPlayerPerformance] | type[TeamLocation],
        rows: Sequence[dict[str, object]],
        *,
        conflict_columns: tuple[str, ...],
    ) -> None:
        if self.db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")

        dialect = self.db.bind.dialect.name
        mutable_columns = {
            column.name
            for column in model.__table__.columns
            if column.name not in {"id", "ingested_at", *conflict_columns}
        }
        try:
            if dialect == "postgresql":
                pg_statement = pg_insert(model).values(rows)
                self.db.execute(
                    pg_statement.on_conflict_do_update(
                        index_elements=list(conflict_columns),
                        set_={
                            name: getattr(pg_statement.excluded, name)
                            for name in mutable_columns
                        },
                    )
                )
            elif dialect == "sqlite":
                sqlite_statement = sqlite_insert(model).values(rows)
                self.db.execute(
                    sqlite_statement.on_conflict_do_update(
                        index_elements=list(conflict_columns),
                        set_={
                            name: getattr(sqlite_statement.excluded, name)
                            for name in mutable_columns
                        },
                    )
                )
            else:
                for row in rows:
                    filters = [
                        getattr(model, column) == row[column]
                        for column in conflict_columns
                    ]
                    existing = self.db.query(model).filter(*filters).one_or_none()
                    if existing is None:
                        self.db.add(model(**row))
                        continue
                    for name in mutable_columns:
                        if name in row:
                            setattr(existing, name, row[name])
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise
