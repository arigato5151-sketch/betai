from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Mapping

from sqlalchemy import and_, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import HistoricalFixture, utc_now


class HistoricalFixtureRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert_many(self, fixtures: Iterable[dict]) -> int:
        # Keep the last API representation if a response unexpectedly repeats an ID.
        rows_by_fixture_id = {row["fixture_id"]: row for row in fixtures}
        rows = list(rows_by_fixture_id.values())
        if not rows:
            return 0
        if self.db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")

        timestamp = utc_now()
        normalized = [
            {
                **row,
                "ingested_at": row.get("ingested_at", timestamp),
                "updated_at": timestamp,
            }
            for row in rows
        ]
        dialect = self.db.bind.dialect.name

        try:
            if dialect == "postgresql":
                statement = pg_insert(HistoricalFixture).values(normalized)
                mutable_columns = {
                    column.name: getattr(statement.excluded, column.name)
                    for column in HistoricalFixture.__table__.columns
                    if column.name not in {"id", "fixture_id", "ingested_at"}
                }
                self.db.execute(
                    statement.on_conflict_do_update(
                        index_elements=["fixture_id"], set_=mutable_columns
                    )
                )
            elif dialect == "sqlite":
                sqlite_statement = sqlite_insert(HistoricalFixture).values(normalized)
                sqlite_mutable_columns = {
                    column.name: getattr(sqlite_statement.excluded, column.name)
                    for column in HistoricalFixture.__table__.columns
                    if column.name not in {"id", "fixture_id", "ingested_at"}
                }
                self.db.execute(
                    sqlite_statement.on_conflict_do_update(
                        index_elements=["fixture_id"], set_=sqlite_mutable_columns
                    )
                )
            else:
                for row in normalized:
                    existing = self.get_by_fixture_id(row["fixture_id"])
                    if existing is None:
                        self.db.add(HistoricalFixture(**row))
                    else:
                        for key, value in row.items():
                            if key != "ingested_at":
                                setattr(existing, key, value)
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise
        return len(normalized)

    def get_by_fixture_id(self, fixture_id: int) -> HistoricalFixture | None:
        return (
            self.db.query(HistoricalFixture)
            .filter(HistoricalFixture.fixture_id == fixture_id)
            .first()
        )

    def get_all(self) -> list[HistoricalFixture]:
        return (
            self.db.query(HistoricalFixture)
            .order_by(
                HistoricalFixture.kickoff.asc(),
                HistoricalFixture.fixture_id.asc(),
            )
            .all()
        )

    def update_xg_many(self, updates: Iterable[Mapping[str, object]]) -> int:
        """Apply validated provider xG updates in one transaction."""
        rows_by_fixture_id: dict[int, tuple[float, float, str, str, float]] = {}
        for row in updates:
            fixture_id = row.get("fixture_id")
            if (
                isinstance(fixture_id, bool)
                or not isinstance(fixture_id, int)
                or fixture_id == 0
            ):
                raise ValueError("fixture_id must be a non-zero integer")
            source = str(row.get("xg_source") or "").strip()
            provider_match_id = str(row.get("xg_provider_match_id") or "").strip()
            if not source or not provider_match_id:
                raise ValueError("xG provenance cannot be blank")
            rows_by_fixture_id[fixture_id] = (
                self._validated_xg(row.get("home_xg"), "home_xg"),
                self._validated_xg(row.get("away_xg"), "away_xg"),
                source[:50],
                provider_match_id[:100],
                self._validated_confidence(row.get("xg_confidence")),
            )
        if not rows_by_fixture_id:
            return 0
        fixtures = (
            self.db.query(HistoricalFixture)
            .filter(HistoricalFixture.fixture_id.in_(rows_by_fixture_id))
            .all()
        )
        timestamp = utc_now()
        try:
            for fixture in fixtures:
                home_xg, away_xg, source, provider_match_id, confidence = (
                    rows_by_fixture_id[fixture.fixture_id]
                )
                fixture.home_xg = home_xg
                fixture.away_xg = away_xg
                fixture.xg_source = source
                fixture.xg_provider_match_id = provider_match_id
                fixture.xg_updated_at = timestamp
                fixture.xg_confidence = confidence
            self.db.commit()
        except (KeyError, TypeError, ValueError, SQLAlchemyError):
            self.db.rollback()
            raise
        return len(fixtures)

    @staticmethod
    def _validated_xg(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 15.0:
            raise ValueError(f"{label} must be between 0 and 15")
        return parsed

    @staticmethod
    def _validated_confidence(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("xg_confidence must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
            raise ValueError("xg_confidence must be between 0 and 1")
        return parsed

    def get_league_history(
        self, *, league_id: int, before: datetime, season: int | None = None
    ) -> list[HistoricalFixture]:
        query = self.db.query(HistoricalFixture).filter(
            HistoricalFixture.league_id == league_id,
            HistoricalFixture.kickoff < before,
        )
        if season is not None:
            query = query.filter(HistoricalFixture.season == season)
        return query.order_by(
            HistoricalFixture.kickoff.asc(), HistoricalFixture.fixture_id.asc()
        ).all()

    def get_h2h(
        self, *, home_team_id: int, away_team_id: int, before: datetime, limit: int = 10
    ) -> list[HistoricalFixture]:
        pair = or_(
            and_(
                HistoricalFixture.home_team_id == home_team_id,
                HistoricalFixture.away_team_id == away_team_id,
            ),
            and_(
                HistoricalFixture.home_team_id == away_team_id,
                HistoricalFixture.away_team_id == home_team_id,
            ),
        )
        return (
            self.db.query(HistoricalFixture)
            .filter(pair, HistoricalFixture.kickoff < before)
            .order_by(HistoricalFixture.kickoff.desc())
            .limit(limit)
            .all()
        )

    def get_team_history(
        self,
        *,
        team_id: int,
        league_id: int,
        before: datetime,
        limit: int = 5,
    ) -> list[HistoricalFixture]:
        return (
            self.db.query(HistoricalFixture)
            .filter(
                HistoricalFixture.league_id == league_id,
                or_(
                    HistoricalFixture.home_team_id == team_id,
                    HistoricalFixture.away_team_id == team_id,
                ),
                HistoricalFixture.kickoff < before,
            )
            .order_by(HistoricalFixture.kickoff.desc())
            .limit(limit)
            .all()
        )

    def get_team_schedule(
        self,
        *,
        team_id: int,
        since: datetime,
        before: datetime,
    ) -> list[HistoricalFixture]:
        """Return the strict point-in-time schedule across all competitions."""
        return (
            self.db.query(HistoricalFixture)
            .filter(
                or_(
                    HistoricalFixture.home_team_id == team_id,
                    HistoricalFixture.away_team_id == team_id,
                ),
                HistoricalFixture.kickoff >= since,
                HistoricalFixture.kickoff < before,
            )
            .order_by(
                HistoricalFixture.kickoff.asc(),
                HistoricalFixture.fixture_id.asc(),
            )
            .all()
        )

    def get_last_starting_xi(
        self,
        *,
        team_id: int,
        before: datetime,
        league_id: int | None = None,
    ) -> list[int] | None:
        query = self.db.query(HistoricalFixture).filter(
            or_(
                HistoricalFixture.home_team_id == team_id,
                HistoricalFixture.away_team_id == team_id,
            ),
            HistoricalFixture.kickoff < before,
        )
        if league_id is not None:
            query = query.filter(HistoricalFixture.league_id == league_id)
        fixtures = query.order_by(HistoricalFixture.kickoff.desc()).limit(50).all()
        for fixture in fixtures:
            lineup = (
                fixture.home_starting_xi
                if fixture.home_team_id == team_id
                else fixture.away_starting_xi
            )
            if isinstance(lineup, list):
                player_ids = [
                    player_id
                    for player_id in lineup
                    if isinstance(player_id, int) and player_id > 0
                ]
                if len(set(player_ids)) == 11:
                    return list(dict.fromkeys(player_ids))
        return None
