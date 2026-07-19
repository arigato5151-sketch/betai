from __future__ import annotations

from datetime import datetime
from typing import Iterable

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

    def get_league_history(
        self, *, league_id: int, season: int, before: datetime
    ) -> list[HistoricalFixture]:
        return (
            self.db.query(HistoricalFixture)
            .filter(
                HistoricalFixture.league_id == league_id,
                HistoricalFixture.season == season,
                HistoricalFixture.kickoff < before,
            )
            .order_by(
                HistoricalFixture.kickoff.asc(), HistoricalFixture.fixture_id.asc()
            )
            .all()
        )

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
