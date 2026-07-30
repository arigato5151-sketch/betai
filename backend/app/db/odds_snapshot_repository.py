from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import FixtureOddsSnapshot, utc_now

OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _fixture_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("fixture_id must be a positive integer")
    return value


def _odds(raw: Mapping[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for outcome in OUTCOMES:
        value = raw.get(outcome)
        if isinstance(value, bool):
            raise ValueError(f"{outcome} odd must be numeric")
        try:
            numeric = float(str(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{outcome} odd must be numeric") from exc
        if not math.isfinite(numeric) or not 1.0 < numeric <= 1000.0:
            raise ValueError(f"{outcome} odd must be between 1 and 1000")
        normalized[outcome] = round(numeric, 6)
    return normalized


@dataclass(frozen=True, slots=True)
class OddsSnapshotWindow:
    opening: FixtureOddsSnapshot
    current: FixtureOddsSnapshot

    @staticmethod
    def outcome_dict(snapshot: FixtureOddsSnapshot) -> dict[str, float]:
        return {
            "HOME_WIN": snapshot.home_odd,
            "DRAW": snapshot.draw_odd,
            "AWAY_WIN": snapshot.away_odd,
        }


class OddsSnapshotRepository:
    def __init__(self, db: Session):
        self.db = db

    def record(
        self,
        *,
        fixture_id: int,
        raw_odds: Mapping[str, object],
        captured_at: datetime,
        source: str = "api_football_odds",
        bookmaker: str | None = None,
        minimum_interval_seconds: int = 300,
        details: Mapping[str, object] | None = None,
    ) -> FixtureOddsSnapshot:
        fixture_id = _fixture_id(fixture_id)
        captured_at = _utc(captured_at)
        odds = _odds(raw_odds)
        source = source.strip()
        if not source:
            raise ValueError("odds source cannot be blank")
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")

        latest = self.latest(fixture_id=fixture_id, before=captured_at)
        if (
            latest is not None
            and (captured_at - _utc(latest.captured_at)).total_seconds()
            < minimum_interval_seconds
            and self._same_odds(latest, odds)
        ):
            return latest

        snapshot = FixtureOddsSnapshot(
            fixture_id=fixture_id,
            home_odd=odds["HOME_WIN"],
            draw_odd=odds["DRAW"],
            away_odd=odds["AWAY_WIN"],
            source=source[:50],
            bookmaker=(bookmaker or "").strip()[:100] or None,
            captured_at=captured_at,
            details=dict(details or {}),
            created_at=utc_now(),
        )
        try:
            self.db.add(snapshot)
            self.db.commit()
            self.db.refresh(snapshot)
        except SQLAlchemyError:
            self.db.rollback()
            raise
        return snapshot

    def latest(
        self,
        *,
        fixture_id: int,
        before: datetime,
    ) -> FixtureOddsSnapshot | None:
        return (
            self.db.query(FixtureOddsSnapshot)
            .filter(
                FixtureOddsSnapshot.fixture_id == _fixture_id(fixture_id),
                FixtureOddsSnapshot.captured_at <= _utc(before),
            )
            .order_by(
                FixtureOddsSnapshot.captured_at.desc(),
                FixtureOddsSnapshot.id.desc(),
            )
            .first()
        )

    def movement_window(
        self,
        *,
        fixture_id: int,
        before: datetime,
        minimum_interval_seconds: int,
    ) -> OddsSnapshotWindow | None:
        rows = (
            self.db.query(FixtureOddsSnapshot)
            .filter(
                FixtureOddsSnapshot.fixture_id == _fixture_id(fixture_id),
                FixtureOddsSnapshot.captured_at < _utc(before),
            )
            .order_by(
                FixtureOddsSnapshot.captured_at.asc(),
                FixtureOddsSnapshot.id.asc(),
            )
            .all()
        )
        if len(rows) < 2:
            return None
        opening, current = rows[0], rows[-1]
        elapsed = (
            _utc(current.captured_at) - _utc(opening.captured_at)
        ).total_seconds()
        if elapsed < minimum_interval_seconds:
            return None
        return OddsSnapshotWindow(opening=opening, current=current)

    @staticmethod
    def _same_odds(
        snapshot: FixtureOddsSnapshot,
        odds: Mapping[str, float],
    ) -> bool:
        return (
            snapshot.home_odd == odds["HOME_WIN"]
            and snapshot.draw_odd == odds["DRAW"]
            and snapshot.away_odd == odds["AWAY_WIN"]
        )
