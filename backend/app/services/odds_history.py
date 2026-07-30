from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.odds_snapshot_repository import OddsSnapshotRepository, OddsSnapshotWindow
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _stored_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class OddsHistoryService:
    """Turn current API-Football markets into point-in-time movement inputs."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self.session_factory = session_factory

    def enrich_prefill(
        self,
        prefill: Mapping[str, Any],
        *,
        captured_at: datetime | None = None,
    ) -> dict[str, Any]:
        enriched = dict(prefill)
        fixture = prefill.get("fixture")
        market = prefill.get("market_1x2")
        if not isinstance(fixture, Mapping) or not isinstance(market, Mapping):
            return enriched

        fixture_id = fixture.get("fixture_id")
        kickoff = _datetime(fixture.get("kickoff"))
        raw_odds = market.get("raw_odds")
        observed_at = (captured_at or datetime.now(UTC)).astimezone(UTC)
        if (
            isinstance(fixture_id, bool)
            or not isinstance(fixture_id, int)
            or fixture_id <= 0
            or kickoff is None
            or observed_at >= kickoff
            or not isinstance(raw_odds, Mapping)
        ):
            return enriched

        try:
            with self.session_factory() as db:
                repository = OddsSnapshotRepository(db)
                repository.record(
                    fixture_id=fixture_id,
                    raw_odds=raw_odds,
                    captured_at=observed_at,
                    source="api_football_odds",
                    bookmaker=(
                        str(market.get("bookmaker"))
                        if market.get("bookmaker")
                        else None
                    ),
                    minimum_interval_seconds=settings.ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS,
                    details={
                        "method": str(market.get("method") or "proportional_devig"),
                        "overround_pct": market.get("overround_pct"),
                    },
                )
                window = repository.movement_window(
                    fixture_id=fixture_id,
                    before=kickoff,
                    minimum_interval_seconds=(
                        settings.ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS
                    ),
                )
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning(
                "Odds snapshot could not be persisted",
                extra={
                    "fixture_id": fixture_id,
                    "error_type": type(exc).__name__,
                },
            )
            return enriched

        enriched["odds_history"] = self._history_metadata(window)
        if window is None:
            return enriched
        enriched.update(
            opening_odds_1x2=OddsSnapshotWindow.outcome_dict(window.opening),
            current_odds_1x2=OddsSnapshotWindow.outcome_dict(window.current),
            opening_odds_at=_stored_utc(window.opening.captured_at).isoformat(),
            current_odds_at=_stored_utc(window.current.captured_at).isoformat(),
        )
        return enriched

    def should_collect(
        self,
        *,
        fixture_id: int,
        kickoff: datetime,
        observed_at: datetime,
        refresh_interval_seconds: int,
        closing_window_hours: int,
    ) -> bool:
        """Return whether a quota-conscious background observation is due."""
        kickoff = _stored_utc(kickoff)
        observed_at = _stored_utc(observed_at)
        if (
            fixture_id <= 0
            or refresh_interval_seconds <= 0
            or closing_window_hours <= 0
            or observed_at >= kickoff
        ):
            return False
        try:
            with self.session_factory() as db:
                latest = OddsSnapshotRepository(db).latest(
                    fixture_id=fixture_id,
                    before=observed_at,
                )
        except (SQLAlchemyError, ValueError) as exc:
            logger.warning(
                "Odds collection schedule could not be evaluated",
                extra={
                    "fixture_id": fixture_id,
                    "error_type": type(exc).__name__,
                },
            )
            return False
        if latest is None:
            return True
        latest_at = _stored_utc(latest.captured_at)
        if (observed_at - latest_at).total_seconds() < refresh_interval_seconds:
            return False
        return kickoff - observed_at <= timedelta(hours=closing_window_hours)

    @staticmethod
    def _history_metadata(
        window: OddsSnapshotWindow | None,
    ) -> dict[str, object]:
        if window is None:
            return {
                "status": "collecting",
                "source": "api_football_odds",
                "minimum_interval_seconds": (
                    settings.ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS
                ),
            }
        return {
            "status": "ready",
            "source": "api_football_odds",
            "confidence": settings.ODDS_SNAPSHOT_CONFIDENCE,
            "opening_captured_at": _stored_utc(window.opening.captured_at).isoformat(),
            "current_captured_at": _stored_utc(window.current.captured_at).isoformat(),
        }


odds_history_service = OddsHistoryService()
