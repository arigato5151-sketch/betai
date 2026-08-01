from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.allowed_leagues import ALLOWED_LEAGUES
from app.db.models import HistoricalFixture, MatchPrediction, SyncRun


def _percentage(numerator: int, denominator: int) -> float:
    return round((numerator / denominator) * 100.0, 2) if denominator else 0.0


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _valid_starting_xi(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 11
        and len(set(value)) == 11
        and all(isinstance(player_id, int) and player_id > 0 for player_id in value)
    )


class DataQualityService:
    def __init__(self, db: Session):
        self.db = db

    def snapshot(self, now: datetime | None = None) -> dict[str, Any]:
        current_time = _utc(now) or datetime.now(UTC)
        fixture_total = self.db.query(func.count(HistoricalFixture.id)).scalar() or 0
        lineup_total = sum(
            _valid_starting_xi(home_lineup) and _valid_starting_xi(away_lineup)
            for home_lineup, away_lineup in self.db.query(
                HistoricalFixture.home_starting_xi,
                HistoricalFixture.away_starting_xi,
            ).all()
        )
        oldest_kickoff, newest_kickoff, last_updated = self.db.query(
            func.min(HistoricalFixture.kickoff),
            func.max(HistoricalFixture.kickoff),
            func.max(HistoricalFixture.updated_at),
        ).one()
        league_count = (
            self.db.query(
                func.count(func.distinct(HistoricalFixture.league_id))
            ).scalar()
            or 0
        )
        season_count = (
            self.db.query(func.count(func.distinct(HistoricalFixture.season))).scalar()
            or 0
        )
        source_counts = {
            str(source): int(count)
            for source, count in self.db.query(
                HistoricalFixture.data_source,
                func.count(HistoricalFixture.id),
            )
            .group_by(HistoricalFixture.data_source)
            .all()
        }
        current_season = (
            current_time.year if current_time.month >= 7 else current_time.year - 1
        )
        current_season_counts = {
            int(league_id): int(count)
            for league_id, count in self.db.query(
                HistoricalFixture.league_id,
                func.count(HistoricalFixture.id),
            )
            .filter(HistoricalFixture.season == current_season)
            .group_by(HistoricalFixture.league_id)
            .all()
        }
        league_coverage = [
            {
                "league_id": cast(int, league["id"]),
                "league_name": str(league["name"]),
                "fixtures": current_season_counts.get(cast(int, league["id"]), 0),
                "covered": current_season_counts.get(cast(int, league["id"]), 0) > 0,
            }
            for league in ALLOWED_LEAGUES
        ]

        prediction_total = self.db.query(func.count(MatchPrediction.id)).scalar() or 0
        labeled_total = (
            self.db.query(func.count(MatchPrediction.id))
            .filter(MatchPrediction.actual_result.isnot(None))
            .scalar()
            or 0
        )
        closing_total = (
            self.db.query(func.count(MatchPrediction.id))
            .filter(MatchPrediction.closing_odds.isnot(None))
            .scalar()
            or 0
        )
        provenance_total = (
            self.db.query(func.count(MatchPrediction.id))
            .filter(
                MatchPrediction.feature_schema_version.isnot(None),
                MatchPrediction.ensemble_version.isnot(None),
                MatchPrediction.analyzed_at.isnot(None),
            )
            .scalar()
            or 0
        )

        latest_run = (
            self.db.query(SyncRun)
            .order_by(SyncRun.started_at.desc(), SyncRun.id.desc())
            .first()
        )
        normalized_updated = _utc(last_updated)
        freshness_hours = (
            round((current_time - normalized_updated).total_seconds() / 3600.0, 2)
            if normalized_updated
            else None
        )
        lineup_coverage = _percentage(lineup_total, fixture_total)
        labeled_coverage = _percentage(labeled_total, prediction_total)
        closing_coverage = _percentage(closing_total, labeled_total)
        provenance_coverage = _percentage(provenance_total, prediction_total)

        score = 0.0
        if fixture_total:
            score += 20.0
        if freshness_hours is not None:
            score += (
                20.0
                if freshness_hours <= 48
                else 10.0 if freshness_hours <= 168 else 0.0
            )
        score += min(20.0, lineup_coverage * 0.2)
        score += min(20.0, labeled_coverage * 0.2)
        score += min(10.0, closing_coverage * 0.1)
        score += min(10.0, provenance_coverage * 0.1)
        score = round(score, 2)

        return {
            "status": (
                "healthy" if score >= 80 else "warning" if score >= 50 else "critical"
            ),
            "score": score,
            "historical": {
                "fixtures": fixture_total,
                "leagues": league_count,
                "seasons": season_count,
                "oldest_kickoff": _utc(oldest_kickoff),
                "newest_kickoff": _utc(newest_kickoff),
                "last_updated": normalized_updated,
                "freshness_hours": freshness_hours,
                "lineup_coverage_pct": lineup_coverage,
                "source_counts": source_counts,
                "current_season": current_season,
                "current_season_coverage": league_coverage,
                "current_season_covered_leagues": sum(
                    item["covered"] for item in league_coverage
                ),
                "current_season_missing_league_ids": [
                    item["league_id"] for item in league_coverage if not item["covered"]
                ],
            },
            "predictions": {
                "total": prediction_total,
                "labeled": labeled_total,
                "labeled_coverage_pct": labeled_coverage,
                "closing_odds_coverage_pct": closing_coverage,
                "provenance_coverage_pct": provenance_coverage,
            },
            "latest_sync": (
                {
                    "job_name": latest_run.job_name,
                    "status": latest_run.status,
                    "started_at": _utc(latest_run.started_at),
                    "finished_at": _utc(latest_run.finished_at),
                    "fixtures_processed": latest_run.fixtures_processed,
                    "failures": latest_run.failures or [],
                    "error_type": latest_run.error_type,
                }
                if latest_run
                else None
            ),
        }

    def start_sync(self, job_name: str, seasons: list[int]) -> SyncRun:
        run = SyncRun(
            job_name=job_name,
            status="running",
            target_seasons=seasons,
            failures=[],
        )
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run

    def finish_sync(
        self,
        run_id: int,
        *,
        processed: int,
        failures: list[dict[str, object]],
        error_type: str | None = None,
    ) -> None:
        run = self.db.get(SyncRun, run_id)
        if run is None:
            return
        run.status = "failed" if error_type else "partial" if failures else "succeeded"
        run.finished_at = datetime.now(UTC)
        run.fixtures_processed = processed
        run.failures = failures
        run.error_type = error_type
        self.db.commit()
