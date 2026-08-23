from __future__ import annotations

import logging
from collections import Counter
from typing import Literal

from celery import shared_task

from app.core.config import settings
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.odds_snapshot_repository import OddsSnapshotRepository, OddsSnapshotWindow
from app.db.repository import MatchPredictionRepository
from app.db.session import SessionLocal
from app.db.models import MatchPrediction
from app.prediction.audit import PredictionAuditor
from app.services.api_football import APIFootballClient
from app.providers.openligadb import OpenLigaDBClient
from app.services.result_verification import (
    ResultVerificationService,
    canonical_result_source,
    provider_request_fixture_id,
)
from app.services.task_lock import DistributedTaskLock
from app.tasks.base import TransientTask
from app.tasks._helpers import _run_async
from app.tasks.ml_tasks import retrain_ml_model_task

logger = logging.getLogger("bet-ai-pro.tasks")


@shared_task(name="app.tasks.jobs.sync_completed_matches_task", base=TransientTask)
def sync_completed_matches_task() -> dict[str, object]:
    """
    Synchronizes past prediction records with actual outcomes.
    Calculates audited ROI and CLV.
    """
    logger.info("Starting past predictions synchronization task...")
    api_client = APIFootballClient()
    openligadb_client = OpenLigaDBClient()

    with DistributedTaskLock("sync-completed-matches", ttl_seconds=1800) as lock:
        if not lock.acquired:
            logger.info("Completed match synchronization already running; skipped.")
            return {"status": "locked", "verified": 0}

        return _sync_completed_matches(api_client, openligadb_client)


def _sync_completed_matches(
    api_client: APIFootballClient,
    openligadb_client: OpenLigaDBClient,
) -> dict[str, object]:
    with SessionLocal() as db:
        repo = MatchPredictionRepository(db)
        historical_repo = HistoricalFixtureRepository(db)
        odds_repo = OddsSnapshotRepository(db)

        predictions = (
            db.query(MatchPrediction)
            .filter(
                MatchPrediction.actual_result.is_(None),
                MatchPrediction.training_eligible.is_(True),
            )
            .order_by(MatchPrediction.id.desc())
            .limit(100)
            .all()
        )

        if not predictions:
            logger.info("No unresolved predictions found to sync.")
            return {"status": "ready", "verified": 0, "pending": 0}

        counters: Counter[str] = Counter()
        for pred in predictions:
            try:
                source = canonical_result_source(pred.fixture_source)
                request_id = provider_request_fixture_id(pred)
                historical_fixture = (
                    historical_repo.get_by_fixture_id(pred.fixture_id)
                    if pred.fixture_id is not None
                    else None
                )
                decision = ResultVerificationService.verify_historical(
                    pred, historical_fixture
                )
                if decision.status == "pending":
                    if request_id is not None and source == "openligadb":
                        fixture = _run_async(
                            openligadb_client.get_fixture_by_id(request_id)
                        )
                        decision = ResultVerificationService.verify(pred, fixture)
                    elif request_id is not None and source == "api_football":
                        fixture = _run_async(api_client.get_fixture_by_id(request_id))
                        decision = ResultVerificationService.verify(pred, fixture)
                    else:
                        composite = historical_repo.get_by_composite_key(
                            league_id=pred.league_id,
                            home_team_id=pred.home_team_id,
                            away_team_id=pred.away_team_id,
                            kickoff=pred.kickoff,
                        )
                        decision = ResultVerificationService.verify_composite(
                            pred, composite
                        )
                counters[decision.status] += 1
                if decision.status != "verified" or decision.result is None:
                    if decision.status in {"conflict", "rejected"}:
                        quarantine_status: Literal["conflict", "rejected"] = (
                            "conflict" if decision.status == "conflict" else "rejected"
                        )
                        repo.mark_result_verification(
                            pred.id,
                            status=quarantine_status,
                            note=decision.reason or "result_verification_failed",
                        )
                    continue
                verified = decision.result

                roi = PredictionAuditor.calculate_bet_roi(
                    pred.prediction, verified.actual_result, pred.odd
                )

                closing_snapshot = (
                    odds_repo.closing_snapshot(
                        fixture_id=pred.fixture_id,
                        kickoff=pred.kickoff,
                        closing_window_hours=(
                            settings.ODDS_COLLECTOR_CLOSING_WINDOW_HOURS
                        ),
                    )
                    if pred.fixture_id is not None and pred.kickoff is not None
                    else None
                )
                market = (
                    {"raw_odds": OddsSnapshotWindow.outcome_dict(closing_snapshot)}
                    if closing_snapshot is not None
                    else None
                )
                closing_odd = PredictionAuditor.select_closing_odd(
                    market, pred.prediction
                )
                clv = (
                    PredictionAuditor.calculate_clv(pred.odd, closing_odd)
                    if closing_odd is not None
                    else None
                )
                closing_odd_snapshot_at = (
                    closing_snapshot.captured_at
                    if closing_snapshot is not None
                    else None
                )
                closing_odd_snapshot_id = (
                    closing_snapshot.id if closing_snapshot is not None else None
                )

                repo.update_result(
                    record_id=pred.id,
                    actual_result=verified.actual_result,
                    actual_score_home=verified.home_score,
                    actual_score_away=verified.away_score,
                    roi=roi,
                    clv=clv,
                    closing_odds=closing_odd,
                    closing_odds_snapshot_at=closing_odd_snapshot_at,
                    closing_odds_snapshot_id=closing_odd_snapshot_id,
                    verification_status="verified",
                    result_source=verified.source,
                    result_provider_fixture_id=verified.provider_fixture_id,
                )

            except Exception as e:
                logger.error("Failed syncing outcome for prediction ID %s: %s", pred.id, e)
                counters["errors"] += 1

        logger.info(
            "Completed match synchronization result: %s",
            dict(counters),
        )

        if counters["verified"] > 0:
            logger.info("Sync completed. Triggering ML model update...")
            try:
                retrain_ml_model_task.delay()
            except Exception:
                logger.warning(
                    "Completed-match sync finished but ML retraining could not "
                    "be enqueued; the next scheduled run will retry.",
                    exc_info=True,
                )

        return {"status": "ready", **dict(counters)}
