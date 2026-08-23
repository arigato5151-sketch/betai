from __future__ import annotations

import logging
from typing import Any

from celery import shared_task

from app.core.config import settings
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import PlayerContextRepository
from app.db.repository import MatchPredictionRepository
from app.db.session import SessionLocal
from app.prediction.ml.model import ml_pipeline
from app.prediction.ml.training_data import HistoricalTrainingDataBuilder
from app.prediction.ensemble_weights import ensemble_weight_manager
from app.services.cache import cache
from app.services.model_monitoring import ModelMonitoringService
from app.services.task_lock import DistributedTaskLock
from app.tasks.base import TransientTask
from app.tasks._helpers import _run_async

logger = logging.getLogger("bet-ai-pro.tasks")


def _run_model_retraining() -> str:
    logger.info("Initializing background ML model retraining job...")

    # Build a detached, in-memory training snapshot while the database session
    # is open. Model fitting can take minutes and must never hold a pool slot.
    with SessionLocal() as db:
        repo = MatchPredictionRepository(db)
        historical_repo = HistoricalFixtureRepository(db)
        labeled_rows = repo.get_all_labeled()
        historical_fixtures = historical_repo.get_recent(
            settings.ML_TRAINING_MAX_HISTORICAL_FIXTURES
        )
        player_context_repo = PlayerContextRepository(db)
        player_performances = player_context_repo.get_performances_for_fixture_ids(
            (fixture.fixture_id for fixture in historical_fixtures),
            max_results=settings.ML_TRAINING_MAX_PLAYER_PERFORMANCES,
        )
        historical_rows = HistoricalTrainingDataBuilder().build(
            historical_fixtures,
            player_performances=player_performances,
            team_locations=player_context_repo.get_all_team_locations(),
        )
        labeled_fixture_ids = {
            row.fixture_id for row in labeled_rows if row.fixture_id is not None
        }
        training_rows: list[Any] = [
            row for row in historical_rows if row.fixture_id not in labeled_fixture_ids
        ]
        training_rows.extend(labeled_rows)

    weight_result = ensemble_weight_manager.optimize_and_activate(labeled_rows)
    logger.info("Ensemble weight calibration result: %s", weight_result)
    logger.info(
        "Prepared %s historical and %s labeled-prediction ML samples.",
        len(training_rows) - len(labeled_rows),
        len(labeled_rows),
    )
    ml_pipeline.status()
    success = ml_pipeline.train_pipeline(training_rows)

    tiered_report: dict[str, object] = {}
    if len(historical_fixtures) >= settings.TIERED_RETRAIN_MIN_FIXTURES:
        try:
            from app.prediction.ml.train_tiered_models import (
                ModelPromotionRejected,
                train_tiered_models,
            )

            tiered_report = train_tiered_models(fixtures=historical_fixtures)
            logger.info("Tiered model retraining completed: %s", tiered_report)
        except ModelPromotionRejected as exc:
            tiered_report = {
                "status": "promotion_rejected",
                "tier1_metrics": exc.tier1_metrics,
            }
            logger.warning(
                "Tiered model candidate rejected by market gate: %s",
                exc.tier1_metrics,
            )
        except Exception:
            logger.exception("Tiered model retraining failed.")

    if success:
        logger.info("ML model retraining job completed successfully.")
        if tiered_report:
            logger.info(
                "Tiered bundle artifact_version=%s",
                tiered_report.get("artifact_version"),
            )
        return "Retraining success."

    logger.warning("ML model retraining job skipped or failed.")
    return "Retraining failed."


@shared_task(name="app.tasks.jobs.retrain_ml_model_task", base=TransientTask)
def retrain_ml_model_task() -> str:
    """Asynchronously trigger a single model retraining job across all workers."""
    with DistributedTaskLock(
        "model_retraining",
        ttl_seconds=settings.MODEL_TRAINING_LOCK_TTL_SECONDS,
    ) as task_lock:
        if not task_lock.acquired:
            return (
                "Retraining lock unavailable."
                if not task_lock.available
                else "Retraining already running."
            )
        return _run_model_retraining()


@shared_task(name="app.tasks.jobs.monitor_model_drift_task", base=TransientTask)
def monitor_model_drift_task() -> dict[str, object]:
    """Queue a challenger training run when recent calibration materially degrades."""
    active_artifact_version = ml_pipeline.status().get("artifact_version")
    artifact_version = (
        active_artifact_version if isinstance(active_artifact_version, str) else None
    )
    with SessionLocal() as db:
        status = ModelMonitoringService(db).snapshot(artifact_version)
    cooldown_key = f"drift-retraining:{artifact_version or 'unavailable'}"
    cooldown_active = bool(_run_async(cache.get("operations", cooldown_key)))
    drift_detected = bool(status["drift_detected"])
    retraining_queued = False
    queue_contention = False
    if drift_detected and not cooldown_active:
        with DistributedTaskLock("drift-retraining-queue", ttl_seconds=60) as lock:
            if not lock.acquired:
                queue_contention = True
            else:
                try:
                    _run_async(
                        cache.set(
                            "operations",
                            cooldown_key,
                            {
                                "artifact_version": artifact_version,
                                "reason": "confirmed_drift",
                            },
                            settings.MODEL_DRIFT_RETRAIN_COOLDOWN_SECONDS,
                        )
                    )
                    retrain_ml_model_task.delay()
                    retraining_queued = True
                except Exception:
                    logger.exception("Failed to enqueue drift-triggered retraining.")
    return {
        **status,
        "retraining_queued": retraining_queued,
        "retraining_suppressed_by_cooldown": drift_detected and cooldown_active,
        "queue_contention": queue_contention,
    }
