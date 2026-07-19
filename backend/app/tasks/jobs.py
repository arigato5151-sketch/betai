import asyncio
import logging
from datetime import date
from typing import cast
from celery import shared_task
from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.session import SessionLocal
from app.db.repository import MatchPredictionRepository
from app.db.models import MatchPrediction
from app.services.api_football import APIFootballClient
from app.prediction.ml.model import ml_pipeline
from app.prediction.ensemble_weights import ensemble_weight_manager
from app.prediction.audit import PredictionAuditor

logger = logging.getLogger("bet-ai-pro.tasks")


def _run_async(coro):
    """Utility helper to run async coroutines in synchronous Celery task threads."""
    return asyncio.run(coro)


def _current_football_season(today: date | None = None) -> int:
    current = today or date.today()
    return current.year if current.month >= 7 else current.year - 1


@shared_task(name="app.tasks.jobs.sync_historical_fixtures_task")
def sync_historical_fixtures_task(seasons: list[int] | None = None) -> dict:
    """Ingest completed fixtures; pass seasons explicitly for a repeatable backfill."""
    target_seasons: list[int] = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    league_ids: list[int] = sorted(cast(set[int], ALLOWED_LEAGUE_IDS))
    api_client = APIFootballClient()
    fixture_rows: list[dict] = []
    failures: list[dict[str, int | str]] = []

    # Network work completes before opening a database transaction.
    for season in target_seasons:
        for league_id in league_ids:
            try:
                fixture_rows.extend(
                    _run_async(api_client.get_completed_fixtures(league_id, season))
                )
            except Exception as exc:
                logger.exception(
                    "Historical fixture fetch failed for league=%s season=%s",
                    league_id,
                    season,
                )
                failures.append(
                    {
                        "league_id": league_id,
                        "season": season,
                        "error": type(exc).__name__,
                    }
                )

    with SessionLocal() as db:
        processed = HistoricalFixtureRepository(db).upsert_many(fixture_rows)

    result: dict[str, object] = {
        "seasons": target_seasons,
        "fixtures_processed": processed,
        "failed_league_seasons": failures,
    }
    logger.info("Historical fixture synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.retrain_ml_model_task")
def retrain_ml_model_task() -> str:
    """Asynchronously triggers model retraining on Celery workers."""
    logger.info("Initializing background ML model retraining job...")

    with SessionLocal() as db:
        repo = MatchPredictionRepository(db)
        labeled_rows = repo.get_all_labeled()

        weight_result = ensemble_weight_manager.optimize_and_activate(labeled_rows)
        logger.info("Ensemble weight calibration result: %s", weight_result)
        success = ml_pipeline.train_pipeline(labeled_rows)

        if success:
            logger.info("ML model retraining job completed successfully.")
            return "Retraining success."
        else:
            logger.warning("ML model retraining job skipped or failed.")
            return "Retraining failed."


@shared_task(name="app.tasks.jobs.sync_completed_matches_task")
def sync_completed_matches_task() -> str:
    """
    Synchronizes past prediction records with actual outcomes.
    Calculates audited ROI and CLV.
    """
    logger.info("Starting past predictions synchronization task...")
    api_client = APIFootballClient()

    with SessionLocal() as db:
        repo = MatchPredictionRepository(db)

        # Get all predictions where actual outcome is not resolved yet
        predictions = (
            db.query(MatchPrediction)
            .filter(MatchPrediction.actual_result.is_(None))
            .order_by(MatchPrediction.id.desc())
            .limit(100)
            .all()
        )

        if not predictions:
            logger.info("No unresolved predictions found to sync.")
            return "No predictions to sync."

        count = 0
        for pred in predictions:
            if not pred.fixture_id:
                continue

            try:
                # Retrieve actual fixture status from API asynchronously
                fixture = _run_async(api_client.get_fixture_by_id(pred.fixture_id))

                if not fixture or fixture.get("status") not in {"FT", "AET", "PEN"}:
                    continue

                # Parse score
                score_str = fixture.get("score")  # Ex: "2 - 1"
                if not score_str:
                    continue

                parts = score_str.split("-")
                home_score = int(parts[0].strip())
                away_score = int(parts[1].strip())

                # Determine actual outcome
                if home_score > away_score:
                    result = "HOME_WIN"
                elif home_score < away_score:
                    result = "AWAY_WIN"
                else:
                    result = "DRAW"

                # Calculate auditing metrics
                roi = PredictionAuditor.calculate_bet_roi(
                    pred.prediction, result, pred.odd
                )

                # Fetch closing odds dynamically if available to compute CLV
                market = _run_async(api_client.get_fixture_market(pred.fixture_id))
                closing_odd = market["raw_odds"]["HOME_WIN"] if market else pred.odd
                clv = PredictionAuditor.calculate_clv(pred.odd, closing_odd)

                # Persist verified audit metrics to db
                repo.update_result(
                    record_id=pred.id,
                    actual_result=result,
                    actual_score_home=home_score,
                    actual_score_away=away_score,
                    roi=roi,
                    clv=clv,
                    closing_odds=closing_odd,
                )
                count += 1

            except Exception as e:
                logger.error(f"Failed syncing outcome for prediction ID {pred.id}: {e}")

        logger.info(
            f"Synchronized outcomes for {count} resolved predictions successfully."
        )

        # Trigger ML retraining if new samples synced successfully
        if count > 0:
            logger.info("Sync completed. Triggering ML model update...")
            retrain_ml_model_task.delay()

        return f"Synced {count} predictions."
