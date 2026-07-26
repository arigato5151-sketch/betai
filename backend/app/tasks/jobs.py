import asyncio
import logging
from datetime import date
from typing import Any, cast
from celery import shared_task
from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS
from app.core.config import settings
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import (
    PlayerContextRepository,
    is_fixture_player_context_complete,
)
from app.db.session import SessionLocal
from app.db.repository import MatchPredictionRepository
from app.db.models import MatchPrediction
from app.services.api_football import APIFootballClient
from app.services.football_data_csv import (
    FootballDataCSVClient,
    FootballDataDownloadError,
    FootballDataImport,
)
from app.prediction.ml.model import ml_pipeline
from app.prediction.ml.training_data import HistoricalTrainingDataBuilder
from app.prediction.ensemble_weights import ensemble_weight_manager
from app.prediction.audit import PredictionAuditor
from app.services.data_quality import DataQualityService

logger = logging.getLogger("bet-ai-pro.tasks")


def _run_async(coro):
    """Utility helper to run async coroutines in synchronous Celery task threads."""
    return asyncio.run(coro)


def _current_football_season(today: date | None = None) -> int:
    current = today or date.today()
    return current.year if current.month >= 7 else current.year - 1


async def _enrich_historical_player_context(
    api_client: APIFootballClient,
    fixture_rows: list[dict],
    existing_fixture_ids: set[int],
) -> int:
    """Backfill a bounded number of immutable fixture-player responses per run."""
    limit = settings.PLAYER_CONTEXT_SYNC_MAX_FIXTURES
    if limit <= 0:
        return 0

    candidates_by_id: dict[int, dict] = {}
    for row in fixture_rows:
        fixture_id = row.get("fixture_id")
        if (
            not isinstance(fixture_id, int)
            or fixture_id <= 0
            or fixture_id in existing_fixture_ids
        ):
            continue
        embedded_performances = row.get("player_performances")
        home_team_id = row.get("home_team_id")
        away_team_id = row.get("away_team_id")
        embedded_context_is_complete = (
            isinstance(home_team_id, int)
            and not isinstance(home_team_id, bool)
            and isinstance(away_team_id, int)
            and not isinstance(away_team_id, bool)
            and isinstance(embedded_performances, list)
            and is_fixture_player_context_complete(
                (
                    performance
                    for performance in embedded_performances
                    if isinstance(performance, dict)
                ),
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
        )
        if embedded_context_is_complete:
            continue
        candidates_by_id[fixture_id] = row
    candidates = sorted(
        candidates_by_id.values(),
        key=lambda row: (str(row.get("kickoff") or ""), row["fixture_id"]),
        reverse=True,
    )[:limit]
    semaphore = asyncio.Semaphore(settings.PLAYER_CONTEXT_SYNC_CONCURRENCY)

    async def enrich(row: dict) -> bool:
        async with semaphore:
            try:
                context = await api_client.get_fixture_player_context(
                    fixture_id=row["fixture_id"],
                    league_id=row["league_id"],
                    kickoff=row["kickoff"],
                    home_team_id=row["home_team_id"],
                    away_team_id=row["away_team_id"],
                )
            except Exception:
                logger.exception(
                    "Fixture player context fetch failed for fixture=%s",
                    row["fixture_id"],
                )
                return False

        performances = context.get("player_performances")
        if not isinstance(performances, list) or not performances:
            return False
        normalized_performances = [
            performance for performance in performances if isinstance(performance, dict)
        ]
        if not is_fixture_player_context_complete(
            normalized_performances,
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
        ):
            logger.warning(
                "Incomplete fixture player context rejected for fixture=%s",
                row["fixture_id"],
            )
            return False
        row["player_performances"] = normalized_performances
        for side in ("home", "away"):
            lineup = context.get(f"{side}_starting_xi")
            if isinstance(lineup, list) and len(lineup) == 11:
                row[f"{side}_starting_xi"] = lineup
        return True

    results = await asyncio.gather(*(enrich(row) for row in candidates))
    return sum(not succeeded for succeeded in results)


@shared_task(name="app.tasks.jobs.sync_historical_fixtures_task")
def sync_historical_fixtures_task(seasons: list[int] | None = None) -> dict:
    """Ingest completed fixtures; pass seasons explicitly for a repeatable backfill."""
    target_seasons: list[int] = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    with SessionLocal() as db:
        sync_run_id = (
            DataQualityService(db).start_sync("historical_fixtures", target_seasons).id
        )

    league_ids: list[int] = sorted(cast(set[int], ALLOWED_LEAGUE_IDS))
    api_client = APIFootballClient()
    fixture_rows: list[dict] = []
    failures: list[dict[str, int | str]] = []

    # Bulk fixture network work completes before opening a database transaction.
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

    try:
        fixture_ids = {
            row["fixture_id"]
            for row in fixture_rows
            if isinstance(row.get("fixture_id"), int) and row["fixture_id"] > 0
        }
        with SessionLocal() as db:
            existing_context_ids = PlayerContextRepository(
                db
            ).get_fixture_ids_with_complete_player_context(fixture_ids)
        player_context_failures = _run_async(
            _enrich_historical_player_context(
                api_client,
                fixture_rows,
                existing_context_ids,
            )
        )

        performance_rows: list[dict[str, object]] = []
        normalized_fixture_rows: list[dict] = []
        for fixture_row in fixture_rows:
            normalized_fixture = dict(fixture_row)
            nested_performances = normalized_fixture.pop("player_performances", [])
            if isinstance(nested_performances, list):
                performance_rows.extend(
                    row for row in nested_performances if isinstance(row, dict)
                )
            normalized_fixture_rows.append(normalized_fixture)
        with SessionLocal() as db:
            processed = HistoricalFixtureRepository(db).upsert_many(
                normalized_fixture_rows
            )
            player_performances_processed = PlayerContextRepository(
                db
            ).upsert_performances(performance_rows)
            DataQualityService(db).finish_sync(
                sync_run_id,
                processed=processed,
                failures=cast(list[dict[str, object]], failures),
            )
    except Exception as exc:
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                sync_run_id,
                processed=0,
                failures=cast(list[dict[str, object]], failures),
                error_type=type(exc).__name__,
            )
        raise

    result: dict[str, object] = {
        "seasons": target_seasons,
        "fixtures_processed": processed,
        "player_performances_processed": player_performances_processed,
        "player_context_failures": player_context_failures,
        "failed_league_seasons": failures,
    }
    logger.info("Historical fixture synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_football_data_fixtures_task")
def sync_football_data_fixtures_task(seasons: list[int] | None = None) -> dict:
    """Import completed league fixtures from the public Football-Data CSV feeds."""
    target_seasons = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    client = FootballDataCSVClient()
    prefetched: dict[tuple[int, int], FootballDataImport] = {}
    fallback_from_season: int | None = None

    if seasons is None:
        current_season = target_seasons[0]
        probe_league_id = min(client.supported_league_ids)
        try:
            prefetched[(probe_league_id, current_season)] = _run_async(
                client.get_completed_fixtures(probe_league_id, current_season)
            )
        except FootballDataDownloadError as exc:
            if exc.status_code == 404:
                fallback_from_season = current_season
                target_seasons = [current_season - 1]
                logger.info(
                    "Football-Data season=%s is not published; falling back to %s.",
                    current_season,
                    target_seasons[0],
                )

    with SessionLocal() as db:
        sync_run_id = (
            DataQualityService(db).start_sync("football_data_csv", target_seasons).id
        )

    fixture_rows: list[dict] = []
    skipped_rows = 0
    failures: list[dict[str, int | str]] = []

    for season in target_seasons:
        for league_id in sorted(client.supported_league_ids):
            try:
                imported = prefetched.pop((league_id, season), None)
                if imported is None:
                    imported = _run_async(
                        client.get_completed_fixtures(league_id, season)
                    )
                fixture_rows.extend(imported.fixtures)
                skipped_rows += imported.skipped_rows
            except Exception as exc:
                logger.exception(
                    "Football-Data fetch failed for league=%s season=%s",
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

    try:
        with SessionLocal() as db:
            processed = HistoricalFixtureRepository(db).upsert_many(fixture_rows)
            DataQualityService(db).finish_sync(
                sync_run_id,
                processed=processed,
                failures=cast(list[dict[str, object]], failures),
            )
    except Exception as exc:
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                sync_run_id,
                processed=0,
                failures=cast(list[dict[str, object]], failures),
                error_type=type(exc).__name__,
            )
        raise

    result: dict[str, object] = {
        "seasons": target_seasons,
        "fixtures_processed": processed,
        "skipped_incomplete_rows": skipped_rows,
        "failed_league_seasons": failures,
    }
    if fallback_from_season is not None:
        result["fallback_from_season"] = fallback_from_season
    logger.info("Football-Data fixture synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.retrain_ml_model_task")
def retrain_ml_model_task() -> str:
    """Asynchronously triggers model retraining on Celery workers."""
    logger.info("Initializing background ML model retraining job...")

    with SessionLocal() as db:
        repo = MatchPredictionRepository(db)
        labeled_rows = repo.get_all_labeled()
        historical_fixtures = HistoricalFixtureRepository(db).get_all()
        player_context_repo = PlayerContextRepository(db)
        historical_rows = HistoricalTrainingDataBuilder().build(
            historical_fixtures,
            player_performances=player_context_repo.get_all_performances(),
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
        # Celery workers do not execute the FastAPI lifespan hook.
        ml_pipeline.status()
        success = ml_pipeline.train_pipeline(training_rows)

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
