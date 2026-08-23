from __future__ import annotations

import logging
from typing import Any, cast

from celery import shared_task

from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS
from app.core.config import settings
from app.core.team_identity import normalize_team_name
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import PlayerContextRepository
from app.db.session import SessionLocal
from app.services.api_football import APIFootballClient
from app.services.data_quality import DataQualityService
from app.services.fixture_aggregator import FixtureAggregator
from app.services.fixture_download import FixtureDownloadClient, UEFA_FEEDS
from app.services.fixture_readiness import FixtureReadinessService
from app.services.football_data_csv import FootballDataCSVClient, FootballDataDownloadError
from app.services.openfootball_json import OpenFootballJSONClient
from app.services.task_lock import DistributedTaskLock
from app.providers.statsbomb_open import StatsBombOpenDataClient, database_row
from app.tasks.base import TransientTask
from app.tasks._helpers import (
    _current_football_season,
    _fetch_thesportsdb_team_history,
    _missing_fixture_history_scopes,
    _missing_fixture_team_targets,
    _missing_api_team_targets,
    _run_async,
)

logger = logging.getLogger("bet-ai-pro.tasks")


@shared_task(
    name="app.tasks.jobs.sync_missing_fixture_history_task",
    base=TransientTask,
)
def sync_missing_fixture_history_task(
    fixtures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Fill detected gaps from free feeds and bounded paid-plan team history."""
    if fixtures is None:
        aggregator = FixtureAggregator()
        discovered = _run_async(aggregator.get_upcoming_fixtures(days=7, limit=100))
        with SessionLocal() as db:
            fixtures = FixtureReadinessService(db).annotate(discovered)

    scopes = _missing_fixture_history_scopes(fixtures)
    team_targets = _missing_fixture_team_targets(fixtures)
    api_team_targets = (
        _missing_api_team_targets(fixtures)
        if settings.API_FOOTBALL_PLAN != "free"
        else []
    )
    client = FootballDataCSVClient()
    supported = [scope for scope in scopes if scope[0] in client.supported_league_ids]
    unsupported = [
        {"league_id": league_id, "season": season}
        for league_id, season in scopes
        if league_id not in client.supported_league_ids
    ]
    if not supported and not team_targets and not api_team_targets:
        return {
            "status": "no_gaps",
            "fixtures_processed": 0,
            "unsupported_scopes": unsupported,
        }

    scope_key = "-".join(f"{league_id}:{season}" for league_id, season in supported)
    target_key = "-".join(normalize_team_name(name) for name, _before in team_targets)
    api_target_key = "-".join(str(team_id) for team_id, _name in api_team_targets)
    with DistributedTaskLock(
        f"missing-fixture-history:{scope_key}:{target_key}:{api_target_key}",
        ttl_seconds=900,
    ) as lock:
        if not lock.acquired:
            return {"status": "already_running", "fixtures_processed": 0}

        rows: list[dict[str, object]] = []
        failures: list[dict[str, int | str]] = []
        pending_scopes: list[dict[str, int | str]] = []
        for league_id, season in supported:
            try:
                imported = _run_async(client.get_completed_fixtures(league_id, season))
                rows.extend(imported.fixtures)
            except FootballDataDownloadError as exc:
                issue: dict[str, int | str] = {
                    "league_id": league_id,
                    "season": season,
                    "error": type(exc).__name__,
                }
                if exc.status_code == 404:
                    # Current-season CSVs are legitimately absent until published.
                    pending_scopes.append(issue)
                else:
                    failures.append(issue)
            except Exception as exc:
                logger.exception(
                    "On-demand history fetch failed for league=%s season=%s",
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

        thesportsdb_rows, team_failures = _run_async(
            _fetch_thesportsdb_team_history(team_targets)
        )
        rows.extend(thesportsdb_rows)

        api_team_failures: list[dict[str, int | str]] = []
        api_client = APIFootballClient()
        for team_id, team_name in api_team_targets[:40]:
            try:
                rows.extend(
                    _run_async(
                        api_client.get_team_recent_completed_fixtures(
                            team_id,
                            last=settings.RECENT_FORM_MATCH_COUNT,
                        )
                    )
                )
            except Exception as exc:
                logger.exception(
                    "API-Football recent history failed for team=%s (%s)",
                    team_id,
                    team_name,
                )
                api_team_failures.append(
                    {
                        "team_id": team_id,
                        "team": team_name,
                        "error": type(exc).__name__,
                    }
                )

        with SessionLocal() as db:
            fixture_rows = []
            for row in rows:
                fixture_row = dict(row)
                # Player rows belong to their dedicated repository; this task
                # only needs completed fixtures to establish team form.
                fixture_row.pop("player_performances", None)
                fixture_rows.append(fixture_row)
            processed = HistoricalFixtureRepository(db).upsert_many(fixture_rows)

    result: dict[str, object] = {
        "status": (
            "partial"
            if failures or team_failures or api_team_failures
            else "completed"
        ),
        "fixtures_processed": processed,
        "requested_scopes": [
            {"league_id": league_id, "season": season}
            for league_id, season in supported
        ],
        "unsupported_scopes": unsupported,
        "pending_scopes": pending_scopes,
        "failures": failures,
        "team_targets": [name for name, _before in team_targets],
        "team_failures": team_failures,
        "api_team_targets": [name for _team_id, name in api_team_targets[:40]],
        "api_team_failures": api_team_failures,
    }
    logger.info("On-demand fixture history synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_historical_fixtures_task", base=TransientTask)
def sync_historical_fixtures_task(
    seasons: list[int] | None = None,
    league_ids: list[int] | None = None,
    enrich_player_context: bool = True,
) -> dict:
    """Ingest a validated league scope; pass seasons for a repeatable backfill."""
    if not settings.API_FOOTBALL_HISTORICAL_SYNC_ENABLED:
        return {
            "status": "disabled",
            "reason": "api_football_historical_sync_disabled",
        }

    with DistributedTaskLock("sync-historical-fixtures", ttl_seconds=7200) as lock:
        if not lock.acquired:
            return {"status": "locked"}
        return _sync_historical_fixtures_impl(
            seasons=seasons,
            league_ids=league_ids,
            enrich_player_context=enrich_player_context,
        )


def _sync_historical_fixtures_impl(
    *,
    seasons: list[int] | None,
    league_ids: list[int] | None,
    enrich_player_context: bool,
) -> dict:
    target_seasons: list[int] = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    allowed_ids = cast(set[int], ALLOWED_LEAGUE_IDS)
    requested_ids = league_ids if league_ids is not None else list(allowed_ids)
    if not requested_ids or any(
        not isinstance(league_id, int) or isinstance(league_id, bool)
        for league_id in requested_ids
    ):
        raise ValueError("league_ids must contain at least one integer league ID")
    unsupported_ids = set(requested_ids) - allowed_ids
    if unsupported_ids:
        raise ValueError(f"Unsupported league_ids: {sorted(unsupported_ids)}")
    target_league_ids = sorted(set(requested_ids))

    with SessionLocal() as db:
        sync_run_id = (
            DataQualityService(db).start_sync("historical_fixtures", target_seasons).id
        )

    api_client = APIFootballClient()
    fixture_rows: list[dict] = []
    failures: list[dict[str, int | str]] = []

    for season in target_seasons:
        for league_id in target_league_ids:
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
        player_context_failures = 0
        if enrich_player_context:
            from app.tasks.predictions import _enrich_historical_player_context

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


@shared_task(name="app.tasks.jobs.sync_football_data_fixtures_task", base=TransientTask)
def sync_football_data_fixtures_task(seasons: list[int] | None = None) -> dict:
    """Import completed league fixtures from the public Football-Data CSV feeds."""
    automatic_season = seasons is None
    target_seasons = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    client = FootballDataCSVClient()

    with SessionLocal() as db:
        sync_run_id = (
            DataQualityService(db).start_sync("football_data_csv", target_seasons).id
        )

    fixture_rows: list[dict] = []
    skipped_rows = 0
    failures: list[dict[str, int | str]] = []
    imported_seasons: set[int] = set()
    league_season_fallbacks: list[dict[str, int]] = []

    for league_id in sorted(client.supported_league_ids):
        for requested_season in target_seasons:
            effective_season = requested_season
            try:
                try:
                    imported = _run_async(
                        client.get_completed_fixtures(league_id, requested_season)
                    )
                except FootballDataDownloadError as exc:
                    if not automatic_season or exc.status_code != 404:
                        raise
                    effective_season = requested_season - 1
                    imported = _run_async(
                        client.get_completed_fixtures(league_id, effective_season)
                    )
                    league_season_fallbacks.append(
                        {
                            "league_id": league_id,
                            "from_season": requested_season,
                            "to_season": effective_season,
                        }
                    )
                    logger.info(
                        "Football-Data league=%s season=%s is not published; "
                        "falling back to %s for this league only.",
                        league_id,
                        requested_season,
                        effective_season,
                    )
                fixture_rows.extend(imported.fixtures)
                skipped_rows += imported.skipped_rows
                imported_seasons.add(effective_season)
            except Exception as exc:
                logger.exception(
                    "Football-Data fetch failed for league=%s season=%s",
                    league_id,
                    requested_season,
                )
                failures.append(
                    {
                        "league_id": league_id,
                        "season": requested_season,
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
        "seasons": sorted(imported_seasons),
        "fixtures_processed": processed,
        "skipped_incomplete_rows": skipped_rows,
        "failed_league_seasons": failures,
    }
    if league_season_fallbacks:
        result["league_season_fallbacks"] = league_season_fallbacks
    logger.info("Football-Data fixture synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_openfootball_fixtures_task", base=TransientTask)
def sync_openfootball_fixtures_task(seasons: list[int] | None = None) -> dict:
    """Import completed fixtures from the public-domain OpenFootball datasets."""
    if not settings.OPENFOOTBALL_ENABLED:
        return {"status": "disabled", "fixtures_processed": 0}

    target_seasons = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    client = OpenFootballJSONClient()
    with SessionLocal() as db:
        sync_run_id = (
            DataQualityService(db).start_sync("openfootball_json", target_seasons).id
        )

    fixture_rows: list[dict[str, object]] = []
    failures: list[dict[str, int | str]] = []
    for league_id in sorted(client.supported_league_ids):
        for season in target_seasons:
            try:
                fixture_rows.extend(
                    _run_async(client.get_completed_fixtures(league_id, season))
                )
            except Exception as exc:
                logger.exception(
                    "OpenFootball fetch failed for league=%s season=%s",
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
        "failed_league_seasons": failures,
    }
    logger.info("OpenFootball fixture synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_uefa_fixtures_task", base=TransientTask)
def sync_uefa_fixtures_task(seasons: list[int] | None = None) -> dict[str, object]:
    """Import completed UEFA fixtures from the public JSON result feeds."""
    target_seasons = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    with SessionLocal() as db:
        run_id = (
            DataQualityService(db)
            .start_sync("fixture_download_uefa", target_seasons)
            .id
        )

    client = FixtureDownloadClient()
    fixture_rows: list[dict[str, Any]] = []
    failures: list[dict[str, object]] = []
    for season in target_seasons:
        for league_id in UEFA_FEEDS:
            try:
                fixture_rows.extend(
                    _run_async(client.get_completed_fixtures(league_id, season))
                )
            except Exception as exc:
                logger.exception(
                    "UEFA fixture fetch failed for league=%s season=%s",
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
                run_id,
                processed=processed,
                failures=failures,
            )
    except Exception as exc:
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                run_id,
                processed=0,
                failures=failures,
                error_type=type(exc).__name__,
            )
        raise
    result: dict[str, object] = {
        "seasons": target_seasons,
        "fixtures_processed": processed,
        "failed_league_seasons": failures,
    }
    logger.info("UEFA fixture synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_statsbomb_open_data_task", base=TransientTask)
def sync_statsbomb_open_data_task() -> dict[str, object]:
    """Import public match metadata, then incrementally enrich events and xG."""
    if not settings.STATSBOMB_OPEN_DATA_ENABLED:
        return {"status": "disabled", "fixtures_processed": 0}

    client = StatsBombOpenDataClient()
    with SessionLocal() as db:
        run_id = (
            DataQualityService(db)
            .start_sync(
                "statsbomb_open",
                [settings.STATSBOMB_OPEN_DATA_MIN_SEASON],
            )
            .id
        )
    metadata_processed = enriched_processed = 0
    failures: list[dict[str, object]] = []
    try:
        catalog = _run_async(
            client.get_catalog(min_season=settings.STATSBOMB_OPEN_DATA_MIN_SEASON)
        )
        catalog_by_id = {int(row["fixture_id"]): row for row in catalog}
        with SessionLocal() as db:
            from app.db.models import HistoricalFixture

            existing_ids = {
                fixture_id
                for (fixture_id,) in db.query(HistoricalFixture.fixture_id)
                .filter(HistoricalFixture.data_source == "statsbomb_open")
                .all()
            }
            new_rows = [
                database_row(row)
                for fixture_id, row in catalog_by_id.items()
                if fixture_id not in existing_ids
            ]
            metadata_processed = HistoricalFixtureRepository(db).upsert_many(new_rows)

        with SessionLocal() as db:
            from app.db.models import HistoricalFixture

            pending_ids = [
                fixture_id
                for (fixture_id,) in db.query(HistoricalFixture.fixture_id)
                .filter(
                    HistoricalFixture.data_source == "statsbomb_open",
                    HistoricalFixture.xg_source.is_(None),
                )
                .order_by(HistoricalFixture.kickoff.desc())
                .limit(settings.STATSBOMB_OPEN_DATA_ENRICH_BATCH_SIZE)
                .all()
            ]
        pending = [
            catalog_by_id[fixture_id]
            for fixture_id in pending_ids
            if fixture_id in catalog_by_id
        ]
        enriched, failed_ids = _run_async(client.enrich_matches(pending))
        failures.extend(
            {"fixture_id": fixture_id, "error": "EventEnrichmentFailed"}
            for fixture_id in failed_ids
        )
        with SessionLocal() as db:
            enriched_processed = HistoricalFixtureRepository(db).upsert_many(
                database_row(row) for row in enriched
            )
            DataQualityService(db).finish_sync(
                run_id,
                processed=metadata_processed + enriched_processed,
                failures=failures,
            )
    except Exception as exc:
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                run_id,
                processed=metadata_processed + enriched_processed,
                failures=failures,
                error_type=type(exc).__name__,
            )
        raise
    result: dict[str, object] = {
        "status": "partial" if failures else "completed",
        "catalog_matches": len(catalog),
        "metadata_processed": metadata_processed,
        "events_enriched": enriched_processed,
        "failed_events": len(failures),
    }
    logger.info("StatsBomb Open Data synchronization completed: %s", result)
    return result
