import asyncio
import logging
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, cast
from celery import shared_task
from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS, ALLOWED_LEAGUES
from app.core.config import settings
from app.core.team_identity import normalize_team_name
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import (
    PlayerContextRepository,
    is_fixture_player_context_complete,
)
from app.db.session import SessionLocal
from app.db.repository import MatchPredictionRepository
from app.db.models import HistoricalFixture, MatchPrediction
from app.services.api_football import APIFootballClient
from app.services.football_data_csv import (
    FootballDataCSVClient,
    FootballDataDownloadError,
)
from app.services.fixture_download import FixtureDownloadClient, UEFA_FEEDS
from app.providers.understat import UnderstatClient
from app.providers.wikidata import WikidataError, WikidataTeamLocationClient
from app.providers.geonames_city import GeoNamesCityResolver
from app.services.understat_xg import match_understat_xg
from app.services.odds_history import OddsHistoryService, odds_history_service
from app.prediction.ml.model import ml_pipeline
from app.prediction.ml.training_data import HistoricalTrainingDataBuilder
from app.prediction.ensemble_weights import ensemble_weight_manager
from app.prediction.audit import PredictionAuditor
from app.services.data_quality import DataQualityService
from app.services.derived_xg import DerivedXGService
from app.services.model_monitoring import ModelMonitoringService

logger = logging.getLogger("bet-ai-pro.tasks")


def _run_async(coro):
    """Utility helper to run async coroutines in synchronous Celery task threads."""
    return asyncio.run(coro)


def _current_football_season(today: date | None = None) -> int:
    current = today or date.today()
    return current.year if current.month >= 7 else current.year - 1


def _fixture_kickoff(value: object) -> datetime | None:
    if not isinstance(value, (str, datetime)):
        return None
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


async def _collect_upcoming_odds(
    client: APIFootballClient,
    service: OddsHistoryService,
    *,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    """Collect bounded opening/near-kickoff snapshots without quota-heavy polling."""
    captured_at = (observed_at or datetime.now(UTC)).astimezone(UTC)
    fixtures = await client.get_upcoming_fixtures(
        days=settings.ODDS_COLLECTOR_HORIZON_DAYS,
        limit=settings.ODDS_COLLECTOR_MAX_FIXTURES,
    )
    semaphore = asyncio.Semaphore(settings.ODDS_COLLECTOR_CONCURRENCY)
    invalid_fixtures = 0
    candidates: list[tuple[int, datetime]] = []
    for fixture in fixtures:
        fixture_id = fixture.get("fixture_id")
        kickoff = _fixture_kickoff(fixture.get("kickoff"))
        if (
            isinstance(fixture_id, bool)
            or not isinstance(fixture_id, int)
            or fixture_id <= 0
            or kickoff is None
            or fixture.get("is_demo") is True
        ):
            invalid_fixtures += 1
            continue
        candidates.append((fixture_id, kickoff))

    async def collect(fixture_id: int, kickoff: datetime) -> str:
        if not service.should_collect(
            fixture_id=fixture_id,
            kickoff=kickoff,
            observed_at=captured_at,
            refresh_interval_seconds=settings.ODDS_COLLECTOR_RUN_INTERVAL_SECONDS,
            closing_window_hours=settings.ODDS_COLLECTOR_CLOSING_WINDOW_HOURS,
        ):
            return "not_due"
        async with semaphore:
            market = await client.get_fixture_market(fixture_id)
        if not isinstance(market, Mapping):
            return "market_unavailable"
        enriched = service.enrich_prefill(
            {
                "fixture": {
                    "fixture_id": fixture_id,
                    "kickoff": kickoff.isoformat(),
                },
                "market_1x2": dict(market),
            },
            captured_at=captured_at,
        )
        return "recorded" if "odds_history" in enriched else "rejected"

    outcomes = await asyncio.gather(
        *(collect(fixture_id, kickoff) for fixture_id, kickoff in candidates)
    )
    counts = Counter(outcomes)
    return {
        "status": "succeeded",
        "fixtures_seen": len(fixtures),
        "eligible_fixtures": len(candidates),
        "snapshots_recorded": counts["recorded"],
        "not_due": counts["not_due"],
        "market_unavailable": counts["market_unavailable"],
        "rejected": counts["rejected"],
        "invalid_fixtures": invalid_fixtures,
        "captured_at": captured_at.isoformat(),
    }


@shared_task(name="app.tasks.jobs.collect_upcoming_odds_task")
def collect_upcoming_odds_task() -> dict[str, object]:
    """Periodically build opening/current 1X2 pairs for upcoming fixtures."""
    if not settings.ODDS_COLLECTOR_ENABLED:
        return {"status": "disabled"}
    api_client = APIFootballClient()
    if api_client._is_demo_key():
        return {"status": "demo_disabled"}
    result = _run_async(_collect_upcoming_odds(api_client, odds_history_service))
    logger.info("Upcoming odds collection completed: %s", result)
    return cast(dict[str, object], result)


async def _collect_upcoming_lineups(
    client: APIFootballClient,
    *,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    """Warm confirmed lineup cache only inside the configured pre-kickoff window."""
    captured_at = (observed_at or datetime.now(UTC)).astimezone(UTC)
    fixtures = await client.get_upcoming_fixtures(
        days=settings.LINEUP_COLLECTOR_HORIZON_DAYS,
        limit=settings.LINEUP_COLLECTOR_MAX_FIXTURES,
    )
    semaphore = asyncio.Semaphore(settings.LINEUP_COLLECTOR_CONCURRENCY)
    invalid_fixtures = 0
    outside_window = 0
    candidates: list[tuple[int, int, int]] = []
    window_seconds = settings.LINEUP_COLLECTOR_WINDOW_MINUTES * 60
    for fixture in fixtures:
        fixture_id = fixture.get("fixture_id")
        home_team_id = fixture.get("home_team_id")
        away_team_id = fixture.get("away_team_id")
        kickoff = _fixture_kickoff(fixture.get("kickoff"))
        identifiers = (fixture_id, home_team_id, away_team_id)
        if (
            any(
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or identifier <= 0
                for identifier in identifiers
            )
            or kickoff is None
            or fixture.get("is_demo") is True
        ):
            invalid_fixtures += 1
            continue
        seconds_to_kickoff = (kickoff - captured_at).total_seconds()
        if not 0 < seconds_to_kickoff <= window_seconds:
            outside_window += 1
            continue
        candidates.append(
            (
                cast(int, fixture_id),
                cast(int, home_team_id),
                cast(int, away_team_id),
            )
        )

    async def collect(
        fixture_id: int,
        home_team_id: int,
        away_team_id: int,
    ) -> str:
        async with semaphore:
            lineups = await client.get_fixture_lineups(
                fixture_id,
                home_team_id,
                away_team_id,
            )
        if not isinstance(lineups, Mapping):
            return "unavailable"
        home_starting_xi = lineups.get("home_starting_xi")
        away_starting_xi = lineups.get("away_starting_xi")
        confirmed = (
            isinstance(home_starting_xi, list)
            and len(home_starting_xi) == 11
            and isinstance(away_starting_xi, list)
            and len(away_starting_xi) == 11
        )
        return "confirmed" if confirmed else "unavailable"

    outcomes = await asyncio.gather(
        *(
            collect(fixture_id, home_team_id, away_team_id)
            for fixture_id, home_team_id, away_team_id in candidates
        )
    )
    counts = Counter(outcomes)
    return {
        "status": "succeeded",
        "fixtures_seen": len(fixtures),
        "eligible_fixtures": len(candidates),
        "lineups_confirmed": counts["confirmed"],
        "lineups_unavailable": counts["unavailable"],
        "outside_window": outside_window,
        "invalid_fixtures": invalid_fixtures,
        "captured_at": captured_at.isoformat(),
    }


@shared_task(name="app.tasks.jobs.collect_upcoming_lineups_task")
def collect_upcoming_lineups_task() -> dict[str, object]:
    """Periodically cache official lineups shortly before kickoff."""
    if not settings.LINEUP_COLLECTOR_ENABLED:
        return {"status": "disabled"}
    api_client = APIFootballClient()
    if api_client._is_demo_key():
        return {"status": "demo_disabled"}
    result = _run_async(_collect_upcoming_lineups(api_client))
    logger.info("Upcoming lineup collection completed: %s", result)
    return cast(dict[str, object], result)


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
def sync_historical_fixtures_task(
    seasons: list[int] | None = None,
    league_ids: list[int] | None = None,
    enrich_player_context: bool = True,
) -> dict:
    """Ingest a validated league scope; pass seasons for a repeatable backfill."""
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

    # Bulk fixture network work completes before opening a database transaction.
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


@shared_task(name="app.tasks.jobs.sync_understat_xg_task")
def sync_understat_xg_task(seasons: list[int] | None = None) -> dict[str, object]:
    """Enrich top-five-league historical fixtures with match-level xG."""
    if not settings.UNDERSTAT_ENABLED:
        return {"status": "disabled", "fixtures_updated": 0}

    target_seasons = sorted(
        set(seasons if seasons is not None else [_current_football_season()])
    )
    client = UnderstatClient()
    with SessionLocal() as db:
        sync_run_id = (
            DataQualityService(db).start_sync("understat_xg", target_seasons).id
        )

    fetched = updated = unmatched = ambiguous = 0
    failures: list[dict[str, object]] = []
    request_count = len(client.supported_league_ids) * len(target_seasons)
    request_index = 0
    try:
        for league_id in sorted(client.supported_league_ids):
            for season in target_seasons:
                try:
                    observations = _run_async(
                        client.get_completed_fixture_xg(league_id, season)
                    )
                    fetched += len(observations)
                    with SessionLocal() as db:
                        repository = HistoricalFixtureRepository(db)
                        historical = repository.get_league_history(
                            league_id=league_id,
                            season=season,
                            before=datetime.now(UTC),
                        )
                        matches = match_understat_xg(
                            historical,
                            observations,
                            tolerance_hours=settings.UNDERSTAT_MATCH_TOLERANCE_HOURS,
                        )
                        updated += repository.update_xg_many(matches.updates)
                    unmatched += len(matches.unmatched_provider_ids)
                    ambiguous += len(matches.ambiguous_provider_ids)
                except Exception as exc:
                    logger.exception(
                        "Understat xG sync failed for league=%s season=%s",
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
                finally:
                    request_index += 1
                    if request_index < request_count:
                        time.sleep(settings.UNDERSTAT_REQUEST_INTERVAL_SECONDS)
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                sync_run_id,
                processed=updated,
                failures=failures,
            )
    except Exception as exc:
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                sync_run_id,
                processed=updated,
                failures=failures,
                error_type=type(exc).__name__,
            )
        raise

    result: dict[str, object] = {
        "status": "partial" if failures else "completed",
        "seasons": target_seasons,
        "fixtures_fetched": fetched,
        "fixtures_updated": updated,
        "unmatched_fixtures": unmatched,
        "ambiguous_fixtures": ambiguous,
        "failed_league_seasons": failures,
    }
    logger.info("Understat xG synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.derive_historical_xg_task")
def derive_historical_xg_task() -> dict[str, object]:
    """Fill non-observed xG only after the holdout quality gate passes."""
    if not settings.DERIVED_XG_ENABLED:
        return {"status": "disabled", "fixtures_updated": 0}

    with SessionLocal() as db:
        repository = HistoricalFixtureRepository(db)
        result = DerivedXGService().build_updates(repository.get_all())
        updated = (
            repository.update_xg_many(result.updates) if result.status == "ready" else 0
        )
    response: dict[str, object] = {
        "status": result.status,
        "training_matches": result.training_matches,
        "fixtures_updated": updated,
        "holdout_mae": result.holdout_mae,
        "baseline_mae": result.baseline_mae,
        "holdout_r2": result.holdout_r2,
    }
    logger.info("Derived xG synchronization completed: %s", response)
    return response


async def _sync_wikidata_team_locations(
    client: WikidataTeamLocationClient,
    teams: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    semaphore = asyncio.Semaphore(settings.WIKIDATA_LOCATION_CONCURRENCY)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for team in teams:
        key = (
            normalize_team_name(str(team["name"])),
            str(team.get("country") or ""),
        )
        grouped.setdefault(key, []).append(team)

    async def resolve(
        identity: tuple[str, str],
        targets: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        async with semaphore:
            try:
                location = await client.resolve(
                    team_name=identity[0],
                    country=identity[1] or None,
                )
            except WikidataError:
                logger.warning(
                    "Wikidata team location lookup failed",
                    extra={"team_name": identity[0]},
                )
                return []
        if location is None:
            return []
        return [
            {
                "data_source": team["data_source"],
                "team_id": team["team_id"],
                "name": team["name"],
                "latitude": location.latitude,
                "longitude": location.longitude,
                "location_source": "wikidata",
                "confidence": location.confidence,
                "details": {
                    "club_qid": location.club_qid,
                    "location_qid": location.location_qid,
                    "location_name": location.location_name,
                    "method": location.method,
                    "approximation": "home_venue_or_club_location",
                },
            }
            for team in targets
        ]

    resolved = await asyncio.gather(
        *(resolve(identity, targets) for identity, targets in grouped.items())
    )
    rows = [row for group in resolved for row in group]
    return rows, len(teams) - len(rows)


@shared_task(name="app.tasks.jobs.sync_wikidata_team_locations_task")
def sync_wikidata_team_locations_task(
    seasons: list[int] | None = None,
) -> dict[str, object]:
    """Backfill signed open-feed team IDs with verified venue coordinates."""
    if not settings.WIKIDATA_LOCATION_ENABLED:
        return {"status": "disabled", "locations_processed": 0}
    target_seasons = sorted(set(seasons or [2024, 2025]))
    countries = {
        cast(int, league["id"]): cast(str, league["country"])
        for league in ALLOWED_LEAGUES
    }
    with SessionLocal() as db:
        fixtures = (
            db.query(HistoricalFixture)
            .filter(HistoricalFixture.season.in_(target_seasons))
            .order_by(HistoricalFixture.kickoff.desc())
            .all()
        )
        repository = PlayerContextRepository(db)
        existing = {
            (row.data_source, row.team_id)
            for row in repository.get_all_team_locations()
            if row.latitude is not None and row.longitude is not None
        }

    teams_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for fixture in fixtures:
        source = str(fixture.data_source or "api_football").strip().lower()
        country = countries.get(fixture.league_id)
        for team_id, name in (
            (fixture.home_team_id, fixture.home_team),
            (fixture.away_team_id, fixture.away_team),
        ):
            key = (source, team_id)
            if key not in existing:
                teams_by_key.setdefault(
                    key,
                    {
                        "data_source": source,
                        "team_id": team_id,
                        "name": name,
                        "country": None if country == "Europe" else country,
                    },
                )
    candidates = list(teams_by_key.values())[: settings.WIKIDATA_LOCATION_MAX_TEAMS]
    processed = 0
    unresolved = 0
    batch_size = 40
    client = WikidataTeamLocationClient()
    for start in range(0, len(candidates), batch_size):
        rows, batch_unresolved = _run_async(
            _sync_wikidata_team_locations(
                client,
                candidates[start : start + batch_size],
            )
        )
        with SessionLocal() as db:
            processed += PlayerContextRepository(db).upsert_team_locations(rows)
        unresolved += batch_unresolved
    result: dict[str, object] = {
        "status": "completed",
        "seasons": target_seasons,
        "teams_considered": len(candidates),
        "locations_processed": processed,
        "unresolved_teams": unresolved,
    }
    logger.info("Wikidata team location synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_free_team_locations_task")
def sync_free_team_locations_task(
    seasons: list[int] | None = None,
    offset: int | None = None,
) -> dict[str, object]:
    """Use the free team directory plus offline GeoNames as a fallback."""
    target_seasons = sorted(set(seasons or [2024, 2025]))
    countries = {
        cast(int, league["id"]): cast(str, league["country"])
        for league in ALLOWED_LEAGUES
    }
    with SessionLocal() as db:
        fixtures = (
            db.query(HistoricalFixture)
            .filter(HistoricalFixture.season.in_(target_seasons))
            .order_by(HistoricalFixture.kickoff.desc())
            .all()
        )
        existing = {
            (row.data_source, row.team_id)
            for row in PlayerContextRepository(db).get_all_team_locations()
            if row.latitude is not None and row.longitude is not None
        }

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for fixture in fixtures:
        source = str(fixture.data_source or "api_football").strip().lower()
        country = countries.get(fixture.league_id)
        for team_id, name in (
            (fixture.home_team_id, fixture.home_team),
            (fixture.away_team_id, fixture.away_team),
        ):
            if (source, team_id) in existing:
                continue
            identity = (
                normalize_team_name(name),
                "" if country == "Europe" else str(country or ""),
            )
            targets = grouped.setdefault(identity, [])
            target = {"data_source": source, "team_id": team_id, "name": name}
            if target not in targets:
                targets.append(target)

    async def collect() -> tuple[list[dict[str, object]], int]:
        client = APIFootballClient()
        resolver = GeoNamesCityResolver()
        rows: list[dict[str, object]] = []
        unresolved = 0
        ordered = sorted(
            grouped.items(),
            key=lambda item: (not bool(item[0][1]), item[0][1], item[0][0]),
        )
        if not ordered:
            return rows, unresolved
        if offset is not None and (
            isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
        ):
            raise ValueError("offset must be a non-negative integer")
        batch_limit = settings.FREE_TEAM_LOCATION_MAX_TEAMS
        # Rotate the quota-safe window weekly so permanently unresolved aliases
        # cannot starve later clubs indefinitely.
        scan_offset = (
            offset
            if offset is not None
            else (date.today().toordinal() // 7) * batch_limit
        ) % len(ordered)
        rotated = ordered[scan_offset:] + ordered[:scan_offset]
        identities = rotated[:batch_limit]
        for (team_name, country), targets in identities:
            context = await client.search_team_venue_context(
                team_name,
                country=country or None,
            )
            if context is None:
                unresolved += len(targets)
                continue
            resolved = resolver.resolve(
                city=str(context["city"]),
                country=str(context["country"]),
            )
            if resolved is None:
                unresolved += len(targets)
                continue
            rows.extend(
                {
                    "data_source": target["data_source"],
                    "team_id": target["team_id"],
                    "name": target["name"],
                    "latitude": resolved.latitude,
                    "longitude": resolved.longitude,
                    "location_source": "api_football_geonames",
                    "confidence": resolved.confidence,
                    "details": {
                        "city": resolved.city,
                        "country_code": resolved.country_code,
                        "geoname_id": resolved.geoname_id,
                        "provider_team_id": context.get("team_id"),
                        "provider_team_name": context.get("team_name"),
                        "venue_id": context.get("venue_id"),
                        "venue_name": context.get("venue_name"),
                        "approximation": "city_centre",
                    },
                }
                for target in targets
            )
        return rows, unresolved

    rows, unresolved = _run_async(collect())
    with SessionLocal() as db:
        processed = PlayerContextRepository(db).upsert_team_locations(rows)
    result: dict[str, object] = {
        "status": "completed",
        "seasons": target_seasons,
        "identities_considered": min(
            len(grouped), settings.FREE_TEAM_LOCATION_MAX_TEAMS
        ),
        "locations_processed": processed,
        "unresolved_team_ids": unresolved,
        "scan_offset": (
            (
                offset
                if offset is not None
                else (date.today().toordinal() // 7)
                * settings.FREE_TEAM_LOCATION_MAX_TEAMS
            )
            % max(1, len(grouped))
        ),
    }
    logger.info("Free team location synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_uefa_fixtures_task")
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


@shared_task(name="app.tasks.jobs.monitor_model_drift_task")
def monitor_model_drift_task() -> dict[str, object]:
    """Queue a challenger training run when recent calibration materially degrades."""
    with SessionLocal() as db:
        status = ModelMonitoringService(db).snapshot()
    retraining_queued = bool(status["drift_detected"])
    if retraining_queued:
        retrain_ml_model_task.delay()
    return {**status, "retraining_queued": retraining_queued}


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
