from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, date, datetime
from typing import cast

from celery import shared_task

from app.core.allowed_leagues import ALLOWED_LEAGUES
from app.core.config import settings
from app.core.team_identity import normalize_team_name
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import PlayerContextRepository
from app.db.session import SessionLocal
from app.db.models import HistoricalFixture, TeamLocation
from app.providers.open_meteo import OpenMeteoClient
from app.providers.understat import UnderstatClient
from app.providers.wikidata import WikidataError, WikidataTeamLocationClient
from app.providers.geonames_city import GeoNamesCityResolver
from app.services.api_football import APIFootballClient
from app.services.data_quality import DataQualityService
from app.services.derived_xg import DerivedXGService
from app.services.understat_xg import match_understat_xg
from app.tasks.base import TransientTask
from app.tasks._helpers import _current_football_season, _run_async

logger = logging.getLogger("bet-ai-pro.tasks")


@shared_task(name="app.tasks.jobs.sync_understat_xg_task", base=TransientTask)
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


@shared_task(name="app.tasks.jobs.derive_historical_xg_task", base=TransientTask)
def derive_historical_xg_task() -> dict[str, object]:
    """Fill non-observed xG only after the holdout quality gate passes."""
    if not settings.DERIVED_XG_ENABLED:
        return {"status": "disabled", "fixtures_updated": 0}

    with SessionLocal() as db:
        repository = HistoricalFixtureRepository(db)
        observed = repository.get_recent_observed_xg(
            settings.DERIVED_XG_TRAINING_FIXTURE_LIMIT
        )
        pending = repository.get_missing_xg(settings.DERIVED_XG_UPDATE_FIXTURE_LIMIT)
        fixtures = {
            fixture.fixture_id: fixture for fixture in (*observed, *pending)
        }.values()
        result = DerivedXGService().build_updates(fixtures)
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


@shared_task(
    name="app.tasks.jobs.sync_wikidata_team_locations_task", base=TransientTask
)
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
        targets = PlayerContextRepository(db).list_missing_team_location_targets(
            seasons=target_seasons,
            limit=settings.WIKIDATA_LOCATION_MAX_TEAMS,
        )
    candidates = [
        {
            **target,
            "country": (
                None
                if countries.get(cast(int, target["league_id"])) == "Europe"
                else countries.get(cast(int, target["league_id"]))
            ),
        }
        for target in targets
    ]
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


@shared_task(name="app.tasks.jobs.sync_free_team_locations_task", base=TransientTask)
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
    if offset is not None and (
        isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
    ):
        raise ValueError("offset must be a non-negative integer")
    batch_limit = settings.FREE_TEAM_LOCATION_MAX_TEAMS
    scan_offset = (
        offset
        if offset is not None
        else (date.today().toordinal() // 7) * batch_limit
    )
    with SessionLocal() as db:
        repository = PlayerContextRepository(db)
        targets = repository.list_missing_team_location_targets(
            seasons=target_seasons,
            limit=batch_limit,
            offset=scan_offset,
        )
        if not targets and scan_offset:
            # The weekly cursor can pass the current missing-team count after
            # successful runs; wrap without loading the entire fixture table.
            scan_offset = 0
            targets = repository.list_missing_team_location_targets(
                seasons=target_seasons,
                limit=batch_limit,
            )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for target in targets:
        country = countries.get(cast(int, target["league_id"]))
        identity = (
            normalize_team_name(str(target["name"])),
            "" if country == "Europe" else str(country or ""),
        )
        grouped.setdefault(identity, []).append(
            {
                "data_source": target["data_source"],
                "team_id": target["team_id"],
                "name": target["name"],
            }
        )

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
        identities = ordered[:batch_limit]
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
        "scan_offset": scan_offset,
    }
    logger.info("Free team location synchronization completed: %s", result)
    return result


@shared_task(name="app.tasks.jobs.sync_open_meteo_weather_task", base=TransientTask)
def sync_open_meteo_weather_task() -> dict[str, object]:
    """Incrementally add match-time weather where a home venue is known."""
    if not settings.OPEN_METEO_ENABLED:
        return {"status": "disabled", "processed": 0, "failed": 0}

    with SessionLocal() as db:
        run_id = DataQualityService(db).start_sync("open_meteo", []).id
        candidates = (
            db.query(
                HistoricalFixture.fixture_id,
                TeamLocation.latitude,
                TeamLocation.longitude,
                HistoricalFixture.kickoff,
            )
            .join(
                TeamLocation,
                (TeamLocation.data_source == HistoricalFixture.data_source)
                & (TeamLocation.team_id == HistoricalFixture.home_team_id),
            )
            .filter(
                HistoricalFixture.weather_source.is_(None),
                HistoricalFixture.kickoff < datetime.now(UTC),
                TeamLocation.latitude.is_not(None),
                TeamLocation.longitude.is_not(None),
            )
            .order_by(HistoricalFixture.kickoff.desc())
            .limit(settings.OPEN_METEO_BACKFILL_BATCH_SIZE)
            .all()
        )

    requests = [
        (fixture_id, float(latitude), float(longitude), kickoff)
        for fixture_id, latitude, longitude, kickoff in candidates
        if latitude is not None and longitude is not None
    ]
    processed = 0
    failures: list[dict[str, object]] = []
    try:
        observations, failed_ids = _run_async(OpenMeteoClient().get_many(requests))
        failures = [
            {"fixture_id": fixture_id, "error": "WeatherLookupFailed"}
            for fixture_id in failed_ids
        ]
        updates = [
            {
                "fixture_id": fixture_id,
                "weather_temperature_c": observation.temperature_c,
                "weather_precipitation_mm": observation.precipitation_mm,
                "weather_wind_speed_kmh": observation.wind_speed_kmh,
                "weather_source": observation.source,
                "weather_observed_at": observation.observed_at,
                "weather_updated_at": observation.fetched_at,
            }
            for fixture_id, observation in observations.items()
        ]
        with SessionLocal() as db:
            processed = HistoricalFixtureRepository(db).update_weather_many(updates)
            DataQualityService(db).finish_sync(
                run_id,
                processed=processed,
                failures=failures,
            )
    except Exception as exc:
        with SessionLocal() as db:
            DataQualityService(db).finish_sync(
                run_id,
                processed=processed,
                failures=failures,
                error_type=type(exc).__name__,
            )
        raise
    result: dict[str, object] = {
        "status": "partial" if failures else "completed",
        "processed": processed,
        "failed": len(failures),
        "candidates": len(requests),
    }
    logger.info("Open-Meteo synchronization completed: %s", result)
    return result
