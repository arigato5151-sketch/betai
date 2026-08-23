from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Awaitable, Callable, cast

from celery import shared_task

from app.core.config import settings
from app.db.odds_snapshot_repository import OddsSnapshotRepository
from app.db.player_context_repository import is_fixture_player_context_complete
from app.db.session import SessionLocal
from app.db.models import MatchPrediction
from app.prediction.eligibility import (
    PredictionEligibilityDecision,
    PredictionIneligibleError,
)
from app.services.api_football import APIFootballClient
from app.services.api_provider_health import api_football_health
from app.services.cloudflare_odds_collector import (
    CloudflareOddsCollectorError,
    cloudflare_odds_collector,
)
from app.services.fixture_aggregator import FixtureAggregator
from app.services.fixture_context import fixture_context_service
from app.services.odds_history import OddsHistoryService, odds_history_service
from app.services.task_lock import DistributedTaskLock
from app.tasks.base import TransientTask
from app.tasks._helpers import _fixture_kickoff, _run_async

logger = logging.getLogger("bet-ai-pro.tasks")


async def _generate_upcoming_predictions(
    aggregator: FixtureAggregator,
    analyzer: Callable[[int], Awaitable[object]],
    existing_fixture_ids: set[int],
    *,
    observed_at: datetime | None = None,
) -> dict[str, object]:
    """Generate one prediction per eligible fixture while keeping reruns idempotent."""
    started_at = (observed_at or datetime.now(UTC)).astimezone(UTC)
    fixtures = await aggregator.get_upcoming_fixtures(
        days=settings.AUTO_PREDICTION_HORIZON_DAYS,
        limit=settings.AUTO_PREDICTION_MAX_FIXTURES,
    )
    earliest_kickoff = started_at + timedelta(
        minutes=settings.AUTO_PREDICTION_MIN_LEAD_MINUTES
    )
    candidates: list[int] = []
    skipped_existing = 0
    skipped_invalid = 0
    for fixture in fixtures:
        fixture_id = fixture.get("fixture_id")
        kickoff = _fixture_kickoff(fixture.get("kickoff"))
        if (
            isinstance(fixture_id, bool)
            or not isinstance(fixture_id, int)
            or fixture_id <= 0
            or kickoff is None
            or kickoff <= earliest_kickoff
            or fixture.get("is_demo") is True
        ):
            skipped_invalid += 1
            continue
        if fixture_id in existing_fixture_ids:
            skipped_existing += 1
            continue
        candidates.append(fixture_id)

    semaphore = asyncio.Semaphore(settings.AUTO_PREDICTION_CONCURRENCY)

    async def analyze(fixture_id: int) -> str:
        try:
            async with semaphore:
                result = await analyzer(fixture_id)
            if (
                isinstance(result, Mapping)
                and result.get("decision_status") == "abstain"
            ):
                logger.info(
                    "Automatic prediction decision abstained for fixture_id=%s reasons=%s",
                    fixture_id,
                    result.get("decision_reasons", []),
                )
                return "abstained"
            if (
                isinstance(result, Mapping)
                and result.get("decision_status") == "conditional"
            ):
                logger.info(
                    "Automatic prediction conditional for fixture_id=%s confidence_tier=%s",
                    fixture_id,
                    result.get("confidence_tier", "unknown"),
                )
                return "conditional"
            return "generated"
        except PredictionIneligibleError as exc:
            logger.info(
                "Automatic prediction abstained for fixture_id=%s reasons=%s",
                fixture_id,
                exc.decision.reasons,
            )
            return "abstained"
        except Exception:
            logger.exception(
                "Automatic prediction failed for fixture_id=%s", fixture_id
            )
            return "failed"

    outcomes = await asyncio.gather(*(analyze(fixture_id) for fixture_id in candidates))
    counts = Counter(outcomes)
    return {
        "status": "partial" if counts["failed"] else "succeeded",
        "fixtures_seen": len(fixtures),
        "eligible_fixtures": len(candidates),
        "predictions_generated": counts["generated"],
        "abstained": counts["abstained"],
        "conditional": counts.get("conditional", 0),
        "failed": counts["failed"],
        "skipped_existing": skipped_existing,
        "skipped_invalid": skipped_invalid,
        "started_at": started_at.isoformat(),
    }


async def _analyze_upcoming_fixture(
    aggregator: FixtureAggregator,
    fixture_id: int,
) -> object:
    # Local import prevents the task module and API router from importing each other.
    from app.api.endpoints import _build_payload_from_prefill, _run_analysis

    prefill = await fixture_context_service.get_or_create(
        fixture_id,
        loader=aggregator.get_fixture_prefill,
        enricher=odds_history_service.enrich_prefill,
    )
    _validate_automatic_prefill(prefill)
    prefill = cast(dict[str, object], prefill)
    payload = _build_payload_from_prefill(prefill)
    return await _run_analysis(
        payload,
        require_eligible=True,
        analysis_origin="automatic",
        interactive=False,
    )


def _validate_automatic_prefill(prefill: object) -> None:
    """Convert incomplete provider context into a safe abstention."""
    reasons: tuple[str, ...] = ()
    if not isinstance(prefill, dict):
        reasons = ("fixture_prefill_unavailable",)
    else:
        raw_odd = prefill.get("odd")
        if isinstance(raw_odd, bool) or not isinstance(raw_odd, (int, float, str)):
            odd = math.nan
        else:
            try:
                odd = float(raw_odd)
            except ValueError:
                odd = math.nan
        if not math.isfinite(odd) or odd <= 1.0:
            reasons = ("market_unavailable",)

    if reasons:
        raise PredictionIneligibleError(
            PredictionEligibilityDecision(
                eligible=False,
                status="abstain",
                reasons=reasons,
                data_quality_score=0.0,
            )
        )


def _run_upcoming_prediction_batch() -> dict[str, object]:
    if not settings.AUTO_PREDICTION_ENABLED:
        return {"status": "disabled"}

    now = datetime.now(UTC)
    horizon_end = now + timedelta(days=settings.AUTO_PREDICTION_HORIZON_DAYS)
    with SessionLocal() as db:
        existing_fixture_ids = {
            fixture_id
            for (fixture_id,) in db.query(MatchPrediction.fixture_id)
            .filter(
                MatchPrediction.fixture_id.isnot(None),
                MatchPrediction.kickoff >= now,
                MatchPrediction.kickoff <= horizon_end,
            )
            .all()
        }

    aggregator = FixtureAggregator()

    async def analyzer(fixture_id: int) -> object:
        return await _analyze_upcoming_fixture(aggregator, fixture_id)

    result = _run_async(
        _generate_upcoming_predictions(
            aggregator,
            analyzer,
            existing_fixture_ids,
        )
    )
    logger.info("Automatic prediction generation completed: %s", result)
    return cast(dict[str, object], result)


@shared_task(
    name="app.tasks.jobs.generate_upcoming_predictions_task", base=TransientTask
)
def generate_upcoming_predictions_task() -> dict[str, object]:
    """Periodically analyze new upcoming fixtures and persist their predictions."""
    with DistributedTaskLock(
        "generate_upcoming_predictions",
        ttl_seconds=settings.AUTO_PREDICTION_LOCK_TTL_SECONDS,
    ) as task_lock:
        if not task_lock.acquired:
            return {
                "status": (
                    "lock_unavailable" if not task_lock.available else "already_running"
                )
            }
        return _run_upcoming_prediction_batch()


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

    due_candidates = [
        (fixture_id, kickoff)
        for fixture_id, kickoff in candidates
        if service.should_collect(
            fixture_id=fixture_id,
            kickoff=kickoff,
            observed_at=captured_at,
            refresh_interval_seconds=settings.ODDS_COLLECTOR_RUN_INTERVAL_SECONDS,
            closing_window_hours=settings.ODDS_COLLECTOR_CLOSING_WINDOW_HOURS,
        )
    ]
    due_candidates.sort(key=lambda candidate: candidate[1])

    provider_health = await api_football_health.snapshot()
    daily_remaining = provider_health.get("daily_remaining")
    market_budget = settings.ODDS_COLLECTOR_MARKET_REQUEST_BUDGET
    if isinstance(daily_remaining, int) and not isinstance(daily_remaining, bool):
        market_budget = min(
            market_budget,
            max(0, daily_remaining - settings.ODDS_COLLECTOR_DAILY_QUOTA_RESERVE),
        )
    selected_candidates = due_candidates[:market_budget]
    quota_deferred = len(due_candidates) - len(selected_candidates)
    semaphore = asyncio.Semaphore(settings.ODDS_COLLECTOR_CONCURRENCY)

    async def collect(fixture_id: int, kickoff: datetime) -> str:
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
        *(collect(fixture_id, kickoff) for fixture_id, kickoff in selected_candidates)
    )
    counts = Counter(outcomes)
    return {
        "status": "succeeded",
        "fixtures_seen": len(fixtures),
        "eligible_fixtures": len(candidates),
        "snapshots_recorded": counts["recorded"],
        "not_due": len(candidates) - len(due_candidates),
        "quota_deferred": quota_deferred,
        "market_request_budget": market_budget,
        "market_unavailable": counts["market_unavailable"],
        "rejected": counts["rejected"],
        "invalid_fixtures": invalid_fixtures,
        "captured_at": captured_at.isoformat(),
    }


@shared_task(name="app.tasks.jobs.collect_upcoming_odds_task", base=TransientTask)
def collect_upcoming_odds_task() -> dict[str, object]:
    """Periodically build opening/current 1X2 pairs for upcoming fixtures."""
    if not settings.ODDS_COLLECTOR_ENABLED:
        return {"status": "disabled"}
    api_client = APIFootballClient()
    if api_client._is_demo_key():
        return {"status": "demo_disabled"}
    with DistributedTaskLock("collect-upcoming-odds", ttl_seconds=900) as lock:
        if not lock.acquired:
            return {"status": "locked"}
        result = _run_async(_collect_upcoming_odds(api_client, odds_history_service))
    logger.info("Upcoming odds collection completed: %s", result)
    return cast(dict[str, object], result)


@shared_task(
    name="app.tasks.jobs.sync_cloudflare_odds_snapshots_task",
    base=TransientTask,
)
def sync_cloudflare_odds_snapshots_task() -> dict[str, object]:
    """Import immutable remote snapshots after the local machine comes online."""
    if not cloudflare_odds_collector.enabled:
        return {"status": "disabled"}
    try:
        rows = _run_async(cloudflare_odds_collector.snapshots(limit=250))
    except CloudflareOddsCollectorError as exc:
        logger.warning("Cloudflare odds snapshot sync failed: %s", exc)
        return {"status": "remote_unavailable", "imported": 0}

    imported = 0
    rejected = 0
    with SessionLocal() as db:
        odds_repo = OddsSnapshotRepository(db)
        for row in rows:
            tracked_id = row.get("tracked_fixture_id")
            if not isinstance(tracked_id, str) or not tracked_id.startswith("fixture:"):
                rejected += 1
                continue
            try:
                fixture_id = int(tracked_id.removeprefix("fixture:"))
                captured_at = datetime.fromisoformat(
                    str(row["captured_at"]).replace("Z", "+00:00")
                )
                raw_details = row.get("details_json")
                provider_details = (
                    json.loads(raw_details)
                    if isinstance(raw_details, str) and raw_details
                    else {}
                )
                details = {
                    "remote_snapshot_id": row.get("id"),
                    "snapshot_kind": row.get("snapshot_kind"),
                    "payload_sha256": row.get("payload_sha256"),
                    "provider_updated_at": row.get("provider_updated_at"),
                    "provider_details": provider_details,
                }
                odds_repo.record(
                    fixture_id=fixture_id,
                    raw_odds={
                        "HOME_WIN": row["home_odd"],
                        "DRAW": row["draw_odd"],
                        "AWAY_WIN": row["away_odd"],
                    },
                    captured_at=captured_at,
                    source="cloudflare_api_football_odds",
                    bookmaker=(str(row["bookmaker"]) if row.get("bookmaker") else None),
                    minimum_interval_seconds=0,
                    details=details,
                )
                imported += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                rejected += 1
                continue
    return {
        "status": "succeeded",
        "remote_rows": len(rows),
        "imported": imported,
        "rejected": rejected,
    }


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


@shared_task(name="app.tasks.jobs.collect_upcoming_lineups_task", base=TransientTask)
def collect_upcoming_lineups_task() -> dict[str, object]:
    """Periodically cache official lineups shortly before kickoff."""
    if not settings.LINEUP_COLLECTOR_ENABLED:
        return {"status": "disabled"}
    api_client = APIFootballClient()
    if api_client._is_demo_key():
        return {"status": "demo_disabled"}
    with DistributedTaskLock("collect-upcoming-lineups", ttl_seconds=900) as lock:
        if not lock.acquired:
            return {"status": "locked"}
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
