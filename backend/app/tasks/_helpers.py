from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime

from sqlalchemy import func, select, union_all

from app.core.config import settings
from app.core.team_identity import normalize_team_name, team_name_variants
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.session import SessionLocal
from app.db.models import HistoricalFixture
from app.services.fixture_aggregator import TheSportsDBFixtureSource

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


def _missing_fixture_history_scopes(
    fixtures: list[dict[str, object]],
) -> list[tuple[int, int]]:
    """Find league/seasons where an upcoming team has no local match history."""
    valid_fixtures: list[tuple[int, int, datetime, str, str]] = []
    for fixture in fixtures[:200]:
        league_id = fixture.get("league_id")
        season = fixture.get("season")
        kickoff = _fixture_kickoff(fixture.get("kickoff"))
        home = normalize_team_name(str(fixture.get("home_team") or ""))
        away = normalize_team_name(str(fixture.get("away_team") or ""))
        if (
            isinstance(league_id, int)
            and not isinstance(league_id, bool)
            and isinstance(season, int)
            and not isinstance(season, bool)
            and kickoff is not None
            and home
            and away
        ):
            valid_fixtures.append((league_id, season, kickoff, home, away))

    latest_by_league: dict[int, datetime] = {}
    for league_id, _season, kickoff, _home, _away in valid_fixtures:
        latest_by_league[league_id] = max(
            kickoff, latest_by_league.get(league_id, kickoff)
        )

    known_by_league: dict[int, set[str]] = {}
    with SessionLocal() as db:
        repository = HistoricalFixtureRepository(db)
        for league_id, before in latest_by_league.items():
            history = repository.get_recent_league_history(
                league_id=league_id,
                before=before,
                limit=settings.FIXTURE_HISTORY_SCOPE_LOOKBACK_FIXTURES,
            )
            known_by_league[league_id] = {
                normalize_team_name(team_name)
                for row in history
                for team_name in (row.home_team, row.away_team)
            }

    scopes: set[tuple[int, int]] = set()
    for league_id, season, _kickoff, home, away in valid_fixtures:
        known = known_by_league.get(league_id, set())
        if home not in known or away not in known:
            scopes.update(((league_id, season), (league_id, season - 1)))
    return sorted(scopes)


def _missing_fixture_team_targets(
    fixtures: list[dict[str, object]],
) -> list[tuple[str, datetime]]:
    """Find teams with no history in any competition, keyed by canonical name."""
    candidates: dict[str, tuple[str, datetime]] = {}
    provider_names: set[str] = set()
    for fixture in fixtures[:200]:
        kickoff = _fixture_kickoff(fixture.get("kickoff"))
        if kickoff is None:
            continue
        for field in ("home_team", "away_team"):
            display_name = str(fixture.get(field) or "").strip()
            canonical = normalize_team_name(display_name)
            if not canonical:
                continue
            provider_names.update(team_name_variants(display_name))
            current = candidates.get(canonical)
            if current is None or kickoff < current[1]:
                candidates[canonical] = (display_name, kickoff)
    if not candidates:
        return []

    with SessionLocal() as db:
        home_names = select(HistoricalFixture.home_team.label("name")).where(
            func.lower(HistoricalFixture.home_team).in_(provider_names)
        )
        away_names = select(HistoricalFixture.away_team.label("name")).where(
            func.lower(HistoricalFixture.away_team).in_(provider_names)
        )
        fixture_names = union_all(home_names, away_names).subquery()
        names = db.execute(select(fixture_names.c.name).distinct()).scalars().all()
    known = {
        normalize_team_name(team_name)
        for team_name in names
        if isinstance(team_name, str) and team_name
    }
    return [target for canonical, target in candidates.items() if canonical not in known]


def _missing_api_team_targets(
    fixtures: list[dict[str, object]],
) -> list[tuple[int, str]]:
    """Return only provider teams explicitly marked history-insufficient."""
    targets: dict[int, str] = {}
    for fixture in fixtures[:200]:
        readiness = fixture.get("data_readiness")
        if not isinstance(readiness, dict):
            continue
        reasons = readiness.get("reasons")
        if not isinstance(reasons, list):
            continue
        for side in ("home", "away"):
            if f"{side}_history_insufficient" not in reasons:
                continue
            team_id = fixture.get(f"{side}_team_id")
            team_name = str(fixture.get(f"{side}_team") or "").strip()
            if (
                isinstance(team_id, int)
                and not isinstance(team_id, bool)
                and 0 < team_id < 100_000_000
                and team_name
            ):
                targets.setdefault(team_id, team_name)
    return list(targets.items())


async def _fetch_thesportsdb_team_history(
    targets: list[tuple[str, datetime]],
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    source = TheSportsDBFixtureSource()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, (team_name, before) in enumerate(targets[:10]):
        try:
            rows.extend(
                await source.get_team_history(
                    team_name,
                    before=before,
                    limit=settings.RECENT_FORM_MATCH_COUNT,
                )
            )
        except Exception as exc:
            logger.warning("TheSportsDB history unavailable for %s: %s", team_name, exc)
            failures.append({"team": team_name, "error": type(exc).__name__})
        if index < min(len(targets), 10) - 1:
            await asyncio.sleep(0.75)
    return rows, failures
