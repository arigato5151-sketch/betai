from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from app.core.team_identity import normalize_team_name
from app.db.models import HistoricalFixture
from app.providers.understat import UnderstatFixtureXG


@dataclass(frozen=True, slots=True)
class XGMatchResult:
    updates: tuple[dict[str, object], ...]
    unmatched_provider_ids: tuple[str, ...]
    ambiguous_provider_ids: tuple[str, ...]


def match_understat_xg(
    historical_fixtures: Iterable[HistoricalFixture],
    provider_fixtures: Iterable[UnderstatFixtureXG],
    *,
    tolerance_hours: int,
) -> XGMatchResult:
    """Match xG conservatively by teams, final score and closest kickoff."""
    if tolerance_hours < 1 or tolerance_hours > 48:
        raise ValueError("tolerance_hours must be between 1 and 48")

    candidates: dict[tuple[str, str, int, int], list[HistoricalFixture]] = {}
    for fixture in historical_fixtures:
        key = (
            normalize_team_name(fixture.home_team),
            normalize_team_name(fixture.away_team),
            fixture.home_goals,
            fixture.away_goals,
        )
        candidates.setdefault(key, []).append(fixture)

    tolerance = timedelta(hours=tolerance_hours)
    updates: list[dict[str, object]] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    used_fixture_ids: set[int] = set()
    for observation in provider_fixtures:
        key = (
            normalize_team_name(observation.home_team),
            normalize_team_name(observation.away_team),
            observation.home_goals,
            observation.away_goals,
        )
        ranked: list[tuple[timedelta, HistoricalFixture]] = []
        for fixture in candidates.get(key, []):
            kickoff = _as_utc(fixture.kickoff)
            distance = abs(kickoff - observation.kickoff)
            if distance <= tolerance and fixture.fixture_id not in used_fixture_ids:
                ranked.append((distance, fixture))
        ranked.sort(key=lambda item: (item[0], item[1].fixture_id))
        if not ranked:
            unmatched.append(observation.provider_match_id)
            continue
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            ambiguous.append(observation.provider_match_id)
            continue

        fixture = ranked[0][1]
        used_fixture_ids.add(fixture.fixture_id)
        updates.append(
            {
                "fixture_id": fixture.fixture_id,
                "home_xg": observation.home_xg,
                "away_xg": observation.away_xg,
                "xg_source": "understat",
                "xg_provider_match_id": observation.provider_match_id,
            }
        )
    return XGMatchResult(
        updates=tuple(updates),
        unmatched_provider_ids=tuple(unmatched),
        ambiguous_provider_ids=tuple(ambiguous),
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
