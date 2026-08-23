from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.team_identity import normalize_team_name, team_name_variants
from app.db.models import HistoricalFixture


class FixtureReadinessService:
    """Evaluate fixture history coverage without running the prediction stack."""

    _HISTORY_ROWS_PER_TARGET = 40

    def __init__(self, db: Session, *, required_matches: int | None = None) -> None:
        self.db = db
        self.required_matches = required_matches or settings.RECENT_FORM_MATCH_COUNT

    def annotate(self, fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parsed = [
            (fixture, self._kickoff(fixture.get("kickoff"))) for fixture in fixtures
        ]
        valid_kickoffs = [kickoff for _fixture, kickoff in parsed if kickoff is not None]
        if not valid_kickoffs:
            return [self._with_unknown_readiness(fixture) for fixture in fixtures]

        rows = self._load_relevant_history(parsed, before=max(valid_kickoffs))
        index = self._build_index(rows)
        return [
            self._annotate_fixture(fixture, kickoff, index)
            if kickoff is not None
            else self._with_unknown_readiness(fixture)
            for fixture, kickoff in parsed
        ]

    def _load_relevant_history(
        self,
        parsed: list[tuple[dict[str, Any], datetime | None]],
        *,
        before: datetime,
    ) -> list[HistoricalFixture]:
        team_ids: set[int] = set()
        names: set[str] = set()
        for fixture, kickoff in parsed:
            if kickoff is None:
                continue
            for id_field, name_field in (
                ("home_team_id", "home_team"),
                ("away_team_id", "away_team"),
            ):
                team_id = fixture.get(id_field)
                if isinstance(team_id, int) and not isinstance(team_id, bool):
                    team_ids.add(team_id)
                team_name = fixture.get(name_field)
                if isinstance(team_name, str):
                    names.update(team_name_variants(team_name))

        predicates: list[Any] = []
        if team_ids:
            predicates.extend(
                (
                    HistoricalFixture.home_team_id.in_(team_ids),
                    HistoricalFixture.away_team_id.in_(team_ids),
                )
            )
        if names:
            predicates.extend(
                (
                    func.lower(HistoricalFixture.home_team).in_(names),
                    func.lower(HistoricalFixture.away_team).in_(names),
                )
            )
        if not predicates:
            return []

        # Readiness only needs a short recent form window. The multiplier leaves
        # room for duplicate providers and alias matching without full-table scans.
        limit = max(
            self._HISTORY_ROWS_PER_TARGET,
            (len(team_ids) + len(names)) * self._HISTORY_ROWS_PER_TARGET,
        )
        targeted_rows = (
            self.db.query(HistoricalFixture)
            .filter(HistoricalFixture.kickoff < before, or_(*predicates))
            .order_by(HistoricalFixture.kickoff.desc(), HistoricalFixture.fixture_id.desc())
            .limit(limit)
            .all()
        )
        # SQLite's LOWER() is ASCII-only, so non-ASCII provider names cannot
        # reliably be matched by the alias predicate above. A bounded recent
        # fallback preserves cross-provider readiness without a full-table read.
        fallback_limit = max(500, len(parsed) * self._HISTORY_ROWS_PER_TARGET)
        fallback_rows = (
            self.db.query(HistoricalFixture)
            .filter(HistoricalFixture.kickoff < before)
            .order_by(HistoricalFixture.kickoff.desc(), HistoricalFixture.fixture_id.desc())
            .limit(fallback_limit)
            .all()
        )
        rows_by_fixture_id = {
            row.fixture_id: row for row in (*targeted_rows, *fallback_rows)
        }
        return list(rows_by_fixture_id.values())

    def _annotate_fixture(
        self,
        fixture: dict[str, Any],
        kickoff: datetime,
        index: dict[str, Any],
    ) -> dict[str, Any]:
        league_id = fixture.get("league_id")
        home_id = fixture.get("home_team_id")
        away_id = fixture.get("away_team_id")
        if (
            not isinstance(league_id, int)
            or isinstance(league_id, bool)
            or not isinstance(home_id, int)
            or isinstance(home_id, bool)
            or not isinstance(away_id, int)
            or isinstance(away_id, bool)
        ):
            return self._with_unknown_readiness(fixture)

        home_dates = self._history_dates(
            index,
            league_id,
            home_id,
            str(fixture.get("home_team") or ""),
            kickoff,
        )
        away_dates = self._history_dates(
            index,
            league_id,
            away_id,
            str(fixture.get("away_team") or ""),
            kickoff,
        )

        home_count = self._count_before(home_dates, kickoff)
        away_count = self._count_before(away_dates, kickoff)
        sufficient = (
            home_count >= self.required_matches
            and away_count >= self.required_matches
        )
        reasons: list[str] = []
        if home_count < self.required_matches:
            reasons.append("home_history_insufficient")
        if away_count < self.required_matches:
            reasons.append("away_history_insufficient")
        return {
            **fixture,
            "data_readiness": {
                "status": "sufficient" if sufficient else "insufficient",
                "home_history_matches": home_count,
                "away_history_matches": away_count,
                "required_history_matches": self.required_matches,
                "reasons": reasons,
            },
        }

    @staticmethod
    def _build_index(rows: list[HistoricalFixture]) -> dict[str, Any]:
        dates_by_league_team: dict[tuple[int, int], list[datetime]] = defaultdict(list)
        dates_by_team: dict[int, list[datetime]] = defaultdict(list)
        dates_by_league_name: dict[tuple[int, str], list[datetime]] = defaultdict(list)
        dates_by_name: dict[str, list[datetime]] = defaultdict(list)
        for row in rows:
            teams = (
                (row.home_team_id, row.home_team),
                (row.away_team_id, row.away_team),
            )
            for team_id, team_name in teams:
                dates_by_league_team[(row.league_id, team_id)].append(row.kickoff)
                dates_by_team[team_id].append(row.kickoff)
                normalized_name = normalize_team_name(team_name)
                if normalized_name:
                    dates_by_league_name[(row.league_id, normalized_name)].append(
                        row.kickoff
                    )
                    dates_by_name[normalized_name].append(row.kickoff)
        return {
            "dates_by_league_team": dates_by_league_team,
            "dates_by_team": dates_by_team,
            "dates_by_league_name": dates_by_league_name,
            "dates_by_name": dates_by_name,
        }

    def _history_dates(
        self,
        index: dict[str, Any],
        league_id: int,
        team_id: int,
        team_name: str,
        kickoff: datetime,
    ) -> list[datetime]:
        normalized_name = normalize_team_name(team_name)
        if normalized_name:
            league_dates = index["dates_by_league_name"].get(
                (league_id, normalized_name), []
            )
            if self._count_before(league_dates, kickoff) >= self.required_matches:
                return league_dates
            # Promoted teams and cup participants need form from other competitions.
            all_dates = index["dates_by_name"].get(normalized_name, [])
            if all_dates:
                return all_dates

        league_dates = index["dates_by_league_team"].get((league_id, team_id), [])
        if self._count_before(league_dates, kickoff) >= self.required_matches:
            return league_dates
        return index["dates_by_team"].get(team_id, league_dates)

    def _count_before(self, dates: list[datetime], before: datetime) -> int:
        return min(
            self.required_matches,
            len({self._as_utc(value) for value in dates if self._as_utc(value) < before}),
        )

    def _with_unknown_readiness(self, fixture: dict[str, Any]) -> dict[str, Any]:
        return {
            **fixture,
            "data_readiness": {
                "status": "unknown",
                "home_history_matches": 0,
                "away_history_matches": 0,
                "required_history_matches": self.required_matches,
                "reasons": ["fixture_identity_incomplete"],
            },
        }

    @classmethod
    def _kickoff(cls, value: object) -> datetime | None:
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

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
