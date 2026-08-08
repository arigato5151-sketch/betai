"""Resolve Football-Data rows to API-Football fixture identifiers safely."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests


class FixtureResolverConfigurationError(RuntimeError):
    """Raised when fixture resolution needs an unavailable API key."""


class FixtureResolverResponseError(RuntimeError):
    """Raised when API-Football returns an invalid fixture response."""


class FixtureResolver:
    """Map date/team triples to API-Football IDs with bounded fuzzy matching."""

    _NAME_ALIASES = {
        "man city": "manchester city",
        "inter": "internazionale",
        "inter milan": "internazionale",
        "fc internazionale": "internazionale",
    }

    def __init__(
        self,
        *,
        cache_path: Path | str = "fixture_cache.json",
        base_url: str = "https://v3.football.api-sports.io",
        threshold: float = 0.80,
        date_tolerance_days: int = 1,
        min_request_interval_seconds: float = 1.0,
        timeout_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in the interval (0, 1]")
        if date_tolerance_days < 0:
            raise ValueError("date_tolerance_days cannot be negative")
        self.cache_path = Path(cache_path)
        self.base_url = base_url.rstrip("/")
        self.threshold = threshold
        self.date_tolerance_days = date_tolerance_days
        self.min_request_interval_seconds = min_request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self.sleep = sleep
        self._fixture_cache = self._load_cache()
        self._daily_candidates: dict[date, list[dict[str, object]]] = {}
        self._last_request_at: float | None = None

    def resolve_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with safely resolved ``fixture_id`` values.

        Existing identifiers are preserved. Invalid dates and weak matches remain
        unset, preventing a similarly named but unrelated match from being joined.
        """
        self._validate_dataframe(dataframe)
        resolved = dataframe.copy(deep=True)
        if "fixture_id" not in resolved:
            resolved["fixture_id"] = pd.Series(
                pd.NA, index=resolved.index, dtype="Int64"
            )

        for index, row in resolved.iterrows():
            if pd.notna(row["fixture_id"]):
                continue
            match_date = self.parse_football_data_date(row["Date"])
            if match_date is None:
                continue
            fixture_id = self.resolve_fixture_id(
                match_date,
                home_team=str(row["HomeTeam"] or ""),
                away_team=str(row["AwayTeam"] or ""),
            )
            if fixture_id is not None:
                resolved.at[index, "fixture_id"] = fixture_id
        return resolved

    def resolve_fixture_id(
        self, match_date: date, *, home_team: str, away_team: str
    ) -> int | None:
        """Resolve one Football-Data match, using JSON cache before the network."""
        if not home_team.strip() or not away_team.strip():
            return None
        cache_key = self._cache_key(match_date, home_team, away_team)
        cached = self._fixture_cache.get(cache_key)
        if cached is not None:
            return cached

        best_fixture_id: int | None = None
        best_score = 0.0
        for offset in range(-self.date_tolerance_days, self.date_tolerance_days + 1):
            candidate_date = match_date + timedelta(days=offset)
            for fixture in self._get_fixtures_for_date(candidate_date):
                fixture_id, api_home, api_away, api_date = self._fixture_identity(
                    fixture
                )
                if (
                    fixture_id is None
                    or api_home is None
                    or api_away is None
                    or api_date is None
                    or abs((api_date - match_date).days) > self.date_tolerance_days
                ):
                    continue
                home_score = self.name_similarity(home_team, api_home)
                away_score = self.name_similarity(away_team, api_away)
                score = min(home_score, away_score)
                if score >= self.threshold and score > best_score:
                    best_fixture_id, best_score = fixture_id, score

        if best_fixture_id is not None:
            self._fixture_cache[cache_key] = best_fixture_id
            self._save_cache()
        return best_fixture_id

    @staticmethod
    def parse_football_data_date(value: object) -> date | None:
        """Parse Football-Data's DD/MM/YYYY and DD/MM/YY date variants."""
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        raw = str(value or "").strip()
        for format_string in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(raw, format_string).date()
            except ValueError:
                continue
        return None

    @classmethod
    def name_similarity(cls, football_data_name: str, api_name: str) -> float:
        """Return a normalized SequenceMatcher score after known name aliases."""
        left = cls._canonical_team_name(football_data_name)
        right = cls._canonical_team_name(api_name)
        if not left or not right:
            return 0.0
        return SequenceMatcher(a=left, b=right, autojunk=False).ratio()

    def _get_fixtures_for_date(self, fixture_date: date) -> list[dict[str, object]]:
        cached = self._daily_candidates.get(fixture_date)
        if cached is not None:
            return cached
        api_key = (os.getenv("FOOTBALL_API_KEY") or "").strip()
        if not api_key:
            raise FixtureResolverConfigurationError(
                "FOOTBALL_API_KEY must be configured before fixture resolution"
            )

        self._wait_for_rate_limit()
        response = requests.get(
            f"{self.base_url}/fixtures",
            headers={"x-apisports-key": api_key},
            params={"date": fixture_date.isoformat()},
            timeout=self.timeout_seconds,
        )
        self._last_request_at = self.clock()
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise FixtureResolverResponseError(
                "API-Football returned invalid JSON"
            ) from exc
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("response"), list
        ):
            raise FixtureResolverResponseError(
                "API-Football fixture payload is invalid"
            )
        fixtures = [
            candidate
            for candidate in payload["response"]
            if isinstance(candidate, dict)
        ]
        self._daily_candidates[fixture_date] = fixtures
        return fixtures

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.min_request_interval_seconds - (
            self.clock() - self._last_request_at
        )
        if remaining > 0:
            self.sleep(remaining)

    @classmethod
    def _fixture_identity(
        cls, fixture: Mapping[str, object]
    ) -> tuple[int | None, str | None, str | None, date | None]:
        metadata = fixture.get("fixture")
        teams = fixture.get("teams")
        if not isinstance(metadata, Mapping) or not isinstance(teams, Mapping):
            return None, None, None, None
        home = teams.get("home")
        away = teams.get("away")
        if not isinstance(home, Mapping) or not isinstance(away, Mapping):
            return None, None, None, None
        try:
            fixture_id = int(str(metadata.get("id")))
        except (TypeError, ValueError):
            fixture_id = None
        api_date = cls._parse_api_date(metadata.get("date"))
        return (
            fixture_id if fixture_id and fixture_id > 0 else None,
            cls._string_value(home.get("name")),
            cls._string_value(away.get("name")),
            api_date,
        )

    @staticmethod
    def _parse_api_date(value: object) -> date | None:
        raw = str(value or "").strip()
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _string_value(value: object) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @classmethod
    def _canonical_team_name(cls, name: str) -> str:
        tokens = [
            token
            for token in "".join(
                character.lower() if character.isalnum() else " " for character in name
            ).split()
            if token not in {"fc", "ac", "afc", "cf", "sc"}
        ]
        normalized = " ".join(tokens)
        return cls._NAME_ALIASES.get(normalized, normalized)

    @classmethod
    def _cache_key(cls, match_date: date, home_team: str, away_team: str) -> str:
        return "|".join(
            (
                match_date.isoformat(),
                cls._canonical_team_name(home_team),
                cls._canonical_team_name(away_team),
            )
        )

    def _load_cache(self) -> dict[str, int]:
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, Mapping):
            return {}
        return {
            str(key): fixture_id
            for key, value in payload.items()
            if (fixture_id := self._positive_int(value)) is not None
        }

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(
                self._fixture_cache, ensure_ascii=False, sort_keys=True, indent=2
            ),
            encoding="utf-8",
        )
        temporary_path.replace(self.cache_path)

    @staticmethod
    def _positive_int(value: object) -> int | None:
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        required = {"Date", "HomeTeam", "AwayTeam"}
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
