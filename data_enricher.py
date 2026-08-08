"""Rate-limited API-Football fallback for missing Football-Data statistics."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import pandas as pd
import requests


class FootballAPIConfigurationError(RuntimeError):
    """Raised when an enrichment request is attempted without an API key."""


class FootballAPIResponseError(RuntimeError):
    """Raised when API-Football returns an invalid statistics payload."""


@dataclass
class FootballDataEnricher:
    """Fill missing shot, corner and foul columns without overwriting CSV values.

    API-Football's statistics endpoint requires its own fixture identifier.  The
    input frame must therefore include ``fixture_id`` or ``FixtureId`` for rows
    that need enrichment. Rows without that identifier are deliberately skipped.
    """

    base_url: str = "https://v3.football.api-sports.io"
    min_request_interval_seconds: float = 1.0
    timeout_seconds: float = 20.0
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_request_at: float | None = field(default=None, init=False, repr=False)

    _TARGET_COLUMNS = ("HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF")
    _STATISTICS_MAP = {
        "Total Shots": "S",
        "Shots on Goal": "ST",
        "Corner Kicks": "C",
        "Fouls": "F",
    }

    def enrich_missing_statistics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with missing statistics filled from API-Football.

        A request is made only when home shots (``HS``) or home corners (``HC``)
        are missing.  Existing statistics stay authoritative, so the API is a
        fallback rather than a second source that can silently rewrite data.
        """
        self._validate_frame(dataframe)
        enriched = dataframe.copy(deep=True)
        missing_mask = enriched["HS"].isna() | enriched["HC"].isna()
        if not bool(missing_mask.any()):
            return enriched

        api_key = os.getenv("FOOTBALL_API_KEY", "").strip()
        if not api_key:
            raise FootballAPIConfigurationError(
                "FOOTBALL_API_KEY must be configured before enrichment"
            )

        fixture_column = self._fixture_id_column(enriched)
        if fixture_column is None:
            return enriched

        for index in enriched.index[missing_mask]:
            fixture_id = self._fixture_id(enriched.at[index, fixture_column])
            if fixture_id is None:
                continue
            statistics = self.fetch_fixture_statistics(fixture_id, api_key=api_key)
            for column, value in statistics.items():
                if column in enriched.columns and pd.isna(enriched.at[index, column]):
                    enriched.at[index, column] = value
        return enriched

    def fetch_fixture_statistics(
        self, fixture_id: int, *, api_key: str | None = None
    ) -> dict[str, int]:
        """Fetch and normalize one API-Football fixture statistics response."""
        if fixture_id <= 0:
            raise ValueError("fixture_id must be a positive API-Football identifier")
        configured_key = (api_key or os.getenv("FOOTBALL_API_KEY") or "").strip()
        if not configured_key:
            raise FootballAPIConfigurationError(
                "FOOTBALL_API_KEY must be configured before enrichment"
            )

        self._wait_for_rate_limit()
        response = requests.get(
            f"{self.base_url.rstrip('/')}/fixtures/statistics",
            headers={"x-apisports-key": configured_key},
            params={"fixture": fixture_id},
            timeout=self.timeout_seconds,
        )
        self._last_request_at = self.clock()
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise FootballAPIResponseError(
                "API-Football returned invalid JSON"
            ) from exc
        return self._parse_statistics(payload)

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        remaining = self.min_request_interval_seconds - (
            self.clock() - self._last_request_at
        )
        if remaining > 0:
            self.sleep(remaining)

    @classmethod
    def _parse_statistics(cls, payload: object) -> dict[str, int]:
        if not isinstance(payload, Mapping):
            raise FootballAPIResponseError("API-Football payload must be an object")
        response_rows = payload.get("response")
        if not isinstance(response_rows, list) or len(response_rows) < 2:
            raise FootballAPIResponseError("Fixture statistics are unavailable")

        normalized: dict[str, dict[str, int]] = {}
        for row in response_rows:
            if not isinstance(row, Mapping):
                continue
            team = row.get("team")
            statistics = row.get("statistics")
            if not isinstance(team, Mapping) or not isinstance(statistics, list):
                continue
            side = "H" if team.get("id") == cls._home_team_id(response_rows) else "A"
            for entry in statistics:
                if not isinstance(entry, Mapping):
                    continue
                suffix = cls._STATISTICS_MAP.get(str(entry.get("type") or ""))
                value = cls._as_non_negative_int(entry.get("value"))
                if suffix is not None and value is not None:
                    normalized.setdefault(side, {})[suffix] = value

        if "H" not in normalized or "A" not in normalized:
            raise FootballAPIResponseError(
                "Fixture response does not include both teams"
            )
        return {
            f"H{suffix}": normalized["H"][suffix]
            for suffix in normalized["H"]
            if f"H{suffix}" in cls._TARGET_COLUMNS
        } | {
            f"A{suffix}": normalized["A"][suffix]
            for suffix in normalized["A"]
            if f"A{suffix}" in cls._TARGET_COLUMNS
        }

    @staticmethod
    def _home_team_id(rows: list[object]) -> object:
        """API-Football lists home statistics first; keep the provider order explicit."""
        first = rows[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("team"), Mapping):
            raise FootballAPIResponseError("Fixture response has no home team")
        return first["team"].get("id")

    @staticmethod
    def _as_non_negative_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _fixture_id(value: object) -> int | None:
        try:
            fixture_id = int(str(value))
        except (TypeError, ValueError):
            return None
        return fixture_id if fixture_id > 0 else None

    @staticmethod
    def _fixture_id_column(dataframe: pd.DataFrame) -> str | None:
        return next(
            (column for column in ("fixture_id", "FixtureId") if column in dataframe),
            None,
        )

    @classmethod
    def _validate_frame(cls, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        required = {"HS", "HC"}
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
