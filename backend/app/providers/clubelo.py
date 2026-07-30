from __future__ import annotations

import asyncio
import csv
import io
import math
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from app.core.config import settings
from app.core.team_identity import stable_team_name_key
from app.providers.base import ExternalDataPoint, ProviderTeamCandidate

MAX_CLUBELO_PAYLOAD_BYTES = 5 * 1024 * 1024
REQUIRED_COLUMNS = frozenset({"Club", "Elo", "From", "To"})


class ClubEloError(RuntimeError):
    """Raised when ClubElo cannot be used safely."""


class ClubEloDownloadError(ClubEloError):
    """Raised when the provider is unavailable."""


class ClubEloFormatError(ClubEloError):
    """Raised when provider output is malformed or outside expected bounds."""


@dataclass(frozen=True, slots=True)
class _ClubEloRow:
    club: str
    elo: float
    country: str | None
    level: int | None
    valid_from: str
    valid_to: str


class ClubEloClient:
    """Fetch ClubElo ratings with exact, accent-insensitive team matching."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        cache_hours: int | None = None,
        confidence: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.CLUBELO_BASE_URL).strip().rstrip("/")
        self.timeout_seconds = timeout_seconds or settings.CLUBELO_TIMEOUT_SECONDS
        self.cache_seconds = float((cache_hours or settings.CLUBELO_CACHE_HOURS) * 3600)
        self.confidence = (
            float(confidence)
            if confidence is not None
            else float(settings.CLUBELO_CONFIDENCE)
        )
        self.transport = transport
        self._cache: dict[str, tuple[float, tuple[_ClubEloRow, ...]]] = {}
        self._cache_lock = asyncio.Lock()

    async def resolve_elo(
        self,
        *,
        team_name: str,
        as_of: datetime | date,
        provider_team_key: str | None = None,
    ) -> tuple[ProviderTeamCandidate, ExternalDataPoint] | None:
        as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
        # Upcoming fixtures may be days ahead; only use information available today.
        requested_date = min(as_of_date, datetime.now(UTC).date())
        rows = await self._rows_for_date(requested_date)
        if provider_team_key:
            matches = [row for row in rows if row.club == provider_team_key.strip()]
            match_confidence = 1.0
        else:
            normalized = stable_team_name_key(team_name)
            if not normalized:
                return None
            matches = [
                row for row in rows if stable_team_name_key(row.club) == normalized
            ]
            match_confidence = self.confidence

        # Ambiguous or missing identities fail closed; no fuzzy guess reaches the model.
        if len(matches) != 1:
            return None

        row = matches[0]
        observed_at = datetime.now(UTC)
        candidate = ProviderTeamCandidate(
            provider_team_key=row.club,
            provider_team_name=row.club,
            confidence=match_confidence,
        )
        point = ExternalDataPoint(
            value=row.elo,
            source="clubelo",
            captured_at=observed_at,
            expires_at=observed_at + timedelta(seconds=self.cache_seconds),
            confidence=min(self.confidence, match_confidence),
            is_fallback=True,
            details={
                "effective_date": requested_date.isoformat(),
                "country": row.country,
                "level": row.level,
                "valid_from": row.valid_from,
                "valid_to": row.valid_to,
                "transport_security": (
                    "tls" if self.base_url.startswith("https://") else "http"
                ),
            },
        )
        return candidate, point

    async def _rows_for_date(self, requested_date: date) -> tuple[_ClubEloRow, ...]:
        cache_key = requested_date.isoformat()
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]

        async with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            content = await self._download(cache_key)
            rows = self._parse(content)
            self._cache[cache_key] = (time.monotonic() + self.cache_seconds, rows)
            return rows

    async def _download(self, cache_key: str) -> bytes:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={"User-Agent": "BetAIPlatform/1.0 external-feature-fallback"},
            ) as client:
                response = await client.get(f"{self.base_url}/{cache_key}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ClubEloDownloadError(
                f"ClubElo rating feed could not be downloaded for {cache_key}"
            ) from exc

        if len(response.content) > MAX_CLUBELO_PAYLOAD_BYTES:
            raise ClubEloFormatError("ClubElo payload exceeds the safety limit")
        return response.content

    @staticmethod
    def _parse(content: bytes) -> tuple[_ClubEloRow, ...]:
        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ClubEloFormatError("ClubElo payload is not valid UTF-8") from exc

        reader = csv.DictReader(io.StringIO(decoded))
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ClubEloFormatError(f"ClubElo payload is missing columns: {missing}")

        rows: list[_ClubEloRow] = []
        for line_number, raw in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in raw.values()):
                continue
            club = str(raw.get("Club") or "").strip()
            if not club or len(club) > 150:
                raise ClubEloFormatError(
                    f"ClubElo row {line_number} has an invalid club name"
                )
            try:
                elo = float(str(raw.get("Elo") or "").strip())
            except ValueError as exc:
                raise ClubEloFormatError(
                    f"ClubElo row {line_number} has a non-numeric Elo"
                ) from exc
            if not math.isfinite(elo) or not 500.0 <= elo <= 3000.0:
                raise ClubEloFormatError(
                    f"ClubElo row {line_number} has an out-of-range Elo"
                )

            valid_from = ClubEloClient._iso_date(raw.get("From"), line_number, "From")
            valid_to = ClubEloClient._iso_date(raw.get("To"), line_number, "To")
            level = ClubEloClient._optional_level(raw.get("Level"), line_number)
            country = str(raw.get("Country") or "").strip()[:10] or None
            rows.append(
                _ClubEloRow(
                    club=club,
                    elo=elo,
                    country=country,
                    level=level,
                    valid_from=valid_from,
                    valid_to=valid_to,
                )
            )
        if not rows:
            raise ClubEloFormatError("ClubElo payload contains no ratings")
        return tuple(rows)

    @staticmethod
    def _iso_date(value: object, line_number: int, column: str) -> str:
        raw = str(value or "").strip()
        try:
            return date.fromisoformat(raw).isoformat()
        except ValueError as exc:
            raise ClubEloFormatError(
                f"ClubElo row {line_number} has an invalid {column} date"
            ) from exc

    @staticmethod
    def _optional_level(value: object, line_number: int) -> int | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            level = int(raw)
        except ValueError as exc:
            raise ClubEloFormatError(
                f"ClubElo row {line_number} has an invalid Level"
            ) from exc
        # ClubElo uses level 0 for clubs outside its current domestic tier model.
        if not 0 <= level <= 20:
            raise ClubEloFormatError(
                f"ClubElo row {line_number} has an out-of-range Level"
            )
        return level
