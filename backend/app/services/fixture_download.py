from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import Any, Mapping

import httpx

from app.core.config import settings
from app.core.team_identity import stable_team_name_key

MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


class FixtureDownloadError(RuntimeError):
    """Base error for the public fixture feeds."""


class FixtureDownloadFormatError(FixtureDownloadError):
    """Raised when a feed violates the expected schema."""


@dataclass(frozen=True)
class UEFAFeed:
    league_id: int
    slug: str


UEFA_FEEDS: Mapping[int, UEFAFeed] = MappingProxyType(
    {
        2: UEFAFeed(2, "champions-league"),
        3: UEFAFeed(3, "europa-league"),
        848: UEFAFeed(848, "conference-league"),
    }
)

UPCOMING_FEEDS: Mapping[int, UEFAFeed] = MappingProxyType(
    {
        39: UEFAFeed(39, "epl"),
        40: UEFAFeed(40, "championship"),
        140: UEFAFeed(140, "la-liga"),
        135: UEFAFeed(135, "serie-a"),
        78: UEFAFeed(78, "bundesliga"),
        61: UEFAFeed(61, "ligue-1"),
        62: UEFAFeed(62, "ligue-2"),
        94: UEFAFeed(94, "primeira-liga"),
        203: UEFAFeed(203, "super-lig"),
        88: UEFAFeed(88, "eredivisie"),
    }
)


def _stable_negative_id(namespace: str, natural_key: str) -> int:
    digest = hashlib.blake2b(
        f"{namespace}:{natural_key}".encode(), digest_size=8
    ).digest()
    identifier = int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)
    return -(identifier or 1)


class FixtureDownloadClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.FIXTURE_DOWNLOAD_BASE_URL).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds or settings.FIXTURE_DOWNLOAD_TIMEOUT_SECONDS
        )
        self.transport = transport

    async def get_completed_fixtures(
        self, league_id: int, season: int, *, now: datetime | None = None
    ) -> list[dict[str, Any]]:
        feed = UEFA_FEEDS.get(league_id)
        if feed is None:
            raise ValueError(f"Unsupported UEFA league_id: {league_id}")
        if season < 2000 or season > datetime.now(UTC).year + 1:
            raise ValueError(f"Invalid season: {season}")
        payload = await self._download(f"{feed.slug}-{season}")
        return self._parse(
            payload,
            league_id=league_id,
            season=season,
            now=now or datetime.now(UTC),
        )

    async def get_scheduled_fixtures(
        self,
        league_id: int,
        season: int,
        *,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        feed = UPCOMING_FEEDS.get(league_id)
        if feed is None:
            raise ValueError(f"Unsupported upcoming league_id: {league_id}")
        if end < start or (end - start).days > 31:
            raise ValueError("Invalid upcoming fixture date range")
        payload = await self._download(f"{feed.slug}-{season}")
        return self._parse_scheduled(
            payload,
            league_id=league_id,
            season=season,
            start=start,
            end=end,
        )

    async def _download(self, slug: str) -> object:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={"User-Agent": "BetAIPlatform/1.0 fixture-importer"},
            ) as client:
                response = await client.get(f"{self.base_url}/{slug}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FixtureDownloadError("Fixture feed could not be downloaded") from exc
        if len(response.content) > MAX_PAYLOAD_BYTES:
            raise FixtureDownloadFormatError("Fixture payload is too large")
        try:
            return response.json()
        except ValueError as exc:
            raise FixtureDownloadFormatError("Fixture payload is not JSON") from exc

    @staticmethod
    def _parse(
        payload: object,
        *,
        league_id: int,
        season: int,
        now: datetime,
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise FixtureDownloadFormatError("UEFA fixture payload must be a list")
        fixtures: list[dict[str, Any]] = []
        natural_keys: set[str] = set()
        for index, raw in enumerate(payload, start=1):
            if not isinstance(raw, dict):
                raise FixtureDownloadFormatError(f"Invalid fixture row {index}")
            home = str(raw.get("HomeTeam") or "").strip()
            away = str(raw.get("AwayTeam") or "").strip()
            date_raw = str(raw.get("DateUtc") or "").strip()
            home_score = raw.get("HomeTeamScore")
            away_score = raw.get("AwayTeamScore")
            if not home or not away or home == away or not date_raw:
                raise FixtureDownloadFormatError(
                    f"Invalid fixture identity at row {index}"
                )
            try:
                kickoff = datetime.strptime(date_raw, "%Y-%m-%d %H:%M:%SZ").replace(
                    tzinfo=UTC
                )
            except ValueError as exc:
                raise FixtureDownloadFormatError(
                    f"Invalid fixture date at row {index}"
                ) from exc
            # Published schedules can include future games without a final score.
            if kickoff >= now or home_score is None or away_score is None:
                continue
            if (
                isinstance(home_score, bool)
                or isinstance(away_score, bool)
                or not isinstance(home_score, int)
                or not isinstance(away_score, int)
                or home_score < 0
                or away_score < 0
            ):
                raise FixtureDownloadFormatError(
                    f"Invalid fixture score at row {index}"
                )
            home_key = stable_team_name_key(home)
            away_key = stable_team_name_key(away)
            natural_key = (
                f"{league_id}:{season}:{kickoff.isoformat()}:{home_key}:{away_key}"
            )
            if natural_key in natural_keys:
                raise FixtureDownloadFormatError(f"Duplicate fixture at row {index}")
            natural_keys.add(natural_key)
            fixtures.append(
                {
                    "fixture_id": _stable_negative_id(
                        "fixture-download-fixture", natural_key
                    ),
                    "league_id": league_id,
                    "season": season,
                    "kickoff": kickoff,
                    "home_team_id": _stable_negative_id(
                        "fixture-download-team", home_key
                    ),
                    "away_team_id": _stable_negative_id(
                        "fixture-download-team", away_key
                    ),
                    "home_team": home[:100],
                    "away_team": away[:100],
                    "home_goals": home_score,
                    "away_goals": away_score,
                    "home_starting_xi": None,
                    "away_starting_xi": None,
                    "actual_result": (
                        "HOME_WIN"
                        if home_score > away_score
                        else "AWAY_WIN" if away_score > home_score else "DRAW"
                    ),
                    "status": "FT",
                    "data_source": "fixture_download",
                }
            )
        return fixtures

    @staticmethod
    def _parse_scheduled(
        payload: object,
        *,
        league_id: int,
        season: int,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise FixtureDownloadFormatError("Fixture payload must be a list")
        fixtures: list[dict[str, Any]] = []
        natural_keys: set[str] = set()
        for index, raw in enumerate(payload, start=1):
            if not isinstance(raw, dict):
                raise FixtureDownloadFormatError(f"Invalid fixture row {index}")
            home = str(raw.get("HomeTeam") or "").strip()
            away = str(raw.get("AwayTeam") or "").strip()
            date_raw = str(raw.get("DateUtc") or "").strip()
            if not home or not away or home == away or not date_raw:
                raise FixtureDownloadFormatError(
                    f"Invalid fixture identity at row {index}"
                )
            try:
                kickoff = datetime.strptime(date_raw, "%Y-%m-%d %H:%M:%SZ").replace(
                    tzinfo=UTC
                )
            except ValueError as exc:
                raise FixtureDownloadFormatError(
                    f"Invalid fixture date at row {index}"
                ) from exc
            if kickoff.date() < start or kickoff.date() > end:
                continue
            home_key = stable_team_name_key(home)
            away_key = stable_team_name_key(away)
            natural_key = (
                f"{league_id}:{season}:{kickoff.isoformat()}:{home_key}:{away_key}"
            )
            if natural_key in natural_keys:
                raise FixtureDownloadFormatError(f"Duplicate fixture at row {index}")
            natural_keys.add(natural_key)
            fixtures.append(
                {
                    "fixture_id": _stable_negative_id(
                        "fixture-download-fixture", natural_key
                    ),
                    "league_id": league_id,
                    "season": season,
                    "kickoff": kickoff,
                    "home_team_id": _stable_negative_id(
                        "fixture-download-team", home_key
                    ),
                    "away_team_id": _stable_negative_id(
                        "fixture-download-team", away_key
                    ),
                    "home_team": home[:100],
                    "away_team": away[:100],
                    "status": "NS",
                    "data_source": "fixture_download",
                }
            )
        return fixtures
