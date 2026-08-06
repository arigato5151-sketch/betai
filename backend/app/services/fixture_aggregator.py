from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS, LEAGUE_PRIORITY
from app.core.config import settings
from app.core.team_identity import stable_team_name_key
from app.providers.openligadb import ID_OFFSET as OPENLIGADB_ID_OFFSET
from app.providers.openligadb import OpenLigaDBClient
from app.services.api_football import APIFootballClient
from app.services.cache import cache
from app.services.fixture_download import FixtureDownloadClient, UPCOMING_FEEDS

logger = logging.getLogger("bet-ai-pro.fixture_aggregator")

ISTANBUL = ZoneInfo("Europe/Istanbul")
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
SOURCE_ID_OFFSETS = {
    "thesportsdb": 1_000_000_000,
    "sportmonks": 1_250_000_000,
    "football_data_org": 1_500_000_000,
    "fixture_download": 1_750_000_000,
}
MAX_PROVIDER_ID = 249_999_999

LEAGUE_ALIASES: dict[tuple[str, str], int] = {
    ("uefa champions league", ""): 2,
    ("champions league", ""): 2,
    ("uefa europa league", ""): 3,
    ("europa league", ""): 3,
    ("uefa europa conference league", ""): 848,
    ("uefa conference league", ""): 848,
    ("english premier league", ""): 39,
    ("premier league", "england"): 39,
    ("english league championship", ""): 40,
    ("championship", "england"): 40,
    ("spanish la liga", ""): 140,
    ("la liga", "spain"): 140,
    ("italian serie a", ""): 135,
    ("serie a", "italy"): 135,
    ("german bundesliga", ""): 78,
    ("bundesliga", "germany"): 78,
    ("german 2 bundesliga", ""): 79,
    ("2 bundesliga", "germany"): 79,
    ("french ligue 1", ""): 61,
    ("ligue 1", "france"): 61,
    ("french ligue 2", ""): 62,
    ("ligue 2", "france"): 62,
    ("italian serie b", ""): 136,
    ("serie b", "italy"): 136,
    ("portuguese primeira liga", ""): 94,
    ("primeira liga", "portugal"): 94,
    ("liga portugal", "portugal"): 94,
    ("turkish super lig", ""): 203,
    ("super lig", "turkey"): 203,
    ("dutch eredivisie", ""): 88,
    ("eredivisie", "netherlands"): 88,
    ("belgian pro league", ""): 144,
    ("jupiler pro league", "belgium"): 144,
    ("russian premier league", ""): 235,
    ("premier league", "russia"): 235,
    ("scottish premiership", ""): 179,
    ("scottish premier league", ""): 179,
    ("premiership", "scotland"): 179,
    ("austrian bundesliga", ""): 218,
    ("bundesliga", "austria"): 218,
    ("swiss super league", ""): 207,
    ("super league", "switzerland"): 207,
    ("greek super league", ""): 197,
    ("super league 1", ""): 197,
    ("super league", "greece"): 197,
    ("danish superliga", ""): 119,
    ("superliga", "denmark"): 119,
    ("superligaen", "denmark"): 119,
}

FOOTBALL_DATA_CODES = {
    "CL": 2,
    "EL": 3,
    "ECL": 848,
    "PL": 39,
    "ELC": 40,
    "PD": 140,
    "SA": 135,
    "SB": 136,
    "BL1": 78,
    "BL2": 79,
    "FL1": 61,
    "FL2": 62,
    "PPL": 94,
    "DED": 88,
}

LEAGUE_NAMES = {
    39: "Premier League",
    40: "Championship",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    62: "Ligue 2",
    94: "Liga Portugal",
    203: "Süper Lig",
    88: "Eredivisie",
    179: "Scottish Premiership",
    218: "Austrian Bundesliga",
    207: "Swiss Super League",
    197: "Super League 1",
    119: "Danish Superliga",
}


class FixtureSourceError(RuntimeError):
    """Raised when an optional fixture provider cannot be read safely."""


def canonical_league_id(name: object, country: object = "") -> int | None:
    league_key = stable_team_name_key(str(name or ""))
    country_key = stable_team_name_key(str(country or ""))
    return LEAGUE_ALIASES.get((league_key, country_key)) or LEAGUE_ALIASES.get(
        (league_key, "")
    )


def _parse_datetime(value: object, *, assume_utc: bool = True) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC if assume_utc else ISTANBUL)
    return parsed.astimezone(ISTANBUL)


def _positive_id(value: object, source: str) -> int | None:
    try:
        raw_id = int(str(value))
    except (TypeError, ValueError):
        return None
    # Namespaced IDs stay inside PostgreSQL's signed INTEGER range.
    if raw_id <= 0 or raw_id > MAX_PROVIDER_ID:
        return None
    return SOURCE_ID_OFFSETS[source] + raw_id


def _hashed_provider_id(value: object, source: str) -> int | None:
    try:
        raw_id = abs(int(str(value)))
    except (TypeError, ValueError):
        return None
    return _positive_id((raw_id % MAX_PROVIDER_ID) or 1, source)


def _provider_fixture_id(fixture_id: object, source: object) -> str | None:
    try:
        namespaced_id = int(str(fixture_id))
    except (TypeError, ValueError):
        return None
    source_name = str(source or "")
    if source_name == "openligadb":
        return str(namespaced_id - OPENLIGADB_ID_OFFSET)
    offset = SOURCE_ID_OFFSETS.get(source_name)
    if offset is not None:
        return str(namespaced_id - offset)
    return str(namespaced_id)


class _HTTPFixtureSource:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    async def _get(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={"Accept": "application/json", **dict(headers or {})},
            ) as client:
                response = await client.get(
                    f"{self.base_url}/{path.lstrip('/')}", params=dict(params or {})
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FixtureSourceError(
                f"Fixture provider request failed: {path}"
            ) from exc
        if len(response.content) > MAX_PAYLOAD_BYTES:
            raise FixtureSourceError("Fixture provider payload exceeds safety limit")
        try:
            payload = response.json()
        except ValueError as exc:
            raise FixtureSourceError("Fixture provider returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise FixtureSourceError("Fixture provider payload must be an object")
        return payload


class TheSportsDBFixtureSource(_HTTPFixtureSource):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        enabled: bool | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.THESPORTSDB_BASE_URL,
            timeout_seconds=timeout_seconds or settings.MULTI_FIXTURE_TIMEOUT_SECONDS,
            transport=transport,
        )
        self.enabled = settings.THESPORTSDB_ENABLED if enabled is None else enabled

    @property
    def configured(self) -> bool:
        return self.enabled

    async def get_fixtures(self, start: date, end: date) -> list[dict[str, Any]]:
        requests = []
        current = start
        while current <= end:
            requests.append(
                self._get(
                    "eventsday.php",
                    params={"d": current.isoformat(), "s": "Soccer"},
                )
            )
            current += timedelta(days=1)

        rows: list[dict[str, Any]] = []
        for payload in await asyncio.gather(*requests):
            events = payload.get("events")
            if isinstance(events, list):
                rows.extend(self._normalize(event) for event in events)
        return [row for row in rows if row]

    @staticmethod
    def _normalize(event: object) -> dict[str, Any]:
        if not isinstance(event, dict):
            return {}
        league_id = canonical_league_id(event.get("strLeague"), event.get("strCountry"))
        fixture_id = _positive_id(event.get("idEvent"), "thesportsdb")
        home = str(event.get("strHomeTeam") or "").strip()
        away = str(event.get("strAwayTeam") or "").strip()
        kickoff = _parse_datetime(event.get("strTimestamp"))
        if kickoff is None:
            raw_date = str(event.get("dateEvent") or "").strip()
            raw_time = str(event.get("strTime") or "00:00:00").strip()
            kickoff = _parse_datetime(f"{raw_date}T{raw_time}")
        if not league_id or not fixture_id or not home or not away or kickoff is None:
            return {}
        return _fixture_row(
            fixture_id=fixture_id,
            provider_fixture_id=event.get("idEvent"),
            league_id=league_id,
            league=str(event.get("strLeague") or ""),
            home=home,
            away=away,
            kickoff=kickoff,
            source="thesportsdb",
            home_team_id=_positive_id(event.get("idHomeTeam"), "thesportsdb"),
            away_team_id=_positive_id(event.get("idAwayTeam"), "thesportsdb"),
        )


class FootballDataOrgFixtureSource(_HTTPFixtureSource):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        enabled: bool | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.FOOTBALL_DATA_ORG_BASE_URL,
            timeout_seconds=timeout_seconds or settings.MULTI_FIXTURE_TIMEOUT_SECONDS,
            transport=transport,
        )
        self.api_key = (api_key or settings.FOOTBALL_DATA_ORG_API_KEY).strip()
        self.enabled = (
            settings.FOOTBALL_DATA_ORG_ENABLED if enabled is None else enabled
        )

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_key)

    async def get_fixtures(self, start: date, end: date) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        payload = await self._get(
            "matches",
            params={"dateFrom": start.isoformat(), "dateTo": end.isoformat()},
            headers={"X-Auth-Token": self.api_key},
        )
        matches = payload.get("matches")
        if not isinstance(matches, list):
            return []
        return [row for match in matches if (row := self._normalize(match))]

    @staticmethod
    def _normalize(match: object) -> dict[str, Any]:
        if not isinstance(match, dict):
            return {}
        competition = match.get("competition")
        home_team = match.get("homeTeam")
        away_team = match.get("awayTeam")
        if not all(
            isinstance(value, dict) for value in (competition, home_team, away_team)
        ):
            return {}
        assert isinstance(competition, dict)
        assert isinstance(home_team, dict)
        assert isinstance(away_team, dict)
        league_id = FOOTBALL_DATA_CODES.get(str(competition.get("code") or ""))
        league_id = league_id or canonical_league_id(competition.get("name"))
        fixture_id = _positive_id(match.get("id"), "football_data_org")
        kickoff = _parse_datetime(match.get("utcDate"))
        home = str(home_team.get("name") or "").strip()
        away = str(away_team.get("name") or "").strip()
        if not league_id or not fixture_id or kickoff is None or not home or not away:
            return {}
        return _fixture_row(
            fixture_id=fixture_id,
            provider_fixture_id=match.get("id"),
            league_id=league_id,
            league=str(competition.get("name") or ""),
            home=home,
            away=away,
            kickoff=kickoff,
            source="football_data_org",
            home_team_id=_positive_id(home_team.get("id"), "football_data_org"),
            away_team_id=_positive_id(away_team.get("id"), "football_data_org"),
        )


class SportmonksFixtureSource(_HTTPFixtureSource):
    def __init__(
        self,
        *,
        api_token: str | None = None,
        enabled: bool | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url or settings.SPORTMONKS_BASE_URL,
            timeout_seconds=timeout_seconds or settings.SPORTMONKS_TIMEOUT_SECONDS,
            transport=transport,
        )
        self.api_token = (api_token or settings.SPORTMONKS_API_TOKEN).strip()
        self.enabled = settings.SPORTMONKS_ENABLED if enabled is None else enabled

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_token)

    async def get_fixtures(self, start: date, end: date) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        payload = await self._get(
            f"fixtures/between/{start.isoformat()}/{end.isoformat()}",
            params={"include": "participants;league.country", "per_page": "100"},
            headers={"Authorization": self.api_token},
        )
        fixtures = payload.get("data")
        if not isinstance(fixtures, list):
            return []
        return [row for item in fixtures if (row := self._normalize(item))]

    @staticmethod
    def _normalize(item: object) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        league = item.get("league")
        participants = item.get("participants")
        if not isinstance(league, dict) or not isinstance(participants, list):
            return {}
        country = league.get("country")
        country_name = country.get("name") if isinstance(country, dict) else ""
        league_id = canonical_league_id(league.get("name"), country_name)
        fixture_id = _positive_id(item.get("id"), "sportmonks")
        kickoff = _parse_datetime(item.get("starting_at"))
        home = _sportmonks_participant(participants, "home")
        away = _sportmonks_participant(participants, "away")
        if not league_id or not fixture_id or kickoff is None or not home or not away:
            return {}
        return _fixture_row(
            fixture_id=fixture_id,
            provider_fixture_id=item.get("id"),
            league_id=league_id,
            league=str(league.get("name") or ""),
            home=home[1],
            away=away[1],
            kickoff=kickoff,
            source="sportmonks",
            home_team_id=_positive_id(home[0], "sportmonks"),
            away_team_id=_positive_id(away[0], "sportmonks"),
        )


class FixtureDownloadFixtureSource:
    def __init__(
        self,
        *,
        client: FixtureDownloadClient | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.client = client or FixtureDownloadClient()
        self.enabled = (
            settings.FIXTURE_DOWNLOAD_UPCOMING_ENABLED if enabled is None else enabled
        )

    @property
    def configured(self) -> bool:
        return self.enabled

    async def get_fixtures(self, start: date, end: date) -> list[dict[str, Any]]:
        season = start.year if start.month >= 7 else start.year - 1
        league_ids = tuple(UPCOMING_FEEDS)
        results = await asyncio.gather(
            *(
                self.client.get_scheduled_fixtures(
                    league_id,
                    season,
                    start=start,
                    end=end,
                )
                for league_id in league_ids
            ),
            return_exceptions=True,
        )
        rows: list[dict[str, Any]] = []
        for league_id, result in zip(league_ids, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "FixtureDownload feed unavailable for league %s: %s",
                    league_id,
                    result,
                )
                continue
            rows.extend(self._normalize(item) for item in result)
        return [row for row in rows if row]

    @staticmethod
    def _normalize(item: object) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        league_id = item.get("league_id")
        kickoff = item.get("kickoff")
        home = str(item.get("home_team") or "").strip()
        away = str(item.get("away_team") or "").strip()
        if (
            not isinstance(league_id, int)
            or not isinstance(kickoff, datetime)
            or not home
            or not away
        ):
            return {}
        fixture_id = _hashed_provider_id(item.get("fixture_id"), "fixture_download")
        home_team_id = _hashed_provider_id(item.get("home_team_id"), "fixture_download")
        away_team_id = _hashed_provider_id(item.get("away_team_id"), "fixture_download")
        if fixture_id is None:
            return {}
        return _fixture_row(
            fixture_id=fixture_id,
            provider_fixture_id=item.get("fixture_id"),
            league_id=league_id,
            league=LEAGUE_NAMES.get(league_id, ""),
            home=home,
            away=away,
            kickoff=kickoff.astimezone(ISTANBUL),
            source="fixture_download",
            home_team_id=home_team_id,
            away_team_id=away_team_id,
        )


def _sportmonks_participant(
    participants: list[object], location: str
) -> tuple[object, str] | None:
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        meta = participant.get("meta")
        if isinstance(meta, dict) and meta.get("location") == location:
            name = str(participant.get("name") or "").strip()
            return (participant.get("id"), name) if name else None
    return None


def _fixture_row(
    *,
    fixture_id: int,
    provider_fixture_id: object,
    league_id: int,
    league: str,
    home: str,
    away: str,
    kickoff: datetime,
    source: str,
    home_team_id: int | None,
    away_team_id: int | None,
) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "provider_fixture_id": str(provider_fixture_id),
        "league": league,
        "home_team": home,
        "away_team": away,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "league_id": league_id,
        "season": kickoff.year,
        "minute": None,
        "score": None,
        "kickoff": kickoff.isoformat(),
        "kickoff_label": kickoff.strftime("%d.%m %H:%M"),
        "status": "NS",
        "is_live": False,
        "is_demo": False,
        "source": source,
        "sources": [source],
    }


class FixtureAggregator:
    def __init__(
        self,
        *,
        api_football: APIFootballClient | None = None,
        football_data: FootballDataOrgFixtureSource | None = None,
        sportmonks: SportmonksFixtureSource | None = None,
        thesportsdb: TheSportsDBFixtureSource | None = None,
        fixture_download: FixtureDownloadFixtureSource | None = None,
        openligadb: OpenLigaDBClient | None = None,
    ) -> None:
        self.api_football = api_football or APIFootballClient()
        self.football_data = football_data or FootballDataOrgFixtureSource()
        self.sportmonks = sportmonks or SportmonksFixtureSource()
        self.thesportsdb = thesportsdb or TheSportsDBFixtureSource()
        self.fixture_download = fixture_download or FixtureDownloadFixtureSource()
        self.openligadb = openligadb or OpenLigaDBClient()

    async def get_upcoming_fixtures(
        self, days: int = 7, limit: int = 100
    ) -> list[dict[str, Any]]:
        # Versioned cache prevents a stale pre-expansion league allowlist result.
        cache_key = f"merged-upcoming:v5:{days}:{limit}"
        cached = await cache.get("fixtures", cache_key)
        if isinstance(cached, list):
            return cached

        today = datetime.now(ISTANBUL).date()
        end = today + timedelta(days=max(1, days) - 1)
        tasks: list[tuple[str, Any]] = [
            (
                "api_football",
                self.api_football.get_upcoming_fixtures(days=days, limit=200),
            ),
        ]
        if self.football_data.configured:
            tasks.append(
                ("football_data_org", self.football_data.get_fixtures(today, end))
            )
        if self.sportmonks.configured:
            tasks.append(("sportmonks", self.sportmonks.get_fixtures(today, end)))
        if self.thesportsdb.configured:
            tasks.append(("thesportsdb", self.thesportsdb.get_fixtures(today, end)))
        if self.fixture_download.configured:
            tasks.append(
                ("fixture_download", self.fixture_download.get_fixtures(today, end))
            )
        if self.openligadb.configured:
            tasks.append(
                ("openligadb", self.openligadb.get_upcoming_fixtures(today, end))
            )

        results = await asyncio.gather(
            *(task for _, task in tasks), return_exceptions=True
        )
        provider_rows: list[tuple[str, list[dict[str, Any]]]] = []
        demo_rows: list[dict[str, Any]] = []
        for (source, _), result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Fixture source %s unavailable: %s", source, result)
                continue
            rows = [row for row in result if isinstance(row, dict) and row]
            if source == "api_football":
                demo_rows = [row for row in rows if row.get("is_demo")]
                rows = [
                    {
                        **row,
                        "provider_fixture_id": str(row.get("fixture_id")),
                        "source": "api_football",
                        "sources": ["api_football"],
                    }
                    for row in rows
                    if not row.get("is_demo")
                ]
            elif source == "openligadb":
                rows = [
                    _fixture_row(
                        fixture_id=int(row["fixture_id"]),
                        provider_fixture_id=int(row["fixture_id"])
                        - OPENLIGADB_ID_OFFSET,
                        league_id=int(row["league_id"]),
                        league=str(row["league"]),
                        home=str(row["home_team"]),
                        away=str(row["away_team"]),
                        kickoff=row["kickoff"].astimezone(ISTANBUL),
                        source="openligadb",
                        home_team_id=int(row["home_team_id"]),
                        away_team_id=int(row["away_team_id"]),
                    )
                    for row in rows
                    if isinstance(row.get("kickoff"), datetime)
                ]
            provider_rows.append((source, rows))

        merged = self._merge(provider_rows, days=days)[:limit]
        if not merged:
            merged = demo_rows[:limit]
        await cache.set("fixtures", cache_key, merged, 900)
        for fixture in merged:
            await cache.set(
                "fixtures", f"merged-fixture:{fixture['fixture_id']}", fixture, 86400
            )
        return merged

    async def get_fixture_prefill(self, fixture_id: int) -> dict[str, Any] | None:
        fixture = await cache.get("fixtures", f"merged-fixture:{fixture_id}")
        if not isinstance(fixture, dict):
            fixtures = await self.get_upcoming_fixtures(days=14, limit=200)
            fixture = next(
                (item for item in fixtures if item.get("fixture_id") == fixture_id),
                None,
            )
        if not isinstance(fixture, dict):
            return await self._api_football_prefill(fixture_id)
        if fixture.get("source") == "api_football":
            return await self._api_football_prefill(fixture_id)
        fixture = {
            **fixture,
            "provider_fixture_id": fixture.get("provider_fixture_id")
            or _provider_fixture_id(fixture_id, fixture.get("source")),
        }
        return {
            "fixture": fixture,
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "odd": 2.0,
            "home_stats": {"form": 50, "attack": 50, "defense": 50, "xg": 1.2},
            "away_stats": {"form": 50, "attack": 50, "defense": 50, "xg": 1.2},
            "market_1x2": None,
            "auto_filled": True,
            "data_quality": "fixture_source_fallback",
            "data_methodology": {
                "fixture": fixture.get("source"),
                "stats": "Yerel tarihsel veri; bulunamazsa nötr başlangıç profili",
                "model": "Poisson + Dixon-Coles + ML ensemble",
            },
        }

    async def _api_football_prefill(self, fixture_id: int) -> dict[str, Any] | None:
        payload = await self.api_football.get_fixture_prefill(fixture_id)
        if not isinstance(payload, dict):
            return None
        fixture = payload.get("fixture")
        if isinstance(fixture, dict):
            payload["fixture"] = {
                **fixture,
                "source": "api_football",
                "sources": ["api_football"],
                "provider_fixture_id": str(fixture_id),
            }
        return payload

    @staticmethod
    def _merge(
        provider_rows: list[tuple[str, list[dict[str, Any]]]], *, days: int
    ) -> list[dict[str, Any]]:
        now = datetime.now(ISTANBUL)
        horizon = datetime.combine(
            now.date() + timedelta(days=max(1, days) - 1), time.max, tzinfo=ISTANBUL
        )
        merged: dict[tuple[int, str, str, str], dict[str, Any]] = {}
        for _source, rows in provider_rows:
            for row in rows:
                league_id = row.get("league_id")
                kickoff = _parse_datetime(row.get("kickoff"), assume_utc=False)
                home_key = stable_team_name_key(str(row.get("home_team") or ""))
                away_key = stable_team_name_key(str(row.get("away_team") or ""))
                if (
                    not isinstance(league_id, int)
                    or league_id not in ALLOWED_LEAGUE_IDS
                    or kickoff is None
                    or kickoff < now
                    or kickoff > horizon
                    or not home_key
                    or not away_key
                ):
                    continue
                key = (league_id, home_key, away_key, kickoff.date().isoformat())
                existing = merged.get(key)
                if existing is None:
                    merged[key] = row
                    continue
                sources = list(existing.get("sources") or [existing.get("source")])
                for source in row.get("sources") or [row.get("source")]:
                    if source and source not in sources:
                        sources.append(source)
                existing["sources"] = sources

        return sorted(
            merged.values(),
            key=lambda row: (
                _parse_datetime(row.get("kickoff"), assume_utc=False)
                or datetime.max.replace(tzinfo=UTC),
                LEAGUE_PRIORITY.get(row.get("league_id"), 99),
                row.get("fixture_id", 0),
            ),
        )
