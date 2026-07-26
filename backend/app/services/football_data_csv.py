from __future__ import annotations

import asyncio
import csv
import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings
from app.core.team_identity import normalize_team_name


class FootballDataError(RuntimeError):
    """Base error for the external Football-Data CSV source."""


class FootballDataDownloadError(FootballDataError):
    """Raised when a CSV feed cannot be downloaded."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FootballDataFormatError(FootballDataError):
    """Raised when a downloaded feed does not match the expected schema."""


@dataclass(frozen=True)
class FootballDataLeague:
    league_id: int
    division: str
    country: str
    timezone: str
    rolling_feed: bool = False


@dataclass(frozen=True)
class FootballDataImport:
    fixtures: list[dict]
    skipped_rows: int


_LEAGUES = (
    FootballDataLeague(39, "E0", "England", "Europe/London"),
    FootballDataLeague(40, "E1", "England", "Europe/London"),
    FootballDataLeague(140, "SP1", "Spain", "Europe/Madrid"),
    FootballDataLeague(135, "I1", "Italy", "Europe/Rome"),
    FootballDataLeague(136, "I2", "Italy", "Europe/Rome"),
    FootballDataLeague(78, "D1", "Germany", "Europe/Berlin"),
    FootballDataLeague(79, "D2", "Germany", "Europe/Berlin"),
    FootballDataLeague(61, "F1", "France", "Europe/Paris"),
    FootballDataLeague(62, "F2", "France", "Europe/Paris"),
    FootballDataLeague(94, "P1", "Portugal", "Europe/Lisbon"),
    FootballDataLeague(203, "T1", "Turkey", "Europe/Istanbul"),
    FootballDataLeague(88, "N1", "Netherlands", "Europe/Amsterdam"),
    FootballDataLeague(144, "B1", "Belgium", "Europe/Brussels"),
    FootballDataLeague(235, "RUS", "Russia", "Europe/Moscow", rolling_feed=True),
)
FOOTBALL_DATA_LEAGUES: Mapping[int, FootballDataLeague] = MappingProxyType(
    {league.league_id: league for league in _LEAGUES}
)
FOOTBALL_DATA_LEAGUE_IDS = frozenset(FOOTBALL_DATA_LEAGUES)

_STANDARD_COLUMNS = frozenset(
    {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
)
_ROLLING_COLUMNS = frozenset(
    {"Country", "League", "Season", "Date", "Home", "Away", "HG", "AG", "Res"}
)
_RESULTS = {"H": "HOME_WIN", "D": "DRAW", "A": "AWAY_WIN"}


def _stable_negative_id(namespace: str, natural_key: str) -> int:
    digest = hashlib.blake2b(
        f"{namespace}:{natural_key}".encode("utf-8"), digest_size=8
    ).digest()
    # Keep the value in PostgreSQL's signed BIGINT range and away from API IDs.
    identifier = int.from_bytes(digest, byteorder="big") & ((1 << 63) - 1)
    return -(identifier or 1)


class FootballDataCSVClient:
    """Download and normalize completed fixtures from Football-Data CSV feeds."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (
            (base_url or settings.FOOTBALL_DATA_BASE_URL).strip().rstrip("/")
        )
        self.timeout_seconds = timeout_seconds or settings.FOOTBALL_DATA_TIMEOUT_SECONDS
        self.transport = transport

    @property
    def supported_league_ids(self) -> frozenset[int]:
        return FOOTBALL_DATA_LEAGUE_IDS

    async def get_completed_fixtures(
        self, league_id: int, season: int
    ) -> FootballDataImport:
        league = FOOTBALL_DATA_LEAGUES.get(league_id)
        if league is None:
            raise ValueError(f"Unsupported Football-Data league_id: {league_id}")
        if season < 2000 or season > datetime.now(UTC).year + 1:
            raise ValueError(f"Invalid season: {season}")

        path = self._feed_path(league, season)
        content = await self._download(path)
        return self._parse(content, league=league, season=season)

    @staticmethod
    def _feed_path(league: FootballDataLeague, season: int) -> str:
        if league.rolling_feed:
            return f"/new/{league.division}.csv"
        season_code = f"{season % 100:02d}{(season + 1) % 100:02d}"
        return f"/mmz4281/{season_code}/{league.division}.csv"

    async def _download(self, path: str) -> bytes:
        url = f"{self.base_url}{path}"
        headers = {"User-Agent": "BetAIPlatform/1.0 historical-data-importer"}
        last_error: Exception | None = None

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            transport=self.transport,
            headers=headers,
        ) as client:
            for attempt in range(3):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.content
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code == 404:
                        raise FootballDataDownloadError(
                            f"Football-Data feed is not published yet: {path}",
                            status_code=404,
                        ) from exc
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2**attempt))
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2**attempt))

        status_code = (
            last_error.response.status_code
            if isinstance(last_error, httpx.HTTPStatusError)
            else None
        )
        raise FootballDataDownloadError(
            f"Football-Data feed could not be downloaded: {path}",
            status_code=status_code,
        ) from last_error

    @staticmethod
    def _decode(content: bytes) -> str:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError:
            # Older league files occasionally use Windows-1252 team names.
            return content.decode("cp1252")

    def _parse(
        self,
        content: bytes,
        *,
        league: FootballDataLeague,
        season: int,
    ) -> FootballDataImport:
        reader = csv.DictReader(io.StringIO(self._decode(content)))
        fieldnames = set(reader.fieldnames or [])
        required_columns = (
            _ROLLING_COLUMNS if league.rolling_feed else _STANDARD_COLUMNS
        )
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise FootballDataFormatError(
                f"{league.division} feed is missing columns: {missing_columns}"
            )

        fixtures: list[dict] = []
        skipped_rows = 0
        fixture_keys_by_id: dict[int, str] = {}
        team_keys_by_id: dict[int, str] = {}

        for line_number, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            if not self._belongs_to_feed(row, league=league, season=season):
                continue

            columns = (
                ("Home", "Away", "HG", "AG", "Res")
                if league.rolling_feed
                else ("HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")
            )
            home_team, away_team, home_goals, away_goals, result = (
                str(row.get(column) or "").strip() for column in columns
            )
            date_value = str(row.get("Date") or "").strip()
            time_value = str(row.get("Time") or "").strip()

            # In an in-progress season, future fixtures legitimately have no result.
            if not all(
                (home_team, away_team, date_value, home_goals, away_goals, result)
            ):
                skipped_rows += 1
                continue

            try:
                home_score = int(home_goals)
                away_score = int(away_goals)
                kickoff = self._parse_kickoff(
                    date_value, time_value, timezone_name=league.timezone
                )
            except ValueError as exc:
                raise FootballDataFormatError(
                    f"{league.division} feed has an invalid row at line {line_number}"
                ) from exc
            if home_score < 0 or away_score < 0 or result not in _RESULTS:
                raise FootballDataFormatError(
                    f"{league.division} feed has an invalid result at line {line_number}"
                )

            actual_result = _RESULTS[result]
            score_result = (
                "HOME_WIN"
                if home_score > away_score
                else "AWAY_WIN" if away_score > home_score else "DRAW"
            )
            if actual_result != score_result:
                raise FootballDataFormatError(
                    f"{league.division} feed has inconsistent scores at line {line_number}"
                )

            home_key = f"{league.country}:{normalize_team_name(home_team)}"
            away_key = f"{league.country}:{normalize_team_name(away_team)}"
            natural_fixture_key = (
                f"{league.division}:{season}:{kickoff.isoformat()}:"
                f"{home_key}:{away_key}"
            )
            fixture_id = _stable_negative_id(
                "football-data-fixture", natural_fixture_key
            )
            home_team_id = _stable_negative_id("football-data-team", home_key)
            away_team_id = _stable_negative_id("football-data-team", away_key)
            self._assert_no_collision(
                fixture_keys_by_id, fixture_id, natural_fixture_key
            )
            self._assert_no_collision(team_keys_by_id, home_team_id, home_key)
            self._assert_no_collision(team_keys_by_id, away_team_id, away_key)

            fixtures.append(
                {
                    "fixture_id": fixture_id,
                    "league_id": league.league_id,
                    "season": season,
                    "kickoff": kickoff,
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "home_team": home_team[:100],
                    "away_team": away_team[:100],
                    "home_goals": home_score,
                    "away_goals": away_score,
                    "home_starting_xi": None,
                    "away_starting_xi": None,
                    "actual_result": actual_result,
                    "status": "FT",
                    "data_source": "football_data_csv",
                }
            )

        return FootballDataImport(fixtures=fixtures, skipped_rows=skipped_rows)

    @staticmethod
    def _belongs_to_feed(
        row: Mapping[str, object],
        *,
        league: FootballDataLeague,
        season: int,
    ) -> bool:
        if not league.rolling_feed:
            return str(row.get("Div") or "").strip() == league.division
        return (
            str(row.get("Country") or "").strip().casefold()
            == league.country.casefold()
            and str(row.get("League") or "").strip().casefold() == "premier league"
            and str(row.get("Season") or "").strip() == f"{season}/{season + 1}"
        )

    @staticmethod
    def _parse_kickoff(
        date_value: str,
        time_value: str,
        *,
        timezone_name: str,
    ) -> datetime:
        parsed_date: datetime | None = None
        for date_format in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(date_value, date_format)
                break
            except ValueError:
                continue
        if parsed_date is None:
            raise ValueError(f"Unsupported date format: {date_value}")

        parsed_time = datetime.strptime(time_value or "12:00", "%H:%M").time()
        local_kickoff = datetime.combine(
            parsed_date.date(), parsed_time, tzinfo=ZoneInfo(timezone_name)
        )
        return local_kickoff.astimezone(UTC)

    @staticmethod
    def _assert_no_collision(
        keys_by_id: dict[int, str], identifier: int, natural_key: str
    ) -> None:
        existing = keys_by_id.setdefault(identifier, natural_key)
        if existing != natural_key:
            raise FootballDataFormatError(
                f"Stable identifier collision between {existing!r} and {natural_key!r}"
            )
