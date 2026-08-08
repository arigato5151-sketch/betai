from __future__ import annotations

import asyncio
import csv
import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Mapping, cast
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings
from app.core.team_identity import stable_team_name_key


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
    rolling_league_name: str = "Premier League"


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
    FootballDataLeague(
        197,
        "G1",
        "Greece",
        "Europe/Athens",
    ),
    FootballDataLeague(235, "RUS", "Russia", "Europe/Moscow", rolling_feed=True),
    FootballDataLeague(179, "SC0", "Scotland", "Europe/London"),
    FootballDataLeague(
        218,
        "AUT",
        "Austria",
        "Europe/Vienna",
        rolling_feed=True,
        rolling_league_name="Bundesliga",
    ),
    FootballDataLeague(
        207,
        "SWZ",
        "Switzerland",
        "Europe/Zurich",
        rolling_feed=True,
        rolling_league_name="Super League",
    ),
    FootballDataLeague(
        119,
        "DNK",
        "Denmark",
        "Europe/Copenhagen",
        rolling_feed=True,
        rolling_league_name="Superliga",
    ),
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
_ODDS_PLACEHOLDERS = frozenset({"#", "-"})
_OPENING_ODDS_TRIPLETS = (
    ("B365H", "B365D", "B365A"),
    ("AvgH", "AvgD", "AvgA"),
    ("BFDH", "BFDD", "BFDA"),
    ("MaxH", "MaxD", "MaxA"),
)
_CLOSING_ODDS_TRIPLETS = (
    ("B365CH", "B365CD", "B365CA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("BFDCH", "BFDCD", "BFDCA"),
    ("MaxCH", "MaxCD", "MaxCA"),
)


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

            home_key = f"{league.country}:{stable_team_name_key(home_team)}"
            away_key = f"{league.country}:{stable_team_name_key(away_team)}"
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
                    **self._optional_statistics(row),
                    "home_starting_xi": None,
                    "away_starting_xi": None,
                    "actual_result": actual_result,
                    "status": "FT",
                    "data_source": "football_data_csv",
                }
            )

        return FootballDataImport(fixtures=fixtures, skipped_rows=skipped_rows)

    @classmethod
    def _optional_statistics(
        cls, row: Mapping[str, object]
    ) -> dict[str, int | float | None]:
        """Normalize optional match statistics without inventing missing values."""
        integer_columns = {
            "half_time_home_goals": "HTHG",
            "half_time_away_goals": "HTAG",
            "home_shots": "HS",
            "away_shots": "AS",
            "home_shots_on_target": "HST",
            "away_shots_on_target": "AST",
            "home_fouls": "HF",
            "away_fouls": "AF",
            "home_corners": "HC",
            "away_corners": "AC",
            "home_yellow_cards": "HY",
            "away_yellow_cards": "AY",
            "home_red_cards": "HR",
            "away_red_cards": "AR",
        }
        opening = cls._first_complete_odds(row, _OPENING_ODDS_TRIPLETS)
        closing = cls._first_complete_odds(row, _CLOSING_ODDS_TRIPLETS)
        return {
            **{
                target: cls._optional_non_negative_int(row.get(source), source)
                for target, source in integer_columns.items()
            },
            "opening_home_odd": opening[0] if opening else None,
            "opening_draw_odd": opening[1] if opening else None,
            "opening_away_odd": opening[2] if opening else None,
            "closing_home_odd": closing[0] if closing else None,
            "closing_draw_odd": closing[1] if closing else None,
            "closing_away_odd": closing[2] if closing else None,
        }

    @classmethod
    def _first_complete_odds(
        cls,
        row: Mapping[str, object],
        candidates: tuple[tuple[str, str, str], ...],
    ) -> tuple[float, float, float] | None:
        """Choose one complete 1X2 source; never mix bookmakers in a triplet."""
        for columns in candidates:
            raw_values = tuple(str(row.get(column) or "").strip() for column in columns)
            if not all(raw_values):
                continue
            parsed = tuple(
                cls._optional_decimal_odd(value, column)
                for value, column in zip(raw_values, columns, strict=True)
            )
            if all(value is not None for value in parsed):
                return cast(tuple[float, float, float], parsed)
        return None

    @staticmethod
    def _optional_non_negative_int(value: object, column: str) -> int | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise FootballDataFormatError(f"Invalid {column} value") from exc
        if parsed < 0:
            raise FootballDataFormatError(f"Invalid {column} value")
        return parsed

    @staticmethod
    def _optional_decimal_odd(value: object, column: str) -> float | None:
        raw = str(value or "").strip().lower()
        if not raw or raw in _ODDS_PLACEHOLDERS:
            return None
        try:
            parsed = float(raw)
        except ValueError as exc:
            raise FootballDataFormatError(f"Invalid {column} value") from exc
        if not 1.0 < parsed < 1000.0:
            raise FootballDataFormatError(f"Invalid {column} value")
        return parsed

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
            and str(row.get("League") or "").strip().casefold()
            == league.rolling_league_name.casefold()
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
