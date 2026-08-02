from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.config import settings

MAX_PAYLOAD_BYTES = 12 * 1024 * 1024
FIXTURE_ID_OFFSET = 8_000_000_000
TEAM_ID_OFFSET = 8_500_000_000
PLAYER_ID_OFFSET = 9_000_000_000
COMPETITION_LEAGUES = {
    2: 39,  # Premier League
    7: 61,  # Ligue 1
    9: 78,  # Bundesliga
    11: 140,  # La Liga
    12: 135,  # Serie A
    16: 2,  # Champions League
    35: 3,  # Europa League
}


class StatsBombOpenDataError(RuntimeError):
    """Raised when the public StatsBomb repository returns invalid data."""


def _negative_id(offset: int, value: object) -> int | None:
    try:
        source_id = int(str(value))
    except (TypeError, ValueError):
        return None
    if source_id <= 0 or source_id > 1_000_000_000:
        return None
    return -(offset + source_id)


class StatsBombOpenDataClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        concurrency: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.base_url = (base_url or settings.STATSBOMB_OPEN_DATA_BASE_URL).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds or settings.STATSBOMB_OPEN_DATA_TIMEOUT_SECONDS
        )
        self.concurrency = concurrency or settings.STATSBOMB_OPEN_DATA_CONCURRENCY
        self.transport = transport
        self.enabled = (
            settings.STATSBOMB_OPEN_DATA_ENABLED if enabled is None else enabled
        )

    @property
    def configured(self) -> bool:
        return self.enabled

    async def get_catalog(self, min_season: int = 2004) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        competitions = await self._get_json("competitions.json")
        if not isinstance(competitions, list):
            raise StatsBombOpenDataError("StatsBomb competitions must be a list")
        targets: list[tuple[int, int]] = []
        for row in competitions:
            if not isinstance(row, dict):
                continue
            competition_id = row.get("competition_id")
            season_id = row.get("season_id")
            season_name = str(row.get("season_name") or "")
            try:
                season = int(season_name[:4])
            except ValueError:
                continue
            if (
                competition_id in COMPETITION_LEAGUES
                and isinstance(season_id, int)
                and season >= min_season
                and str(row.get("competition_gender") or "male").lower() == "male"
            ):
                targets.append((int(competition_id), season_id))

        semaphore = asyncio.Semaphore(self.concurrency)

        async def fetch(competition_id: int, season_id: int) -> object:
            async with semaphore:
                return await self._get_json(
                    f"matches/{competition_id}/{season_id}.json"
                )

        results = await asyncio.gather(
            *(
                fetch(competition_id, season_id)
                for competition_id, season_id in targets
            ),
            return_exceptions=True,
        )
        fixtures: list[dict[str, Any]] = []
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                continue
            if isinstance(result, list):
                fixtures.extend(
                    row for raw in result if (row := self._normalize_match(raw))
                )
        if targets and failures == len(targets):
            raise StatsBombOpenDataError("All StatsBomb match catalog requests failed")
        return sorted(fixtures, key=lambda row: (row["kickoff"], row["fixture_id"]))

    async def enrich_matches(
        self, fixtures: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[int]]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def enrich(row: dict[str, Any]) -> dict[str, Any] | None:
            match_id = row.get("provider_match_id")
            if not isinstance(match_id, int):
                return None
            try:
                async with semaphore:
                    events = await self._get_json(f"events/{match_id}.json")
            except StatsBombOpenDataError:
                return None
            return self._apply_events(row, events)

        results = await asyncio.gather(*(enrich(row) for row in fixtures))
        enriched = [row for row in results if row is not None]
        enriched_ids = {row["fixture_id"] for row in enriched}
        failures = [
            int(row["fixture_id"])
            for row in fixtures
            if row["fixture_id"] not in enriched_ids
        ]
        return enriched, failures

    async def _get_json(self, path: str) -> object:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BetAIPlatform/1.0 statsbomb-open-importer",
                },
            ) as client:
                response = await client.get(f"{self.base_url}/{path.lstrip('/')}")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise StatsBombOpenDataError(f"StatsBomb request failed: {path}") from exc
        if len(response.content) > MAX_PAYLOAD_BYTES:
            raise StatsBombOpenDataError("StatsBomb payload exceeds safety limit")
        try:
            return response.json()
        except ValueError as exc:
            raise StatsBombOpenDataError("StatsBomb returned invalid JSON") from exc

    @staticmethod
    def _normalize_match(raw: object) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        competition = raw.get("competition")
        season_data = raw.get("season")
        home_data = raw.get("home_team")
        away_data = raw.get("away_team")
        if (
            not isinstance(competition, dict)
            or not isinstance(season_data, dict)
            or not isinstance(home_data, dict)
            or not isinstance(away_data, dict)
        ):
            return {}
        competition_id = competition.get("competition_id")
        if not isinstance(competition_id, int):
            return {}
        league_id = COMPETITION_LEAGUES.get(competition_id)
        provider_match_id = raw.get("match_id")
        fixture_id = _negative_id(FIXTURE_ID_OFFSET, provider_match_id)
        home_team_id = _negative_id(TEAM_ID_OFFSET, home_data.get("home_team_id"))
        away_team_id = _negative_id(TEAM_ID_OFFSET, away_data.get("away_team_id"))
        home_goals = raw.get("home_score")
        away_goals = raw.get("away_score")
        try:
            provider_match_id_int = int(str(provider_match_id))
            kickoff = datetime.fromisoformat(
                f"{raw.get('match_date')}T{str(raw.get('kick_off') or '00:00:00').split('.')[0]}"
            ).replace(tzinfo=UTC)
            season = int(str(season_data.get("season_name") or "")[:4])
        except (TypeError, ValueError):
            return {}
        if (
            league_id is None
            or fixture_id is None
            or home_team_id is None
            or away_team_id is None
            or isinstance(home_goals, bool)
            or not isinstance(home_goals, int)
            or isinstance(away_goals, bool)
            or not isinstance(away_goals, int)
        ):
            return {}
        home = str(home_data.get("home_team_name") or "").strip()
        away = str(away_data.get("away_team_name") or "").strip()
        if not home or not away:
            return {}
        return {
            "fixture_id": fixture_id,
            "provider_match_id": provider_match_id_int,
            "league_id": league_id,
            "season": season,
            "kickoff": kickoff,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team": home[:100],
            "away_team": away[:100],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "half_time_home_goals": None,
            "half_time_away_goals": None,
            "home_shots": None,
            "away_shots": None,
            "home_shots_on_target": None,
            "away_shots_on_target": None,
            "home_fouls": None,
            "away_fouls": None,
            "home_corners": None,
            "away_corners": None,
            "home_yellow_cards": None,
            "away_yellow_cards": None,
            "home_red_cards": None,
            "away_red_cards": None,
            "opening_home_odd": None,
            "opening_draw_odd": None,
            "opening_away_odd": None,
            "closing_home_odd": None,
            "closing_draw_odd": None,
            "closing_away_odd": None,
            "home_xg": None,
            "away_xg": None,
            "xg_source": None,
            "xg_provider_match_id": None,
            "xg_updated_at": None,
            "xg_confidence": None,
            "home_starting_xi": None,
            "away_starting_xi": None,
            "actual_result": (
                "HOME_WIN"
                if home_goals > away_goals
                else "AWAY_WIN" if away_goals > home_goals else "DRAW"
            ),
            "status": "FT",
            "data_source": "statsbomb_open",
        }

    @staticmethod
    def _apply_events(fixture: dict[str, Any], events: object) -> dict[str, Any] | None:
        if not isinstance(events, list):
            return None
        row = dict(fixture)
        home_provider_id = abs(int(row["home_team_id"])) - TEAM_ID_OFFSET
        away_provider_id = abs(int(row["away_team_id"])) - TEAM_ID_OFFSET
        counters: dict[int, dict[str, Any]] = {
            team_id: {
                "shots": 0,
                "shots_on_target": 0,
                "fouls": 0,
                "corners": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "xg": 0.0,
                "starting_xi": [],
            }
            for team_id in (home_provider_id, away_provider_id)
        }
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            team = event.get("team")
            if not isinstance(event_type, dict) or not isinstance(team, dict):
                continue
            team_id = team.get("id")
            if not isinstance(team_id, int):
                continue
            stats = counters.get(team_id)
            if stats is None:
                continue
            type_name = event_type.get("name")
            if type_name == "Shot":
                shot = event.get("shot")
                if not isinstance(shot, dict):
                    continue
                stats["shots"] += 1
                xg = shot.get("statsbomb_xg")
                if isinstance(xg, (int, float)) and not isinstance(xg, bool):
                    stats["xg"] += float(xg)
                outcome = shot.get("outcome")
                if isinstance(outcome, dict) and outcome.get("name") in {
                    "Goal",
                    "Saved",
                    "Saved to Post",
                }:
                    stats["shots_on_target"] += 1
            elif type_name == "Foul Committed":
                stats["fouls"] += 1
                StatsBombOpenDataClient._count_card(stats, event.get("foul_committed"))
            elif type_name == "Bad Behaviour":
                StatsBombOpenDataClient._count_card(stats, event.get("bad_behaviour"))
            elif type_name == "Pass":
                pass_data = event.get("pass")
                pass_type = (
                    pass_data.get("type") if isinstance(pass_data, dict) else None
                )
                if isinstance(pass_type, dict) and pass_type.get("name") == "Corner":
                    stats["corners"] += 1
            elif type_name == "Starting XI":
                tactics = event.get("tactics")
                lineup = tactics.get("lineup") if isinstance(tactics, dict) else None
                if isinstance(lineup, list):
                    stats["starting_xi"] = [
                        player_id
                        for item in lineup
                        if isinstance(item, dict)
                        and isinstance(item.get("player"), dict)
                        and (
                            player_id := _negative_id(
                                PLAYER_ID_OFFSET, item["player"].get("id")
                            )
                        )
                        is not None
                    ][:11]

        home = counters[home_provider_id]
        away = counters[away_provider_id]
        if home["shots"] == 0 and away["shots"] == 0:
            return None
        row.update(
            {
                "home_shots": home["shots"],
                "away_shots": away["shots"],
                "home_shots_on_target": home["shots_on_target"],
                "away_shots_on_target": away["shots_on_target"],
                "home_fouls": home["fouls"],
                "away_fouls": away["fouls"],
                "home_corners": home["corners"],
                "away_corners": away["corners"],
                "home_yellow_cards": home["yellow_cards"],
                "away_yellow_cards": away["yellow_cards"],
                "home_red_cards": home["red_cards"],
                "away_red_cards": away["red_cards"],
                "home_xg": round(float(home["xg"]), 4),
                "away_xg": round(float(away["xg"]), 4),
                "xg_source": "statsbomb_open",
                "xg_provider_match_id": str(row["provider_match_id"]),
                "xg_updated_at": datetime.now(UTC),
                "xg_confidence": 1.0,
                "home_starting_xi": home["starting_xi"] or None,
                "away_starting_xi": away["starting_xi"] or None,
            }
        )
        return row

    @staticmethod
    def _count_card(stats: dict[str, Any], value: object) -> None:
        if not isinstance(value, dict):
            return
        card = value.get("card")
        name = card.get("name") if isinstance(card, dict) else None
        if name in {"Yellow Card", "Second Yellow"}:
            stats["yellow_cards"] += 1
        if name in {"Red Card", "Second Yellow"}:
            stats["red_cards"] += 1


def database_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "provider_match_id"}
