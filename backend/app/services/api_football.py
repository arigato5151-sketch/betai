import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import httpx

from app.core.config import settings
from app.core.api_mode import get_api_mode
from app.core.demo_data import (
    DEMO_FIXTURE_ODDS,
    DEMO_LIVE_FIXTURES,
    DEMO_TEAM_STATS,
    DEMO_UPCOMING_FIXTURES,
)
from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS, LEAGUE_PRIORITY
from app.core.exceptions import (
    APIFootballException,
    APIDataError,
    APIRateLimitError,
    APITimeoutError,
)
from app.services.cache import cache
from app.prediction.stats_engine import build_team_profile

logger = logging.getLogger("bet-ai-pro.api_football")


class APIFootballClient:
    def __init__(self):
        self.api_key = settings.API_FOOTBALL_KEY
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "x-apisports-key": self.api_key,
            "x-rapidapi-host": "v3.football.api-sports.io",
        }

    def _is_demo_key(self) -> bool:
        return get_api_mode(self.api_key) == "demo"

    async def _request_with_retry(
        self, path: str, params: Dict, retries: int = 3, base_backoff: float = 0.5
    ) -> Optional[dict]:
        """
        Perform HTTP request with exponential backoff retry strategy.
        - Max 3 retries
        - Exponential backoff: 0.5s, 1s, 2s
        - Rate limit (429) gets extended backoff
        - Logs all retry attempts with timestamp and status
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: APIFootballException | None = None

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(
                    headers=self.headers, timeout=10.0
                ) as client:
                    response = await client.get(url, params=params)

                    if response.status_code == 200:
                        logger.info(
                            f"✓ API request successful: {path} (attempt {attempt + 1})"
                        )
                        return response.json()

                    # Handle rate limiting with extended backoff
                    if response.status_code == 429:
                        try:
                            retry_after = int(response.headers.get("Retry-After", 60))
                        except (TypeError, ValueError):
                            retry_after = 60
                        retry_after = max(0, min(retry_after, 120))
                        last_error = APIRateLimitError(path, retry_after)
                        logger.warning(
                            f"⚠ Rate limited (429) on {path}. Retry after {retry_after}s (attempt {attempt + 1}/{retries})"
                        )
                        if attempt < retries - 1:
                            await asyncio.sleep(retry_after)
                        continue

                    # Other error statuses
                    logger.warning(
                        f"⚠ API error {response.status_code} for {path} (attempt {attempt + 1}/{retries})"
                    )
                    last_error = APIDataError(
                        path, response.status_code, response.text[:200]
                    )
                    if 400 <= response.status_code < 500:
                        break

            except httpx.TimeoutException as e:
                logger.warning(
                    f"⚠ Timeout on {path} (attempt {attempt + 1}/{retries}): {str(e)}"
                )
                last_error = APITimeoutError(path, retries)
            except httpx.RequestError as e:
                logger.warning(
                    f"⚠ Request failed on {path} (attempt {attempt + 1}/{retries}): {str(e)}"
                )
                last_error = APIDataError(path, 0, str(e))

            # Exponential backoff between retries
            if attempt < retries - 1:
                wait_time = base_backoff * (2**attempt)
                logger.debug(
                    f"Sleeping {wait_time}s before retry {attempt + 2}/{retries}"
                )
                await asyncio.sleep(wait_time)

        logger.error(
            f"✗ Failed to fetch {path} after {retries} attempts. Last error: {last_error}"
        )
        return None

    async def get_live_fixtures(self, league_id: Optional[int] = None) -> List[Dict]:
        if self._is_demo_key():
            fixtures = DEMO_LIVE_FIXTURES
            if league_id:
                fixtures = [f for f in fixtures if f["league_id"] == league_id]
            return fixtures

        cache_key = f"live:{league_id}"
        cached = await cache.get("fixtures", cache_key)
        if cached:
            return cached

        params = {"live": "all"}
        if league_id:
            params["league"] = str(league_id)

        data = await self._request_with_retry("fixtures", params)
        if data:
            raw = data.get("response", [])
            fixtures = [
                self._normalize_fixture(item, is_live=True) for item in raw[:12]
            ]
            await cache.set("fixtures", cache_key, fixtures, 900)  # 15 min TTL
            return fixtures

        return DEMO_LIVE_FIXTURES

    async def get_upcoming_fixtures(
        self, days: int = 10, limit: int = 100
    ) -> List[Dict]:
        if self._is_demo_key():
            return self._filter_allowed_leagues(DEMO_UPCOMING_FIXTURES)[:limit]

        cache_key = f"upcoming:{days}:{limit}"
        cached = await cache.get("fixtures", cache_key)
        if cached:
            return cached

        fixtures: List[Dict] = []
        today = date.today()

        for offset in range(days):
            match_day = (today + timedelta(days=offset)).isoformat()
            data = await self._request_with_retry(
                "fixtures",
                {"date": match_day, "timezone": "Europe/Istanbul"},
            )
            if not data:
                continue

            for item in data.get("response", []):
                league_id = item.get("league", {}).get("id")
                if league_id not in ALLOWED_LEAGUE_IDS:
                    continue
                status = item.get("fixture", {}).get("status", {}).get("short")
                if status not in {"NS", "TBD", "PST"}:
                    continue
                fixtures.append(self._normalize_fixture(item, is_live=False))

        fixtures = self._filter_allowed_leagues(fixtures)
        fixtures.sort(
            key=lambda f: (
                LEAGUE_PRIORITY.get(f.get("league_id"), 99),
                f.get("kickoff") or "",
            )
        )

        result = (
            fixtures[:limit]
            if fixtures
            else self._filter_allowed_leagues(DEMO_UPCOMING_FIXTURES)[:limit]
        )
        await cache.set("fixtures", cache_key, result, 900)  # 15 min TTL
        return result

    @staticmethod
    def _filter_allowed_leagues(fixtures: List[Dict]) -> List[Dict]:
        return [f for f in fixtures if f.get("league_id") in ALLOWED_LEAGUE_IDS]

    async def get_fixture_by_id(self, fixture_id: int) -> Optional[Dict]:
        if self._is_demo_key():
            return next(
                (f for f in DEMO_UPCOMING_FIXTURES if f["fixture_id"] == fixture_id),
                None,
            )

        # Check in upcoming fixtures cache or live fixtures
        data = await self._request_with_retry("fixtures", {"id": str(fixture_id)})
        if data:
            items = data.get("response", [])
            if items:
                status = items[0].get("fixture", {}).get("status", {}).get("short", "")
                is_live = status in {"1H", "2H", "HT", "ET", "BT", "P", "LIVE"}
                return self._normalize_fixture(items[0], is_live=is_live)

        fixtures = await self.get_upcoming_fixtures(days=14, limit=200)
        return next((f for f in fixtures if f["fixture_id"] == fixture_id), None)

    async def get_h2h(
        self, home_team_id: int, away_team_id: int, last: int = 5
    ) -> Dict[str, float | str]:
        if self._is_demo_key() or not home_team_id or not away_team_id:
            return {
                "home_win_rate": 0.33,
                "draw_rate": 0.33,
                "home_loss_rate": 0.34,
                "source": "demo_default",
            }

        cache_key = f"h2h:{home_team_id}:{away_team_id}:{last}"
        cached = await cache.get("h2h", cache_key)
        if cached:
            return cached

        data = await self._request_with_retry(
            "fixtures/headtohead",
            {"h2h": f"{home_team_id}-{away_team_id}", "last": str(last)},
        )
        if not data:
            return {
                "home_win_rate": 0.33,
                "draw_rate": 0.33,
                "home_loss_rate": 0.34,
                "source": "fallback",
            }

        wins = draws = losses = 0
        for item in data.get("response", [])[:last]:
            match_result = self._result_for_team(item, home_team_id)
            if match_result == "W":
                wins += 1
            elif match_result == "D":
                draws += 1
            else:
                losses += 1

        total = max(1, wins + draws + losses)
        result: Dict[str, float | str] = {
            "home_win_rate": wins / total,
            "draw_rate": draws / total,
            "home_loss_rate": losses / total,
            "source": "api_football_h2h",
        }
        await cache.set("h2h", cache_key, result, 86400)  # 24 hours TTL
        return result

    async def get_team_last_matches_df(
        self, team_id: int, last: int = 5
    ) -> pd.DataFrame:
        if self._is_demo_key() or not team_id:
            return pd.DataFrame()

        cache_key = f"last:{team_id}:{last}"
        cached = await cache.get("stats", cache_key)

        if cached:
            # Reconstruct DataFrame from dict representation
            df = pd.read_json(cached)
            if not df.empty:
                df["match_date"] = pd.to_datetime(df["match_date"])
            return df

        data = await self._request_with_retry(
            "fixtures",
            {"team": str(team_id), "last": str(last), "status": "FT"},
        )
        rows: List[Dict] = []
        if data:
            for item in data.get("response", []):
                row = self._team_match_row(item, team_id)
                if row:
                    rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("match_date", ascending=True).reset_index(drop=True)
            # Store in cache as JSON string
            # We serialize timestamps to isoformat for storage
            df_to_cache = df.copy()
            df_to_cache["match_date"] = df_to_cache["match_date"].dt.strftime(
                "%Y-%m-%d"
            )
            await cache.set(
                "stats", cache_key, df_to_cache.to_json(), 21600
            )  # 6 hours TTL

        return df

    @staticmethod
    def _result_for_team(item: Dict, team_id: int) -> str:
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home = teams.get("home", {})
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            return "D"
        if int(home.get("id", 0)) == team_id:
            if hg > ag:
                return "W"
            if hg < ag:
                return "L"
            return "D"
        if ag > hg:
            return "W"
        if ag < hg:
            return "L"
        return "D"

    def _team_match_row(self, item: Dict, team_id: int) -> Optional[Dict]:
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        fixture = item.get("fixture", {})
        home = teams.get("home", {})
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            return None

        is_home = int(home.get("id", 0)) == team_id
        gf = int(hg) if is_home else int(ag)
        ga = int(ag) if is_home else int(hg)

        if gf > ga:
            result = "W"
            points = 3.0
        elif gf == ga:
            result = "D"
            points = 1.0
        else:
            result = "L"
            points = 0.0

        kickoff = fixture.get("date")
        try:
            match_date = pd.Timestamp(kickoff.replace("Z", "+00:00")).normalize()
        except Exception:
            match_date = pd.Timestamp.today().normalize()

        return {
            "match_date": match_date,
            "goals_for": gf,
            "goals_against": ga,
            "result": result,
            "points": points,
            "clean_sheet": 1 if ga == 0 else 0,
            "scoring": 1 if gf > 0 else 0,
        }

    async def get_team_statistics(
        self, league_id: int, season: int, team_id: int, venue: str = "total"
    ) -> Dict:
        if self._is_demo_key():
            if team_id in DEMO_TEAM_STATS:
                profile = dict(DEMO_TEAM_STATS[team_id])
                profile["venue"] = venue
                profile["source"] = "demo_professional_profile"
                return profile
            return build_team_profile(None, venue=venue)

        cache_key = f"stats:{league_id}:{season}:{team_id}:{venue}"
        cached = await cache.get("stats", cache_key)
        if cached:
            return cached

        data = await self._request_with_retry(
            "teams/statistics",
            {"league": str(league_id), "season": str(season), "team": str(team_id)},
        )
        if data:
            res_data = data.get("response", {})
            if res_data:
                profile = build_team_profile(res_data, venue=venue)
                await cache.set("stats", cache_key, profile, 21600)  # 6 hours TTL
                return profile

        profile = build_team_profile(None, venue=venue)
        await cache.set("stats", cache_key, profile, 21600)
        return profile

    async def get_fixture_market(self, fixture_id: int) -> Optional[Dict]:
        if self._is_demo_key() and fixture_id in DEMO_FIXTURE_ODDS:
            # Fallback to devigged synthetic
            from app.prediction.value_calc import ValueCalc

            home = DEMO_FIXTURE_ODDS[fixture_id]
            return ValueCalc.default_market(home)

        cache_key = f"market:{fixture_id}"
        cached = await cache.get("odds", cache_key)
        if cached:
            return cached

        data = await self._request_with_retry("odds", {"fixture": str(fixture_id)})
        if data:
            response = data.get("response", [])
            if response and response[0].get("bookmakers"):
                from app.prediction.value_calc import ValueCalc

                market = ValueCalc.best_market_from_bookmakers(
                    response[0]["bookmakers"]
                )
                if market:
                    await cache.set("odds", cache_key, market, 300)  # 5 min TTL
                    return market
        return None

    async def get_fixture_odds(self, fixture_id: int) -> float:
        market = await self.get_fixture_market(fixture_id)
        if market:
            return market["raw_odds"]["HOME_WIN"]
        if self._is_demo_key() and fixture_id in DEMO_FIXTURE_ODDS:
            return DEMO_FIXTURE_ODDS[fixture_id]
        return 1.85

    def _normalize_fixture(self, item: Dict, is_live: bool = False) -> Dict:
        fixture = item.get("fixture", {})
        league = item.get("league", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        status_info = fixture.get("status", {})

        home = teams.get("home", {})
        away = teams.get("away", {})
        home_goals = goals.get("home")
        away_goals = goals.get("away")
        score = None
        if home_goals is not None and away_goals is not None:
            score = f"{home_goals} - {away_goals}"

        kickoff_raw = fixture.get("date")
        status_short = status_info.get("short")
        live_now = is_live or status_short in {
            "1H",
            "2H",
            "HT",
            "ET",
            "BT",
            "P",
            "LIVE",
        }

        return {
            "fixture_id": fixture.get("id"),
            "league": league.get("name", "Unknown League"),
            "home_team": home.get("name", "Home"),
            "away_team": away.get("name", "Away"),
            "home_team_id": home.get("id", 0),
            "away_team_id": away.get("id", 0),
            "league_id": league.get("id", 0),
            "season": league.get("season", 2024),
            "minute": status_info.get("elapsed"),
            "score": score,
            "kickoff": kickoff_raw,
            "kickoff_label": self._format_kickoff(kickoff_raw),
            "status": status_short,
            "is_live": live_now,
            "is_demo": self._is_demo_key(),
        }

    def _format_kickoff(self, iso_date: Optional[str]) -> Optional[str]:
        if not iso_date:
            return None
        try:
            normalized = iso_date.replace("Z", "+00:00")
            kickoff = datetime.fromisoformat(normalized)
            return kickoff.strftime("%d.%m %H:%M")
        except ValueError:
            return iso_date[:16]

    async def get_fixture_prefill(self, fixture_id: int) -> Optional[Dict]:
        fixture = await self.get_fixture_by_id(fixture_id)
        if not fixture:
            return None

        season = fixture.get("season", 2024)
        league_id = fixture["league_id"]
        home_stats = await self.get_team_statistics(
            league_id, season, fixture["home_team_id"], venue="home"
        )
        away_stats = await self.get_team_statistics(
            league_id, season, fixture["away_team_id"], venue="away"
        )
        market = await self.get_fixture_market(fixture_id)
        odd = (
            market["raw_odds"]["HOME_WIN"]
            if market
            else await self.get_fixture_odds(fixture_id)
        )

        data_quality = (
            "demo"
            if self._is_demo_key()
            else (
                "live"
                if home_stats.get("source") == "api_football_season_stats"
                and away_stats.get("source") == "api_football_season_stats"
                else "fallback"
            )
        )

        return {
            "fixture": fixture,
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "odd": odd,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "market_1x2": market,
            "auto_filled": True,
            "data_quality": data_quality,
            "data_methodology": {
                "stats": "API-Football sezon istatistikleri (ev/deplasman ayrımı, form decay)",
                "odds": "Bahisçi 1X2 oranları + proportional devig",
                "model": "Poisson + Dixon-Coles düzeltmesi",
            },
        }
