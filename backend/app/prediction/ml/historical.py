from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.team_identity import normalize_team_name
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import HistoricalFixture, HistoricalPlayerPerformance
from app.db.player_context_repository import PlayerContextRepository
from app.prediction.ml.features import FeatureEngine

PlayerRatingValue = float | dict[str, float]


@dataclass(frozen=True)
class HistoricalFeatureContext:
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    home_elo_available: bool = False
    away_elo_available: bool = False
    feature_provenance: dict[str, dict[str, object]] = field(default_factory=dict)
    h2h_rates: dict[str, float | str] | None = None
    h2h_matches: list[dict[str, int]] | None = None
    home_matches_df: pd.DataFrame | None = None
    away_matches_df: pd.DataFrame | None = None
    home_previous_starting_xi: list[int] | None = None
    away_previous_starting_xi: list[int] | None = None
    home_schedule_df: pd.DataFrame | None = None
    away_schedule_df: pd.DataFrame | None = None
    home_player_ratings: dict[int, PlayerRatingValue] = field(default_factory=dict)
    away_player_ratings: dict[int, PlayerRatingValue] = field(default_factory=dict)
    away_travel_distance_km: float = 0.0
    travel_context_available: bool = False
    travel_provenance: dict[str, object] | None = None


class HistoricalFeatureService:
    """Build point-in-time Elo and H2H inputs from completed fixtures."""

    def __init__(
        self,
        repository: HistoricalFixtureRepository,
        player_context_repository: PlayerContextRepository | None = None,
    ):
        self.repository = repository
        self.player_context_repository = player_context_repository

    def build_context(
        self,
        *,
        home_team_id: int,
        away_team_id: int,
        home_team_name: str | None = None,
        away_team_name: str | None = None,
        league_id: int,
        before: datetime,
        recent_match_count: int = 5,
        elo_k_factor: float = 32.0,
        elo_home_advantage_points: float = 0.0,
        elo_season_regression: float = 0.0,
    ) -> HistoricalFeatureContext:
        league_matches = self.repository.get_league_history(
            league_id=league_id, before=before
        )
        home_team_id = self._resolve_team_id(
            league_matches, home_team_id, home_team_name
        )
        away_team_id = self._resolve_team_id(
            league_matches, away_team_id, away_team_name
        )
        elo_rows = [self._elo_row(fixture) for fixture in league_matches]
        ratings = FeatureEngine.calculate_elo_ratings(
            elo_rows,
            k_factor=elo_k_factor,
            home_advantage_points=elo_home_advantage_points,
            season_regression=elo_season_regression,
        )

        h2h_fixtures = self.repository.get_h2h(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            before=before,
        )
        h2h_matches = [
            self._from_home_team_perspective(fixture, home_team_id)
            for fixture in h2h_fixtures
        ]
        h2h_rates = self._rates(h2h_matches) if h2h_matches else None
        home_matches = self.repository.get_team_history(
            team_id=home_team_id,
            league_id=league_id,
            before=before,
            limit=recent_match_count,
        )
        away_matches = self.repository.get_team_history(
            team_id=away_team_id,
            league_id=league_id,
            before=before,
            limit=recent_match_count,
        )
        home_previous_starting_xi = self.repository.get_last_starting_xi(
            team_id=home_team_id, before=before
        )
        away_previous_starting_xi = self.repository.get_last_starting_xi(
            team_id=away_team_id, before=before
        )
        schedule_start = before - timedelta(
            days=FeatureEngine.fatigue_schedule_horizon_days()
        )
        home_schedule = self.repository.get_team_schedule(
            team_id=home_team_id,
            since=schedule_start,
            before=before,
        )
        away_schedule = self.repository.get_team_schedule(
            team_id=away_team_id,
            since=schedule_start,
            before=before,
        )

        home_player_ratings: dict[int, PlayerRatingValue] = {}
        away_player_ratings: dict[int, PlayerRatingValue] = {}
        away_travel_distance_km = 0.0
        travel_context_available = False
        travel_provenance: dict[str, object] | None = None
        if self.player_context_repository is not None:
            home_player_ratings = self._player_rating_map(home_team_id, before)
            away_player_ratings = self._player_rating_map(away_team_id, before)
            try:
                home_location = self.player_context_repository.get_team_location(
                    home_team_id
                )
                away_location = self.player_context_repository.get_team_location(
                    away_team_id
                )
                if (
                    home_location is not None
                    and away_location is not None
                    and home_location.latitude is not None
                    and home_location.longitude is not None
                    and away_location.latitude is not None
                    and away_location.longitude is not None
                ):
                    away_travel_distance_km = (
                        self.player_context_repository.travel_distance_km(
                            away_team_id,
                            home_team_id,
                        )
                    )
                    travel_context_available = True
                    sources = sorted(
                        {
                            home_location.location_source,
                            away_location.location_source,
                        }
                    )
                    latest = max(
                        home_location.updated_at,
                        away_location.updated_at,
                    )
                    travel_provenance = {
                        "source": (
                            "geonames_city"
                            if "geonames_city" in sources
                            else "curated_team_locations"
                        ),
                        "captured_at": latest.isoformat(),
                        "confidence": min(
                            home_location.confidence,
                            away_location.confidence,
                        ),
                        "is_fallback": "geonames_city" in sources,
                    }
            except ValueError:
                # Synthetic/legacy negative team IDs have no provider location mapping.
                away_travel_distance_km = 0.0

        feature_provenance: dict[str, dict[str, object]] = {}
        if home_team_id in ratings:
            feature_provenance["home_elo"] = self._elo_provenance(
                home_matches, recent_match_count
            )
        if away_team_id in ratings:
            feature_provenance["away_elo"] = self._elo_provenance(
                away_matches, recent_match_count
            )

        return HistoricalFeatureContext(
            home_elo=ratings.get(home_team_id, 1500.0),
            away_elo=ratings.get(away_team_id, 1500.0),
            home_elo_available=home_team_id in ratings,
            away_elo_available=away_team_id in ratings,
            feature_provenance=feature_provenance,
            h2h_rates=h2h_rates,
            h2h_matches=h2h_matches,
            home_matches_df=self._team_matches_frame(home_matches, home_team_id),
            away_matches_df=self._team_matches_frame(away_matches, away_team_id),
            home_previous_starting_xi=home_previous_starting_xi,
            away_previous_starting_xi=away_previous_starting_xi,
            home_schedule_df=self._schedule_frame(
                home_schedule,
                team_id=home_team_id,
                since=schedule_start,
                before=before,
            ),
            away_schedule_df=self._schedule_frame(
                away_schedule,
                team_id=away_team_id,
                since=schedule_start,
                before=before,
            ),
            home_player_ratings=home_player_ratings,
            away_player_ratings=away_player_ratings,
            away_travel_distance_km=away_travel_distance_km,
            travel_context_available=travel_context_available,
            travel_provenance=travel_provenance,
        )

    @staticmethod
    def _elo_provenance(
        fixtures: list[HistoricalFixture],
        expected_count: int,
    ) -> dict[str, object]:
        latest = max((fixture.kickoff for fixture in fixtures), default=None)
        confidence = min(1.0, len(fixtures) / max(1, expected_count))
        return {
            "source": "historical_fixtures",
            "captured_at": latest.isoformat() if latest is not None else None,
            "confidence": round(confidence, 4),
            "is_fallback": False,
        }

    def _player_rating_map(
        self,
        team_id: int,
        before: datetime,
    ) -> dict[int, PlayerRatingValue]:
        if self.player_context_repository is None:
            return {}
        try:
            rows = self.player_context_repository.get_team_performances_before(
                team_id,
                before,
            )
        except ValueError:
            return {}

        observations: dict[int, list[HistoricalPlayerPerformance]] = defaultdict(list)
        before_at = self._as_utc_timestamp(before)
        if before_at is None:
            return {}
        # A per-player cutoff prevents transferred/departed players from ranking
        # highly merely because they accumulated minutes earlier in the season.
        freshness_cutoff = before_at - pd.Timedelta(
            days=int(settings.HISTORICAL_FORM_MAX_AGE_DAYS)
        )

        for row in rows:
            kickoff_at = self._as_utc_timestamp(row.kickoff)
            if (
                kickoff_at is None
                or kickoff_at < freshness_cutoff
                or kickoff_at >= before_at
                or row.player_id <= 0
            ):
                continue
            observations[row.player_id].append(row)

        lookback = int(settings.PLAYER_IMPACT_LOOKBACK_MATCHES)
        decay = float(settings.PLAYER_IMPACT_RATING_DECAY)
        ratings: dict[int, PlayerRatingValue] = {}
        for player_id, player_rows in observations.items():
            ordered = sorted(
                player_rows,
                key=lambda row: (
                    self._as_utc_timestamp(row.kickoff)
                    or pd.Timestamp.min.tz_localize("UTC"),
                    row.fixture_id,
                ),
            )[-lookback:]
            weighted_total = 0.0
            total_weight = 0.0
            minutes = appearances = goals = assists = 0.0
            for age, row in enumerate(reversed(ordered)):
                rating = self._valid_rating(row.rating)
                if rating is not None:
                    weight = decay**age
                    weighted_total += rating * weight
                    total_weight += weight
                row_minutes = self._valid_non_negative(row.minutes)
                minutes += row_minutes
                appearances += float(row_minutes > 0.0 or rating is not None)
                goals += self._valid_non_negative(row.goals)
                assists += self._valid_non_negative(row.assists)

            player_summary = {
                "minutes": minutes,
                "appearances": appearances,
                "goals": goals,
                "assists": assists,
            }
            if total_weight > 0.0:
                player_summary["rating"] = round(weighted_total / total_weight, 6)
                ratings[player_id] = player_summary
            elif max(appearances, minutes / 90.0) > 0.0 and goals + assists > 0.0:
                ratings[player_id] = player_summary
        return ratings

    @classmethod
    def _schedule_frame(
        cls,
        fixtures: list[HistoricalFixture],
        *,
        team_id: int,
        since: datetime,
        before: datetime,
    ) -> pd.DataFrame:
        columns = [
            "match_date",
            "fixture_id",
            "league_id",
            "is_home",
            "opponent_team_id",
        ]
        since_at = cls._as_utc_timestamp(since)
        before_at = cls._as_utc_timestamp(before)
        if since_at is None or before_at is None:
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, object]] = []
        for fixture in fixtures:
            kickoff_at = cls._as_utc_timestamp(fixture.kickoff)
            if kickoff_at is None or not since_at <= kickoff_at < before_at:
                continue
            is_home = fixture.home_team_id == team_id
            is_away = fixture.away_team_id == team_id
            if not is_home and not is_away:
                continue
            rows.append(
                {
                    "match_date": kickoff_at,
                    "fixture_id": fixture.fixture_id,
                    "league_id": fixture.league_id,
                    "is_home": is_home,
                    "opponent_team_id": (
                        fixture.away_team_id if is_home else fixture.home_team_id
                    ),
                }
            )
        return pd.DataFrame(rows, columns=columns).sort_values(
            ["match_date", "fixture_id"],
            ignore_index=True,
        )

    @staticmethod
    def _as_utc_timestamp(value: object) -> pd.Timestamp | None:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if pd.isna(timestamp):
            return None
        try:
            if timestamp.tzinfo is None:
                return timestamp.tz_localize("UTC")
            return timestamp.tz_convert("UTC")
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _valid_rating(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            rating = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return rating if math.isfinite(rating) and rating > 0.0 else None

    @staticmethod
    def _valid_non_negative(value: Any) -> float:
        if isinstance(value, bool) or value is None:
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return number if math.isfinite(number) and number >= 0.0 else 0.0

    @staticmethod
    def _resolve_team_id(
        fixtures: list[HistoricalFixture],
        requested_team_id: int,
        requested_team_name: str | None,
    ) -> int:
        known_ids = {
            team_id
            for fixture in fixtures
            for team_id in (fixture.home_team_id, fixture.away_team_id)
        }
        if requested_team_id in known_ids or not requested_team_name:
            return requested_team_id

        target = normalize_team_name(requested_team_name)
        candidates: set[int] = set()
        for fixture in fixtures:
            if normalize_team_name(fixture.home_team) == target:
                candidates.add(fixture.home_team_id)
            if normalize_team_name(fixture.away_team) == target:
                candidates.add(fixture.away_team_id)
        return candidates.pop() if len(candidates) == 1 else requested_team_id

    @staticmethod
    def _elo_row(fixture: HistoricalFixture) -> dict:
        return {
            "created_at": fixture.kickoff,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "actual_result": fixture.actual_result,
            "season": fixture.season,
        }

    @staticmethod
    def _from_home_team_perspective(
        fixture: HistoricalFixture, current_home_team_id: int
    ) -> dict[str, int]:
        if fixture.home_team_id == current_home_team_id:
            return {
                "home_goals": fixture.home_goals,
                "away_goals": fixture.away_goals,
            }
        return {
            "home_goals": fixture.away_goals,
            "away_goals": fixture.home_goals,
        }

    @staticmethod
    def _rates(matches: list[dict[str, int]]) -> dict[str, float | str]:
        total = len(matches)
        wins = sum(row["home_goals"] > row["away_goals"] for row in matches)
        draws = sum(row["home_goals"] == row["away_goals"] for row in matches)
        return {
            "home_win_rate": wins / total,
            "draw_rate": draws / total,
            "home_loss_rate": (total - wins - draws) / total,
            "source": "historical_fixtures",
        }

    @staticmethod
    def _team_matches_frame(
        fixtures: list[HistoricalFixture], team_id: int
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for fixture in fixtures:
            is_home = fixture.home_team_id == team_id
            goals_for = fixture.home_goals if is_home else fixture.away_goals
            goals_against = fixture.away_goals if is_home else fixture.home_goals
            if goals_for > goals_against:
                result, points = "W", 3.0
            elif goals_for == goals_against:
                result, points = "D", 1.0
            else:
                result, points = "L", 0.0

            match_date = pd.Timestamp(fixture.kickoff)
            if match_date.tzinfo is None:
                match_date = match_date.tz_localize("UTC")
            else:
                match_date = match_date.tz_convert("UTC")
            rows.append(
                {
                    "match_date": match_date,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "result": result,
                    "points": points,
                    "clean_sheet": int(goals_against == 0),
                    "scoring": int(goals_for > 0),
                }
            )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
