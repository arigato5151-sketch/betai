from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import NotRequired, TypedDict

from app.core.config import settings
from app.db.models import (
    HistoricalFixture,
    HistoricalPlayerPerformance,
    TeamLocation,
)
from app.db.player_context_repository import haversine_distance_km
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import HistoricalFeatureService, PlayerRatingValue
from app.prediction.player_impact import PlayerImpactCalculator, TeamStrengthImpact
from app.prediction.stats_engine import build_team_profile


class PlayerObservation(TypedDict):
    kickoff: datetime
    minutes: float
    appearances: float
    goals: float
    assists: float
    rating: NotRequired[float]


@dataclass(frozen=True)
class HistoricalTrainingRow:
    fixture_id: int
    league_id: int
    home_team_id: int
    away_team_id: int
    actual_result: str
    feature_snapshot: dict[str, float]
    feature_schema_version: str
    feature_snapshot_at: datetime
    created_at: datetime
    training_source: str = "historical_fixture"


class HistoricalTrainingDataBuilder:
    """Build point-in-time training rows without querying future fixtures."""

    def __init__(
        self,
        *,
        recent_match_count: int | None = None,
        minimum_team_history: int | None = None,
    ) -> None:
        self.recent_match_count = recent_match_count or settings.RECENT_FORM_MATCH_COUNT
        self.minimum_team_history = (
            minimum_team_history
            if minimum_team_history is not None
            else settings.HISTORICAL_TRAINING_MIN_TEAM_MATCHES
        )

    @staticmethod
    def _profile(fixtures: list[HistoricalFixture], team_id: int, venue: str) -> dict:
        frame = HistoricalFeatureService._team_matches_frame(fixtures, team_id)
        if frame.empty:
            return build_team_profile(None, venue)

        goals_for = float(frame["goals_for"].mean())
        goals_against = float(frame["goals_against"].mean())
        form = "".join(frame.sort_values("match_date")["result"].astype(str).tolist())
        clean_sheets = int(frame["clean_sheet"].sum())
        failed_to_score = int((frame["goals_for"] == 0).sum())
        played = len(frame)
        profile = build_team_profile(
            {
                "form": form,
                "goals": {
                    "for": {"average": {venue: goals_for}},
                    "against": {"average": {venue: goals_against}},
                },
                "clean_sheet": {venue: clean_sheets},
                "failed_to_score": {venue: failed_to_score},
                "fixtures": {"played": {venue: played}},
            },
            venue,
        )
        observations: list[tuple[datetime, float]] = []
        for fixture in fixtures:
            kickoff = HistoricalTrainingDataBuilder._as_utc_datetime(fixture.kickoff)
            value = (
                fixture.home_xg if fixture.home_team_id == team_id else fixture.away_xg
            )
            if (
                kickoff is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) <= 15.0
            ):
                observations.append((kickoff, float(value)))
        if observations:
            reference = max(kickoff for kickoff, _ in observations)
            weighted = [
                (
                    value,
                    math.exp(
                        -settings.GOAL_TIME_DECAY_FACTOR
                        * max(0.0, (reference - kickoff).total_seconds() / 86400.0)
                    ),
                )
                for kickoff, value in observations
            ]
            denominator = math.fsum(weight for _, weight in weighted)
            if denominator > 0.0:
                profile["xg"] = round(
                    math.fsum(value * weight for value, weight in weighted)
                    / denominator,
                    4,
                )
        return profile

    @staticmethod
    def _update_elo(
        ratings: dict[int, float],
        fixture: HistoricalFixture,
    ) -> None:
        home_rating = ratings.setdefault(fixture.home_team_id, 1500.0)
        away_rating = ratings.setdefault(fixture.away_team_id, 1500.0)
        expected_home = 1.0 / (
            1.0
            + 10.0
            ** (
                (away_rating - (home_rating + settings.ELO_HOME_ADVANTAGE_POINTS))
                / 400.0
            )
        )
        actual_home = (
            1.0
            if fixture.actual_result == "HOME_WIN"
            else 0.5 if fixture.actual_result == "DRAW" else 0.0
        )
        delta = settings.ELO_K_FACTOR * (actual_home - expected_home)
        ratings[fixture.home_team_id] = home_rating + delta
        ratings[fixture.away_team_id] = away_rating - delta

    @staticmethod
    def _h2h_key(fixture: HistoricalFixture) -> tuple[int, int, int]:
        return (
            fixture.league_id,
            min(fixture.home_team_id, fixture.away_team_id),
            max(fixture.home_team_id, fixture.away_team_id),
        )

    @staticmethod
    def _as_utc_datetime(value: object) -> datetime | None:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @classmethod
    def _performance_timeline(
        cls,
        performances: Iterable[HistoricalPlayerPerformance] | None,
    ) -> list[tuple[datetime, HistoricalPlayerPerformance]]:
        timeline: list[tuple[datetime, HistoricalPlayerPerformance]] = []
        for performance in performances or ():
            kickoff = cls._as_utc_datetime(performance.kickoff)
            if kickoff is not None:
                timeline.append((kickoff, performance))
        return sorted(
            timeline,
            key=lambda item: (
                item[0],
                item[1].fixture_id,
                item[1].player_id,
            ),
        )

    @staticmethod
    def _advance_player_rating_state(
        timeline: list[tuple[datetime, HistoricalPlayerPerformance]],
        cursor: int,
        *,
        before: datetime,
        state: dict[int, dict[int, deque[PlayerObservation]]],
    ) -> int:
        while cursor < len(timeline) and timeline[cursor][0] < before:
            kickoff, performance = timeline[cursor]
            cursor += 1
            rating = HistoricalFeatureService._valid_rating(performance.rating)
            if performance.team_id <= 0 or performance.player_id <= 0:
                continue
            minutes = HistoricalFeatureService._valid_non_negative(performance.minutes)
            goals = HistoricalFeatureService._valid_non_negative(performance.goals)
            assists = HistoricalFeatureService._valid_non_negative(performance.assists)
            observation: PlayerObservation = {
                "kickoff": kickoff,
                "minutes": minutes,
                "appearances": float(minutes > 0.0 or rating is not None),
                "goals": goals,
                "assists": assists,
            }
            if rating is not None:
                observation["rating"] = rating
            elif max(observation["appearances"], minutes / 90.0) <= 0.0 or (
                goals + assists <= 0.0
            ):
                continue
            team_ratings = state.setdefault(performance.team_id, {})
            observations = team_ratings.get(performance.player_id)
            if observations is None:
                observations = deque(maxlen=settings.PLAYER_IMPACT_LOOKBACK_MATCHES)
                team_ratings[performance.player_id] = observations
            observations.append(observation)
        return cursor

    @staticmethod
    def _player_rating_map(
        state: dict[int, dict[int, deque[PlayerObservation]]],
        team_id: int,
        *,
        before: datetime,
    ) -> dict[int, PlayerRatingValue]:
        decay = float(settings.PLAYER_IMPACT_RATING_DECAY)
        freshness_cutoff = before - timedelta(
            days=int(settings.HISTORICAL_FORM_MAX_AGE_DAYS)
        )
        ratings: dict[int, PlayerRatingValue] = {}
        for player_id, observations in state.get(team_id, {}).items():
            fresh_observations = [
                observation
                for observation in observations
                if freshness_cutoff <= observation["kickoff"] < before
            ]
            if not fresh_observations:
                continue
            weighted_total = 0.0
            total_weight = 0.0
            minutes = appearances = goals = assists = 0.0
            for age, observation in enumerate(reversed(fresh_observations)):
                rating = observation.get("rating")
                if rating is not None:
                    weight = decay**age
                    weighted_total += rating * weight
                    total_weight += weight
                minutes += observation["minutes"]
                appearances += observation["appearances"]
                goals += observation["goals"]
                assists += observation["assists"]

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

    @staticmethod
    def _valid_lineup(lineup: object) -> tuple[int, ...]:
        if not isinstance(lineup, list):
            return ()
        player_ids = tuple(
            player_id
            for player_id in lineup
            if isinstance(player_id, int)
            and not isinstance(player_id, bool)
            and player_id > 0
        )
        return (
            player_ids if len(player_ids) == 11 and len(set(player_ids)) == 11 else ()
        )

    @classmethod
    def _player_impact(
        cls,
        ratings: Mapping[int, object],
        reference_lineup: object,
        current_lineup: object,
    ) -> TeamStrengthImpact:
        reference = cls._valid_lineup(reference_lineup) or tuple(
            PlayerImpactCalculator.derive_reference_lineup(ratings) or ()
        )
        current = cls._valid_lineup(current_lineup)
        missing = tuple(
            player_id for player_id in reference if player_id not in current
        )
        return PlayerImpactCalculator.assess(
            ratings,
            reference or None,
            current or None,
            missing if reference and current else None,
        )

    @staticmethod
    def _location_index(
        team_locations: Iterable[TeamLocation] | None,
    ) -> dict[tuple[str, int], TeamLocation]:
        locations: dict[tuple[str, int], TeamLocation] = {}
        for location in team_locations or ():
            source = str(location.data_source or "").strip().lower()
            if source:
                locations[(source, location.team_id)] = location
        return locations

    @staticmethod
    def _away_travel_distance_km(
        fixture: HistoricalFixture,
        locations: dict[tuple[str, int], TeamLocation],
    ) -> float:
        source = str(fixture.data_source or "api_football").strip().lower()
        origin = locations.get((source, fixture.away_team_id))
        destination = locations.get((source, fixture.home_team_id))
        if origin is None or destination is None:
            return 0.0
        try:
            return haversine_distance_km(
                origin.latitude,
                origin.longitude,
                destination.latitude,
                destination.longitude,
            )
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _prune_schedule(
        cls,
        schedule: deque[HistoricalFixture],
        *,
        since: datetime,
    ) -> None:
        while schedule:
            kickoff = cls._as_utc_datetime(schedule[0].kickoff)
            if kickoff is not None and kickoff >= since:
                return
            schedule.popleft()

    def build(
        self,
        fixtures: Iterable[HistoricalFixture],
        *,
        player_performances: Iterable[HistoricalPlayerPerformance] | None = None,
        team_locations: Iterable[TeamLocation] | None = None,
    ) -> list[HistoricalTrainingRow]:
        ordered = sorted(
            fixtures,
            key=lambda row: (
                self._as_utc_datetime(row.kickoff)
                or datetime.min.replace(tzinfo=timezone.utc),
                row.fixture_id,
            ),
        )
        team_history: dict[tuple[int, int], deque[HistoricalFixture]] = defaultdict(
            lambda: deque(maxlen=self.recent_match_count)
        )
        schedule_history: dict[int, deque[HistoricalFixture]] = defaultdict(deque)
        h2h_history: dict[tuple[int, int, int], deque[HistoricalFixture]] = defaultdict(
            lambda: deque(maxlen=10)
        )
        ratings_by_league: dict[int, dict[int, float]] = defaultdict(dict)
        active_season: dict[int, int] = {}
        previous_lineups: dict[int, list[int]] = {}
        performance_timeline = self._performance_timeline(player_performances)
        performance_cursor = 0
        player_rating_state: dict[
            int,
            dict[int, deque[PlayerObservation]],
        ] = {}
        locations = self._location_index(team_locations)
        rows: list[HistoricalTrainingRow] = []
        pending_updates: list[HistoricalFixture] = []
        current_kickoff: datetime | None = None

        for fixture in ordered:
            fixture_kickoff = self._as_utc_datetime(fixture.kickoff)
            if fixture_kickoff is None:
                continue
            if current_kickoff is not None and fixture_kickoff != current_kickoff:
                self._apply_batch_updates(
                    pending_updates,
                    team_history=team_history,
                    schedule_history=schedule_history,
                    h2h_history=h2h_history,
                    ratings_by_league=ratings_by_league,
                    previous_lineups=previous_lineups,
                )
                pending_updates.clear()
            current_kickoff = fixture_kickoff
            performance_cursor = self._advance_player_rating_state(
                performance_timeline,
                performance_cursor,
                before=fixture_kickoff,
                state=player_rating_state,
            )

            league_id = fixture.league_id
            ratings = ratings_by_league[league_id]
            previous_season = active_season.get(league_id)
            if previous_season is not None and fixture.season != previous_season:
                retention = 1.0 - settings.ELO_SEASON_REGRESSION
                ratings_by_league[league_id] = ratings = {
                    team_id: 1500.0 + (rating - 1500.0) * retention
                    for team_id, rating in ratings.items()
                }
            active_season[league_id] = fixture.season

            home_key = (league_id, fixture.home_team_id)
            away_key = (league_id, fixture.away_team_id)
            home_history = list(team_history[home_key])
            away_history = list(team_history[away_key])
            prior_h2h = list(h2h_history[self._h2h_key(fixture)])

            if (
                len(home_history) >= self.minimum_team_history
                and len(away_history) >= self.minimum_team_history
            ):
                home_frame = HistoricalFeatureService._team_matches_frame(
                    home_history, fixture.home_team_id
                )
                away_frame = HistoricalFeatureService._team_matches_frame(
                    away_history, fixture.away_team_id
                )
                h2h_matches = [
                    HistoricalFeatureService._from_home_team_perspective(
                        row, fixture.home_team_id
                    )
                    for row in reversed(prior_h2h)
                ]
                h2h_rates = (
                    HistoricalFeatureService._rates(h2h_matches) if h2h_matches else {}
                )
                schedule_start = fixture_kickoff - timedelta(
                    days=FeatureEngine.fatigue_schedule_horizon_days()
                )
                home_schedule_history = schedule_history[fixture.home_team_id]
                away_schedule_history = schedule_history[fixture.away_team_id]
                self._prune_schedule(home_schedule_history, since=schedule_start)
                self._prune_schedule(away_schedule_history, since=schedule_start)
                home_schedule = HistoricalFeatureService._schedule_frame(
                    list(home_schedule_history),
                    team_id=fixture.home_team_id,
                    since=schedule_start,
                    before=fixture_kickoff,
                )
                away_schedule = HistoricalFeatureService._schedule_frame(
                    list(away_schedule_history),
                    team_id=fixture.away_team_id,
                    since=schedule_start,
                    before=fixture_kickoff,
                )
                home_previous_lineup = previous_lineups.get(fixture.home_team_id)
                away_previous_lineup = previous_lineups.get(fixture.away_team_id)
                home_player_impact = self._player_impact(
                    self._player_rating_map(
                        player_rating_state,
                        fixture.home_team_id,
                        before=fixture_kickoff,
                    ),
                    home_previous_lineup,
                    fixture.home_starting_xi,
                )
                away_player_impact = self._player_impact(
                    self._player_rating_map(
                        player_rating_state,
                        fixture.away_team_id,
                        before=fixture_kickoff,
                    ),
                    away_previous_lineup,
                    fixture.away_starting_xi,
                )
                feature_snapshot = FeatureEngine.build_inference_features(
                    home_stats=self._profile(
                        home_history, fixture.home_team_id, "home"
                    ),
                    away_stats=self._profile(
                        away_history, fixture.away_team_id, "away"
                    ),
                    home_matches_df=home_frame,
                    away_matches_df=away_frame,
                    h2h_rates=h2h_rates,
                    h2h_matches=h2h_matches,
                    home_elo=ratings.get(fixture.home_team_id, 1500.0),
                    away_elo=ratings.get(fixture.away_team_id, 1500.0),
                    lineup_context={
                        "home_starting_xi": fixture.home_starting_xi,
                        "away_starting_xi": fixture.away_starting_xi,
                        "home_previous_starting_xi": home_previous_lineup,
                        "away_previous_starting_xi": away_previous_lineup,
                    },
                    fixture_date=fixture.kickoff,
                    league_id=fixture.league_id,
                    home_team_id=fixture.home_team_id,
                    away_team_id=fixture.away_team_id,
                    home_schedule_df=home_schedule,
                    away_schedule_df=away_schedule,
                    away_travel_distance_km=self._away_travel_distance_km(
                        fixture,
                        locations,
                    ),
                    opening_odds={
                        "HOME_WIN": fixture.opening_home_odd,
                        "DRAW": fixture.opening_draw_odd,
                        "AWAY_WIN": fixture.opening_away_odd,
                    },
                    current_odds={
                        "HOME_WIN": fixture.closing_home_odd,
                        "DRAW": fixture.closing_draw_odd,
                        "AWAY_WIN": fixture.closing_away_odd,
                    },
                    home_player_impact=home_player_impact,
                    away_player_impact=away_player_impact,
                )
                rows.append(
                    HistoricalTrainingRow(
                        fixture_id=fixture.fixture_id,
                        league_id=league_id,
                        home_team_id=fixture.home_team_id,
                        away_team_id=fixture.away_team_id,
                        actual_result=fixture.actual_result,
                        feature_snapshot=feature_snapshot,
                        feature_schema_version=FeatureEngine.SCHEMA_VERSION,
                        feature_snapshot_at=fixture.kickoff,
                        created_at=fixture.kickoff,
                        training_source=(
                            f"historical_fixture:"
                            f"{fixture.data_source or 'api_football'}"
                        ),
                    )
                )

            # Simultaneous fixtures share the same pre-kickoff state.
            pending_updates.append(fixture)

        self._apply_batch_updates(
            pending_updates,
            team_history=team_history,
            schedule_history=schedule_history,
            h2h_history=h2h_history,
            ratings_by_league=ratings_by_league,
            previous_lineups=previous_lineups,
        )
        return rows

    def _apply_batch_updates(
        self,
        fixtures: list[HistoricalFixture],
        *,
        team_history: dict[tuple[int, int], deque[HistoricalFixture]],
        schedule_history: dict[int, deque[HistoricalFixture]],
        h2h_history: dict[tuple[int, int, int], deque[HistoricalFixture]],
        ratings_by_league: dict[int, dict[int, float]],
        previous_lineups: dict[int, list[int]],
    ) -> None:
        for fixture in fixtures:
            league_id = fixture.league_id
            home_key = (league_id, fixture.home_team_id)
            away_key = (league_id, fixture.away_team_id)
            team_history[home_key].append(fixture)
            team_history[away_key].append(fixture)
            schedule_history[fixture.home_team_id].append(fixture)
            schedule_history[fixture.away_team_id].append(fixture)
            h2h_history[self._h2h_key(fixture)].append(fixture)
            home_lineup = self._valid_lineup(fixture.home_starting_xi)
            away_lineup = self._valid_lineup(fixture.away_starting_xi)
            # Missing provider data must not erase the last confirmed XI.
            if home_lineup:
                previous_lineups[fixture.home_team_id] = list(home_lineup)
            if away_lineup:
                previous_lineups[fixture.away_team_id] = list(away_lineup)
            self._update_elo(ratings_by_league[league_id], fixture)
