from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from app.core.config import settings
from app.db.models import HistoricalFixture
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import HistoricalFeatureService
from app.prediction.stats_engine import build_team_profile


@dataclass(frozen=True)
class HistoricalTrainingRow:
    fixture_id: int
    league_id: int
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
        return build_team_profile(
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

    def build(
        self, fixtures: Iterable[HistoricalFixture]
    ) -> list[HistoricalTrainingRow]:
        ordered = sorted(
            fixtures,
            key=lambda row: (row.kickoff, row.fixture_id),
        )
        team_history: dict[tuple[int, int], deque[HistoricalFixture]] = defaultdict(
            lambda: deque(maxlen=self.recent_match_count)
        )
        h2h_history: dict[tuple[int, int, int], deque[HistoricalFixture]] = defaultdict(
            lambda: deque(maxlen=10)
        )
        ratings_by_league: dict[int, dict[int, float]] = defaultdict(dict)
        active_season: dict[int, int] = {}
        previous_lineups: dict[tuple[int, int], list[int] | None] = {}
        rows: list[HistoricalTrainingRow] = []
        pending_updates: list[HistoricalFixture] = []
        current_kickoff: datetime | None = None

        for fixture in ordered:
            if current_kickoff is not None and fixture.kickoff != current_kickoff:
                self._apply_batch_updates(
                    pending_updates,
                    team_history=team_history,
                    h2h_history=h2h_history,
                    ratings_by_league=ratings_by_league,
                    previous_lineups=previous_lineups,
                )
                pending_updates.clear()
            current_kickoff = fixture.kickoff

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
                        "home_previous_starting_xi": previous_lineups.get(home_key),
                        "away_previous_starting_xi": previous_lineups.get(away_key),
                    },
                    fixture_date=fixture.kickoff,
                    league_id=fixture.league_id,
                )
                rows.append(
                    HistoricalTrainingRow(
                        fixture_id=fixture.fixture_id,
                        league_id=league_id,
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
        h2h_history: dict[tuple[int, int, int], deque[HistoricalFixture]],
        ratings_by_league: dict[int, dict[int, float]],
        previous_lineups: dict[tuple[int, int], list[int] | None],
    ) -> None:
        for fixture in fixtures:
            league_id = fixture.league_id
            home_key = (league_id, fixture.home_team_id)
            away_key = (league_id, fixture.away_team_id)
            team_history[home_key].append(fixture)
            team_history[away_key].append(fixture)
            h2h_history[self._h2h_key(fixture)].append(fixture)
            previous_lineups[home_key] = fixture.home_starting_xi
            previous_lineups[away_key] = fixture.away_starting_xi
            self._update_elo(ratings_by_league[league_id], fixture)
