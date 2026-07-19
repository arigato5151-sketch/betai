from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import HistoricalFixture
from app.prediction.ml.features import FeatureEngine


@dataclass(frozen=True)
class HistoricalFeatureContext:
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    h2h_rates: dict[str, float | str] | None = None
    h2h_matches: list[dict[str, int]] | None = None
    home_matches_df: pd.DataFrame | None = None
    away_matches_df: pd.DataFrame | None = None
    home_previous_starting_xi: list[int] | None = None
    away_previous_starting_xi: list[int] | None = None


class HistoricalFeatureService:
    """Build point-in-time Elo and H2H inputs from completed fixtures."""

    def __init__(self, repository: HistoricalFixtureRepository):
        self.repository = repository

    def build_context(
        self,
        *,
        home_team_id: int,
        away_team_id: int,
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
            team_id=home_team_id, league_id=league_id, before=before
        )
        away_previous_starting_xi = self.repository.get_last_starting_xi(
            team_id=away_team_id, league_id=league_id, before=before
        )

        return HistoricalFeatureContext(
            home_elo=ratings.get(home_team_id, 1500.0),
            away_elo=ratings.get(away_team_id, 1500.0),
            h2h_rates=h2h_rates,
            h2h_matches=h2h_matches,
            home_matches_df=self._team_matches_frame(home_matches, home_team_id),
            away_matches_df=self._team_matches_frame(away_matches, away_team_id),
            home_previous_starting_xi=home_previous_starting_xi,
            away_previous_starting_xi=away_previous_starting_xi,
        )

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
