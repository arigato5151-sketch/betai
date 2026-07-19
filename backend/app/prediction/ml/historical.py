from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.models import HistoricalFixture
from app.prediction.ml.features import FeatureEngine


@dataclass(frozen=True)
class HistoricalFeatureContext:
    home_elo: float = 1500.0
    away_elo: float = 1500.0
    h2h_rates: dict[str, float | str] | None = None
    h2h_matches: list[dict[str, int]] | None = None


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
        season: int,
        before: datetime,
    ) -> HistoricalFeatureContext:
        league_matches = self.repository.get_league_history(
            league_id=league_id, season=season, before=before
        )
        elo_rows = [self._elo_row(fixture) for fixture in league_matches]
        ratings = FeatureEngine.calculate_elo_ratings(elo_rows)

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

        return HistoricalFeatureContext(
            home_elo=ratings.get(home_team_id, 1500.0),
            away_elo=ratings.get(away_team_id, 1500.0),
            h2h_rates=h2h_rates,
            h2h_matches=h2h_matches,
        )

    @staticmethod
    def _elo_row(fixture: HistoricalFixture) -> dict:
        return {
            "created_at": fixture.kickoff,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "actual_result": fixture.actual_result,
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
