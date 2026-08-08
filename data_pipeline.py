"""Historical football-data.co.uk CSV ingestion pipeline.

This module intentionally keeps downloading and dataframe preparation separate so
network access can be mocked in CI and data handling can be tested deterministically.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


class FootballDataFetcher:
    """Fetch and normalize completed-match CSV files by league and season."""

    BASE_URL: str = "https://www.football-data.co.uk/mmz4281"
    LEAGUE_MAP: dict[str, str] = {
        "Premier_League": "E0",
        "Championship": "E1",
        "La_Liga": "SP1",
        "Serie_A": "I1",
        "Serie_B": "I2",
        "Bundesliga": "D1",
        "Bundesliga_2": "D2",
        "Ligue_1": "F1",
        "Ligue_2": "F2",
        "Super_Lig": "T1",
        "Eredivisie": "N1",
        "Liga_Portugal": "P1",
        "Jupiler_Pro_League": "B1",
        "Scottish_Premiership": "SC0",
        "Greek_Super_League": "G1",
    }
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
    )

    def fetch_raw_csv(self, season: str, league_code: str) -> str:
        """Download a league-season CSV and return its text after HTTP validation."""
        url = f"{self.BASE_URL}/{season}/{league_code}.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def process_data(self, csv_content: str) -> pd.DataFrame:
        """Keep usable target columns and discard rows without core match details."""
        dataframe = pd.read_csv(StringIO(csv_content))
        available_columns = [
            column for column in self.REQUIRED_COLUMNS if column in dataframe.columns
        ]
        filtered = dataframe.loc[:, available_columns].copy()

        # A row cannot represent a completed match without these four values.
        return filtered.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    def get_league_data(self, season: str, league_key: str) -> pd.DataFrame:
        """Fetch and prepare a supported league's historical season data."""
        try:
            league_code = self.LEAGUE_MAP[league_key]
        except KeyError as exc:
            supported = ", ".join(self.LEAGUE_MAP)
            raise ValueError(
                f"Unsupported league '{league_key}'. Supported leagues: {supported}"
            ) from exc

        raw_csv = self.fetch_raw_csv(season, league_code)
        return self.process_data(raw_csv)
