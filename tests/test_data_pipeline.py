from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipeline import FootballDataFetcher

MOCK_CSV_DATA = """Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,HC,AC,HF,AF,Unused
11/08/2023,Burnley,Man City,0,3,A,6,17,1,8,6,5,11,8,ignored
12/08/2023,Arsenal,Nottingham Forest,2,1,H,15,8,7,2,8,3,12,10,ignored
13/08/2023,Missing FC,Away FC,,,D,5,5,2,2,4,4,9,9,ignored
"""


def test_fetch_raw_csv_success() -> None:
    fetcher = FootballDataFetcher()
    response = MagicMock()
    response.text = MOCK_CSV_DATA

    with patch("data_pipeline.requests.get", return_value=response) as mock_get:
        content = fetcher.fetch_raw_csv("2324", "E0")

    mock_get.assert_called_once_with(
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        timeout=30,
    )
    response.raise_for_status.assert_called_once_with()
    assert content == MOCK_CSV_DATA


def test_process_data_structure() -> None:
    dataframe = FootballDataFetcher().process_data(MOCK_CSV_DATA)

    assert list(dataframe.columns) == list(FootballDataFetcher.REQUIRED_COLUMNS)
    assert len(dataframe) == 2
    assert dataframe.iloc[0]["HomeTeam"] == "Burnley"
    assert dataframe.iloc[0]["AwayTeam"] == "Man City"
    assert dataframe.iloc[0]["FTHG"] == 0
    assert dataframe.iloc[0]["FTAG"] == 3
    assert isinstance(dataframe, pd.DataFrame)


def test_get_league_data_invalid_league() -> None:
    with pytest.raises(ValueError, match="Unsupported league"):
        FootballDataFetcher().get_league_data("2324", "Unknown_League")


def test_pipeline_integration_mocked() -> None:
    fetcher = FootballDataFetcher()

    with patch.object(
        fetcher, "fetch_raw_csv", return_value=MOCK_CSV_DATA
    ) as mock_fetch:
        dataframe = fetcher.get_league_data("2324", "Premier_League")

    mock_fetch.assert_called_once_with("2324", "E0")
    assert dataframe[["HomeTeam", "AwayTeam"]].to_dict("records") == [
        {"HomeTeam": "Burnley", "AwayTeam": "Man City"},
        {"HomeTeam": "Arsenal", "AwayTeam": "Nottingham Forest"},
    ]
