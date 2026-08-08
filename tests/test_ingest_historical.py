"""Verify the historical bulk-ingest script end to end without a network."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scripts.ingest_historical import (
    FootballDataFetcher,
    build_fixture_rows,
    fetch_league_season,
    season_start_year,
)

CSV_SAMPLE = (
    "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HS,AS,HST,AST,HC,AC,HF,AF,B365H,B365D,B365A,PSH,PSD,PSA\n"
    "16/08/2025,Manchester City,Arsenal,2,1,H,1,0,15,11,7,4,6,5,9,12,1.44,4.6,7.5,1.5,4.5,7.0\n"
    "17/08/2025,Chelsea,Liverpool,0,0,D,0,0,8,10,3,5,4,6,11,9,3.2,3.1,2.25,3.0,3.2,2.4\n"
)


class _FakeFetcher(FootballDataFetcher):
    def __init__(self, csv_text: str) -> None:
        self.csv_text = csv_text

    def fetch_raw_csv(self, season: str, league_code: str) -> str:
        del season, league_code
        return self.csv_text


def test_season_start_year_supports_pair_and_calendar_codes() -> None:
    assert season_start_year("2425") == 2024
    assert season_start_year("2324") == 2023
    assert season_start_year("2024") == 2024
    with pytest.raises(ValueError):
        season_start_year("241")


def test_fetch_keeps_stats_and_odds_and_drops_incomplete_rows() -> None:
    frame = fetch_league_season(_FakeFetcher(CSV_SAMPLE), "Premier_League", "2425")
    assert len(frame) == 2
    assert "HTHG" in frame
    assert frame.loc[0, "FTHG"] == 2


def test_build_fixture_rows_inserts_results_stats_odds_and_resolved_ids() -> None:
    frame = fetch_league_season(_FakeFetcher(CSV_SAMPLE), "Premier_League", "2425")
    rows = build_fixture_rows(frame, league_id=39, season_start=2024)

    assert len(rows) == 2
    home_row = rows[0]
    assert home_row["actual_result"] == "HOME_WIN"
    assert home_row["league_id"] in (39,)
    assert home_row["season"] == 2024
    assert home_row["home_goals"] == 2
    assert home_row["away_goals"] == 1
    assert home_row["home_shots"] == 15
    assert home_row["half_time_home_goals"] == 1
    assert home_row["closing_home_odd"] == pytest.approx(1.5)
    assert rows[1]["actual_result"] == "DRAW"
    assert rows[1]["away_corners"] == 6


def test_upsert_many_round_trip_in_memory() -> None:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    from app.db.models import Base

    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine, autoflush=False)

    frame = fetch_league_season(_FakeFetcher(CSV_SAMPLE), "Premier_League", "2425")
    rows = build_fixture_rows(frame, league_id=39, season_start=2024)

    with session_maker() as db:
        from app.db.historical_repository import HistoricalFixtureRepository

        persisted = HistoricalFixtureRepository(db).upsert_many(rows)

    assert persisted == 2
    with session_maker() as db:
        from app.db.historical_repository import HistoricalFixtureRepository

        all_fixtures = list(HistoricalFixtureRepository(db).get_all())
    assert len(all_fixtures) == 2
    assert all_fixtures[0].status == "completed"
    assert all_fixtures[0].data_source == "football_data_csv"
