"""Verify the historical bulk-ingest script end to end without a network."""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.football_data_csv import FootballDataCSVClient
from scripts.ingest_historical import (
    fetch_fixture_rows,
    ingest_league_seasons,
    season_start_year,
)

CSV_SAMPLE = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HS,AS,HST,AST,HC,AC,HF,AF,"
    "B365H,B365D,B365A,B365CH,B365CD,B365CA\n"
    "E0,16/08/2025,Manchester City,Arsenal,2,1,H,1,0,15,11,7,4,6,5,9,12,"
    "1.44,4.6,7.5,1.5,4.5,7.0\n"
    "E0,17/08/2025,Chelsea,Liverpool,0,0,D,0,0,8,10,3,5,4,6,11,9,"
    "3.2,3.1,2.25,3.0,3.2,2.4\n"
)


def _client(handler) -> FootballDataCSVClient:
    return FootballDataCSVClient(
        base_url="https://football-data.test",
        transport=httpx.MockTransport(handler),
    )


def _ok_handler(request: httpx.Request) -> httpx.Response:
    del request
    return httpx.Response(200, text=CSV_SAMPLE)


def test_season_start_year_supports_pair_and_calendar_codes() -> None:
    assert season_start_year("2425") == 2024
    assert season_start_year("2324") == 2023
    assert season_start_year("2024") == 2024
    with pytest.raises(ValueError):
        season_start_year("241")


def test_fetch_uses_canonical_client_and_normalizes_rows() -> None:
    fetched, rows = fetch_fixture_rows(_client(_ok_handler), "Premier_League", ["2425"])

    assert fetched == 2
    assert len(rows) == 2
    home_row = rows[0]
    assert home_row["league_id"] == 39
    assert home_row["season"] == 2024
    assert home_row["actual_result"] == "HOME_WIN"
    assert home_row["home_goals"] == 2
    assert home_row["away_goals"] == 1
    assert home_row["home_shots"] == 15
    assert home_row["half_time_home_goals"] == 1
    assert home_row["closing_home_odd"] == pytest.approx(1.5)
    assert home_row["status"] == "FT"
    assert home_row["data_source"] == "football_data_csv"
    assert home_row["fixture_id"] < 0
    assert rows[1]["actual_result"] == "DRAW"
    assert rows[1]["away_corners"] == 6


def test_unsupported_league_reports_supported_set() -> None:
    with pytest.raises(ValueError) as excinfo:
        fetch_fixture_rows(_client(_ok_handler), "UEFA_Champions_League", ["2425"])
    assert "Unsupported league 'UEFA_Champions_League'" in str(excinfo.value)


def test_ingest_round_trip_persists_rows_in_memory(monkeypatch) -> None:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    from app.db.models import Base

    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine, autoflush=False)

    from scripts import ingest_historical

    monkeypatch.setattr(ingest_historical, "SessionLocal", session_maker)

    fetched, persisted = ingest_league_seasons(
        _client(_ok_handler), "Premier_League", ["2425"]
    )
    assert fetched == 2
    assert persisted == 2

    with session_maker() as db:
        from app.db.historical_repository import HistoricalFixtureRepository

        all_fixtures = list(HistoricalFixtureRepository(db).get_all())
    assert len(all_fixtures) == 2
    assert all_fixtures[0].status == "FT"
    assert all_fixtures[0].data_source == "football_data_csv"
