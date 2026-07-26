from datetime import UTC, datetime

import httpx
import pytest

from app.services.football_data_csv import (
    FootballDataCSVClient,
    FootballDataDownloadError,
    FootballDataFormatError,
)


@pytest.mark.asyncio
async def test_standard_feed_is_normalized_with_stable_external_ids() -> None:
    content = (
        "\ufeffDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "T1,08/08/2025,21:30,Galatasaray,Fatih Karagumruk,3,0,H\n"
        "T1,09/08/2025,19:00,Goztepe,Fenerbahce,,,,\n"
    ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/mmz4281/2526/T1.csv"
        return httpx.Response(200, content=content)

    client = FootballDataCSVClient(
        base_url="https://data.test",
        transport=httpx.MockTransport(handler),
    )

    first = await client.get_completed_fixtures(203, 2025)
    second = await client.get_completed_fixtures(203, 2025)

    assert first.skipped_rows == 1
    assert len(first.fixtures) == 1
    fixture = first.fixtures[0]
    assert fixture["fixture_id"] < -(1 << 31)
    assert fixture["home_team_id"] < -(1 << 31)
    assert fixture["away_team_id"] < -(1 << 31)
    assert fixture["kickoff"] == datetime(2025, 8, 8, 18, 30, tzinfo=UTC)
    assert fixture["actual_result"] == "HOME_WIN"
    assert fixture["data_source"] == "football_data_csv"
    assert second.fixtures == first.fixtures


@pytest.mark.asyncio
async def test_rolling_russian_feed_filters_the_requested_season() -> None:
    content = (
        "Country,League,Season,Date,Time,Home,Away,HG,AG,Res\n"
        "Russia,Premier League,2024/2025,18/05/2025,17:00,Old A,Old B,1,0,H\n"
        "Russia,Premier League,2025/2026,17/05/2026,16:00,Sochi,"
        "Akhmat Grozny,1,1,D\n"
        "Russia,First Division,2025/2026,17/05/2026,16:00,Other A,"
        "Other B,2,0,H\n"
    ).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/new/RUS.csv"
        return httpx.Response(200, content=content)

    client = FootballDataCSVClient(
        base_url="https://data.test",
        transport=httpx.MockTransport(handler),
    )

    imported = await client.get_completed_fixtures(235, 2025)

    assert [(row["home_team"], row["away_team"]) for row in imported.fixtures] == [
        ("Sochi", "Akhmat Grozny")
    ]
    assert imported.fixtures[0]["actual_result"] == "DRAW"


@pytest.mark.asyncio
async def test_feed_schema_drift_fails_closed() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="unexpected,column\nvalue,row\n")

    client = FootballDataCSVClient(
        base_url="https://data.test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(FootballDataFormatError, match="missing columns"):
        await client.get_completed_fixtures(39, 2025)


@pytest.mark.asyncio
async def test_inconsistent_result_is_rejected() -> None:
    content = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,16/08/2025,15:00,Home,Away,2,0,A\n"
    ).encode()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    client = FootballDataCSVClient(
        base_url="https://data.test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(FootballDataFormatError, match="inconsistent scores"):
        await client.get_completed_fixtures(39, 2025)


@pytest.mark.asyncio
async def test_unpublished_feed_reports_404_without_retrying() -> None:
    requests = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(404)

    client = FootballDataCSVClient(
        base_url="https://data.test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(FootballDataDownloadError) as error:
        await client.get_completed_fixtures(39, 2026)

    assert error.value.status_code == 404
    assert requests == 1
