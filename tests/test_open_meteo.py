from datetime import UTC, datetime

import httpx
import pytest

from app.providers.open_meteo import OpenMeteoClient, OpenMeteoError


def transport(payload: object, status_code: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["timezone"] == "UTC"
        assert request.url.params["hourly"] == (
            "temperature_2m,precipitation,wind_speed_10m"
        )
        return httpx.Response(status_code, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_forecast_weather_selects_hour_nearest_to_kickoff() -> None:
    client = OpenMeteoClient(
        enabled=True,
        transport=transport(
            {
                "hourly": {
                    "time": ["2026-08-05T19:00", "2026-08-05T20:00"],
                    "temperature_2m": [24.0, 22.5],
                    "precipitation": [0.0, 1.4],
                    "wind_speed_10m": [8.0, 13.2],
                }
            }
        ),
    )

    result = await client.get_weather(
        latitude=41.0,
        longitude=29.0,
        at=datetime(2026, 8, 5, 20, 20, tzinfo=UTC),
        now=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert result is not None
    assert result.features() == {
        "weather_temperature_c": 22.5,
        "weather_precipitation_mm": 1.4,
        "weather_wind_speed_kmh": 13.2,
        "weather_available": 1.0,
    }
    assert result.source == "open_meteo_forecast"


@pytest.mark.asyncio
async def test_historical_weather_uses_archive_before_2022() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(str(request.url.host))
        return httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2019-03-10T15:00"],
                    "temperature_2m": [9.0],
                    "precipitation": [0.2],
                    "wind_speed_10m": [11.0],
                }
            },
        )

    client = OpenMeteoClient(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    result = await client.get_weather(
        latitude=51.5,
        longitude=-0.1,
        at=datetime(2019, 3, 10, 15, tzinfo=UTC),
        now=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert result is not None
    assert result.source == "open_meteo_archive"
    assert requested_hosts == ["archive-api.open-meteo.com"]


@pytest.mark.asyncio
async def test_invalid_hourly_value_is_rejected() -> None:
    client = OpenMeteoClient(
        enabled=True,
        transport=transport(
            {
                "hourly": {
                    "time": ["2026-08-05T20:00"],
                    "temperature_2m": [22.0],
                    "precipitation": [None],
                    "wind_speed_10m": [10.0],
                }
            }
        ),
    )

    with pytest.raises(OpenMeteoError):
        await client.get_weather(
            latitude=41.0,
            longitude=29.0,
            at=datetime(2026, 8, 5, 20, tzinfo=UTC),
            now=datetime(2026, 8, 2, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_historical_forecast_gap_falls_back_to_archive() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(str(request.url.host))
        precipitation = (
            [None]
            if request.url.host == "historical-forecast-api.open-meteo.com"
            else [0.4]
        )
        return httpx.Response(
            200,
            json={
                "hourly": {
                    "time": ["2023-05-01T18:00"],
                    "temperature_2m": [17.0],
                    "precipitation": precipitation,
                    "wind_speed_10m": [9.0],
                }
            },
        )

    client = OpenMeteoClient(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    result = await client.get_weather(
        latitude=40.0,
        longitude=20.0,
        at=datetime(2023, 5, 1, 18, tzinfo=UTC),
        now=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert result is not None
    assert result.source == "open_meteo_archive"
    assert requested_hosts == [
        "historical-forecast-api.open-meteo.com",
        "archive-api.open-meteo.com",
    ]


@pytest.mark.asyncio
async def test_recent_past_forecast_gap_uses_historical_fallback() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(str(request.url.host))
        status = 400 if request.url.host == "api.open-meteo.com" else 200
        return httpx.Response(
            status,
            json={
                "hourly": {
                    "time": ["2026-08-01T18:00"],
                    "temperature_2m": [20.0],
                    "precipitation": [0.0],
                    "wind_speed_10m": [7.0],
                }
            },
        )

    client = OpenMeteoClient(
        enabled=True,
        transport=httpx.MockTransport(handler),
    )
    result = await client.get_weather(
        latitude=41.0,
        longitude=29.0,
        at=datetime(2026, 8, 1, 18, tzinfo=UTC),
        now=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert result is not None
    assert result.source == "open_meteo_historical_forecast"
    assert requested_hosts == [
        "api.open-meteo.com",
        "historical-forecast-api.open-meteo.com",
    ]
