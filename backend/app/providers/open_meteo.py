from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app.core.config import settings

MAX_PAYLOAD_BYTES = 1024 * 1024
HOURLY_VARIABLES = "temperature_2m,precipitation,wind_speed_10m"


class OpenMeteoError(RuntimeError):
    """Raised when Open-Meteo cannot provide a valid hourly observation."""


@dataclass(frozen=True)
class WeatherObservation:
    temperature_c: float
    precipitation_mm: float
    wind_speed_kmh: float
    source: str
    observed_at: datetime
    fetched_at: datetime

    def features(self) -> dict[str, float]:
        return {
            "weather_temperature_c": self.temperature_c,
            "weather_precipitation_mm": self.precipitation_mm,
            "weather_wind_speed_kmh": self.wind_speed_kmh,
            "weather_available": 1.0,
        }


class OpenMeteoClient:
    def __init__(
        self,
        *,
        forecast_url: str | None = None,
        historical_forecast_url: str | None = None,
        archive_url: str | None = None,
        timeout_seconds: float | None = None,
        concurrency: int | None = None,
        enabled: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.forecast_url = forecast_url or settings.OPEN_METEO_FORECAST_URL
        self.historical_forecast_url = (
            historical_forecast_url or settings.OPEN_METEO_HISTORICAL_FORECAST_URL
        )
        self.archive_url = archive_url or settings.OPEN_METEO_ARCHIVE_URL
        self.timeout_seconds = timeout_seconds or settings.OPEN_METEO_TIMEOUT_SECONDS
        self.concurrency = concurrency or settings.OPEN_METEO_CONCURRENCY
        self.enabled = settings.OPEN_METEO_ENABLED if enabled is None else enabled
        self.transport = transport

    async def get_weather(
        self,
        *,
        latitude: float,
        longitude: float,
        at: datetime,
        now: datetime | None = None,
    ) -> WeatherObservation | None:
        if not self.enabled:
            return None
        self._validate_coordinates(latitude, longitude)
        target = self._as_utc(at)
        current = self._as_utc(now or datetime.now(UTC))
        endpoint, source = self._endpoint(target, current)
        params: dict[str, str | float] = {
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "hourly": HOURLY_VARIABLES,
            "start_date": target.date().isoformat(),
            "end_date": target.date().isoformat(),
            "timezone": "UTC",
        }
        attempts = [(endpoint, source)]
        if target < current and source == "open_meteo_forecast":
            attempts.append(
                (
                    self.historical_forecast_url,
                    "open_meteo_historical_forecast",
                )
            )
        if target < current:
            attempts.append((self.archive_url, "open_meteo_archive"))

        last_error: OpenMeteoError | None = None
        seen_urls: set[str] = set()
        for attempt_url, attempt_source in attempts:
            if attempt_url in seen_urls:
                continue
            seen_urls.add(attempt_url)
            try:
                payload = await self._get_json(attempt_url, params=params)
                return self._parse(
                    payload,
                    target=target,
                    source=attempt_source,
                )
            except OpenMeteoError as exc:
                last_error = exc
        raise last_error or OpenMeteoError("Open-Meteo lookup failed")

    async def get_many(
        self,
        requests: list[tuple[int, float, float, datetime]],
    ) -> tuple[dict[int, WeatherObservation], list[int]]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def fetch(
            fixture_id: int,
            latitude: float,
            longitude: float,
            kickoff: datetime,
        ) -> tuple[int, WeatherObservation | None]:
            try:
                async with semaphore:
                    observation = await self.get_weather(
                        latitude=latitude,
                        longitude=longitude,
                        at=kickoff,
                    )
            except (OpenMeteoError, ValueError):
                observation = None
            return fixture_id, observation

        results = await asyncio.gather(*(fetch(*request) for request in requests))
        observations = {
            fixture_id: observation
            for fixture_id, observation in results
            if observation is not None
        }
        failures = [
            fixture_id for fixture_id, observation in results if observation is None
        ]
        return observations, failures

    def _endpoint(self, target: datetime, now: datetime) -> tuple[str, str]:
        if target >= now - timedelta(days=5):
            return self.forecast_url, "open_meteo_forecast"
        if target.year >= 2022:
            return (
                self.historical_forecast_url,
                "open_meteo_historical_forecast",
            )
        return self.archive_url, "open_meteo_archive"

    async def _get_json(self, url: str, *, params: dict[str, str | float]) -> object:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BetAIPlatform/1.0 open-meteo-client",
                },
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OpenMeteoError("Open-Meteo request failed") from exc
        if len(response.content) > MAX_PAYLOAD_BYTES:
            raise OpenMeteoError("Open-Meteo payload exceeds safety limit")
        try:
            return response.json()
        except ValueError as exc:
            raise OpenMeteoError("Open-Meteo returned invalid JSON") from exc

    @classmethod
    def _parse(
        cls,
        payload: object,
        *,
        target: datetime,
        source: str,
    ) -> WeatherObservation:
        if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
            raise OpenMeteoError("Open-Meteo hourly payload is missing")
        hourly: dict[str, Any] = payload["hourly"]
        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise OpenMeteoError("Open-Meteo hourly timestamps are missing")

        parsed_times: list[datetime] = []
        for value in times:
            try:
                parsed_times.append(cls._as_utc(datetime.fromisoformat(str(value))))
            except ValueError as exc:
                raise OpenMeteoError(
                    "Open-Meteo returned an invalid timestamp"
                ) from exc
        index = min(
            range(len(parsed_times)),
            key=lambda item: abs((parsed_times[item] - target).total_seconds()),
        )
        if abs((parsed_times[index] - target).total_seconds()) > 3600:
            raise OpenMeteoError("Open-Meteo has no hour close to kickoff")

        temperature = cls._number_at(hourly, "temperature_2m", index, -80.0, 65.0)
        precipitation = cls._number_at(hourly, "precipitation", index, 0.0, 500.0)
        wind_speed = cls._number_at(hourly, "wind_speed_10m", index, 0.0, 300.0)
        return WeatherObservation(
            temperature_c=round(temperature, 2),
            precipitation_mm=round(precipitation, 2),
            wind_speed_kmh=round(wind_speed, 2),
            source=source,
            observed_at=parsed_times[index],
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _number_at(
        hourly: dict[str, Any],
        name: str,
        index: int,
        minimum: float,
        maximum: float,
    ) -> float:
        values = hourly.get(name)
        if not isinstance(values, list) or index >= len(values):
            raise OpenMeteoError(f"Open-Meteo variable is missing: {name}")
        value = values[index]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OpenMeteoError(f"Open-Meteo variable is invalid: {name}")
        numeric = float(value)
        if not minimum <= numeric <= maximum:
            raise OpenMeteoError(f"Open-Meteo variable is out of range: {name}")
        return numeric

    @staticmethod
    def _validate_coordinates(latitude: float, longitude: float) -> None:
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError("Weather coordinates are out of range")

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
