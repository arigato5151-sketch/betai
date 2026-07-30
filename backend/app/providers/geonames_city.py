from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import geonamescache

from app.core.team_identity import stable_team_name_key

COUNTRY_CODE_ALIASES = {
    "czech republic": "CZ",
    "england": "GB",
    "ivory coast": "CI",
    "northern ireland": "GB",
    "republic of ireland": "IE",
    "russia": "RU",
    "scotland": "GB",
    "south korea": "KR",
    "turkiye": "TR",
    "uae": "AE",
    "usa": "US",
    "wales": "GB",
}

CITY_QUERY_ALIASES = {
    "moskva": "Moscow",
}


@dataclass(frozen=True, slots=True)
class ResolvedCity:
    latitude: float
    longitude: float
    city: str
    country_code: str
    geoname_id: int
    confidence: float


class GeoNamesCityResolver:
    """Resolve API venue cities against an offline GeoNames city dataset."""

    def __init__(self) -> None:
        self.cache = geonamescache.GeonamesCache()
        self.country_codes = self._country_index(self.cache.get_countries())

    def resolve(self, *, city: str, country: str) -> ResolvedCity | None:
        city = city.strip()
        country = country.strip()
        if not city or not country:
            return None
        query_city = CITY_QUERY_ALIASES.get(stable_team_name_key(city), city)
        country_code = self._country_code(country)
        if country_code is None:
            return None

        candidates: list[Mapping[str, Any]] = []
        for bucket in self.cache.get_cities_by_name(query_city):
            if not isinstance(bucket, Mapping):
                continue
            for row in bucket.values():
                if (
                    isinstance(row, Mapping)
                    and row.get("countrycode") == country_code
                    and stable_team_name_key(str(row.get("name") or ""))
                    == stable_team_name_key(query_city)
                ):
                    candidates.append(row)
        if not candidates:
            return None

        candidates.sort(
            key=lambda row: int(row.get("population") or 0),
            reverse=True,
        )
        selected = candidates[0]
        try:
            latitude = float(selected["latitude"])
            longitude = float(selected["longitude"])
            geoname_id = int(selected["geonameid"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
            or geoname_id <= 0
        ):
            return None
        return ResolvedCity(
            latitude=latitude,
            longitude=longitude,
            city=str(selected["name"]),
            country_code=country_code,
            geoname_id=geoname_id,
            # City-centre coordinates are intentionally lower-confidence than venues.
            confidence=0.70 if len(candidates) > 1 else 0.75,
        )

    def _country_code(self, country: str) -> str | None:
        normalized = stable_team_name_key(country)
        return COUNTRY_CODE_ALIASES.get(normalized) or self.country_codes.get(
            normalized
        )

    @staticmethod
    def _country_index(
        countries: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for country_code, row in countries.items():
            name = stable_team_name_key(str(row.get("name") or ""))
            if name:
                result[name] = country_code
                if name.startswith("the "):
                    result[name.removeprefix("the ")] = country_code
            iso3 = stable_team_name_key(str(row.get("iso3") or ""))
            if iso3:
                result[iso3] = country_code
            result[stable_team_name_key(country_code)] = country_code
        return result
