from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from app.core.config import settings
from app.core.team_identity import normalize_team_name, stable_team_name_key

MAX_WIKIDATA_PAYLOAD_BYTES = 2 * 1024 * 1024
_SPORT_DESCRIPTIONS = (
    "association football",
    "football club",
    "football team",
    "soccer club",
    "multi-sport club",
)
_EXCLUDED_DESCRIPTIONS = (
    "women",
    "youth",
    "reserve",
    "academy",
    "supporters",
)


class WikidataError(RuntimeError):
    """Raised when the optional public Wikidata resolver cannot be used."""


@dataclass(frozen=True, slots=True)
class WikidataTeamLocation:
    latitude: float
    longitude: float
    club_qid: str
    location_qid: str
    location_name: str | None
    method: str
    confidence: float


class WikidataTeamLocationClient:
    """Resolve a club's home venue coordinates from Wikidata, failing closed."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        timeout_seconds: float | None = None,
        request_interval_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_url = (api_url or settings.WIKIDATA_API_URL).strip()
        self.timeout_seconds = timeout_seconds or settings.WIKIDATA_TIMEOUT_SECONDS
        self.request_interval_seconds = (
            settings.WIKIDATA_REQUEST_INTERVAL_SECONDS
            if request_interval_seconds is None
            else max(0.0, request_interval_seconds)
        )
        self.transport = transport

    async def resolve(
        self,
        *,
        team_name: str,
        country: str | None = None,
    ) -> WikidataTeamLocation | None:
        query_name = normalize_team_name(team_name)
        if not query_name:
            return None
        search = await self._request(
            {
                "action": "wbsearchentities",
                "search": f"{query_name} football club",
                "language": "en",
                "uselang": "en",
                "type": "item",
                "limit": "8",
            }
        )
        candidate = self._select_candidate(
            search.get("search"),
            team_name=query_name,
            country=country,
        )
        if candidate is None:
            return None

        club_qid = candidate["id"]
        club_payload = await self._entities([club_qid])
        club = club_payload.get(club_qid)
        if not isinstance(club, Mapping):
            return None

        direct = self._coordinate(club)
        if direct is not None:
            return WikidataTeamLocation(
                *direct,
                club_qid=club_qid,
                location_qid=club_qid,
                location_name=self._label(club),
                method="club_coordinate",
                confidence=0.78,
            )

        related: list[tuple[str, str]] = []
        for property_id, method in (("P115", "home_venue"), ("P159", "headquarters")):
            for qid in self._entity_ids(club, property_id):
                if qid not in {item[0] for item in related}:
                    related.append((qid, method))
        if not related:
            return None

        entities = await self._entities([qid for qid, _method in related])
        for qid, method in related:
            entity = entities.get(qid)
            if not isinstance(entity, Mapping):
                continue
            coordinate = self._coordinate(entity)
            if coordinate is None:
                continue
            return WikidataTeamLocation(
                *coordinate,
                club_qid=club_qid,
                location_qid=qid,
                location_name=self._label(entity),
                method=method,
                confidence=0.90 if method == "home_venue" else 0.75,
            )
        return None

    async def _entities(self, qids: list[str]) -> dict[str, Any]:
        payload = await self._request(
            {
                "action": "wbgetentities",
                "ids": "|".join(qids),
                "props": "claims|labels",
                "languages": "en",
            }
        )
        entities = payload.get("entities")
        return dict(entities) if isinstance(entities, Mapping) else {}

    async def _request(self, params: dict[str, str]) -> dict[str, Any]:
        response: httpx.Response | None = None
        for attempt in range(5):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds,
                    follow_redirects=False,
                    transport=self.transport,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "BetAIPlatform/1.0 (Wikidata location resolver)",
                    },
                ) as client:
                    response = await client.get(
                        self.api_url,
                        params={**params, "format": "json"},
                    )
                if response.status_code not in {429, 502, 503, 504}:
                    response.raise_for_status()
                    break
                if attempt == 4:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 2**attempt
                except ValueError:
                    delay = 2**attempt
                await asyncio.sleep(min(30.0, max(1.0, delay)))
            except httpx.HTTPError as exc:
                if attempt == 4:
                    raise WikidataError("Wikidata request failed") from exc
                await asyncio.sleep(min(8.0, float(2**attempt)))
        if response is None:
            raise WikidataError("Wikidata request failed")
        if self.request_interval_seconds:
            await asyncio.sleep(self.request_interval_seconds)
        if len(response.content) > MAX_WIKIDATA_PAYLOAD_BYTES:
            raise WikidataError("Wikidata response exceeds the safety limit")
        try:
            payload = response.json()
        except ValueError as exc:
            raise WikidataError("Wikidata response is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise WikidataError("Wikidata returned an invalid response")
        return payload

    @staticmethod
    def _select_candidate(
        rows: object,
        *,
        team_name: str,
        country: str | None,
    ) -> dict[str, str] | None:
        if not isinstance(rows, list):
            return None
        target = normalize_team_name(team_name)
        country_key = stable_team_name_key(country or "")
        scored: list[tuple[int, int, dict[str, str]]] = []
        for rank, raw in enumerate(rows):
            if not isinstance(raw, Mapping):
                continue
            qid = raw.get("id")
            label = raw.get("label")
            description = str(raw.get("description") or "").lower()
            if (
                not isinstance(qid, str)
                or not qid.startswith("Q")
                or not isinstance(label, str)
            ):
                continue
            if not any(marker in description for marker in _SPORT_DESCRIPTIONS):
                continue
            if any(marker in description for marker in _EXCLUDED_DESCRIPTIONS):
                continue
            score = 4
            label_key = normalize_team_name(label)
            if label_key == target:
                score += 6
            elif target in label_key or label_key in target:
                score += 3
            if "association football" in description:
                score += 2
            if country_key and country_key in stable_team_name_key(description):
                score += 2
            scored.append((score, -rank, {"id": qid, "label": label}))
        if not scored:
            return None
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
            return None
        return scored[0][2] if scored[0][0] >= 6 else None

    @staticmethod
    def _entity_ids(entity: Mapping[str, Any], property_id: str) -> list[str]:
        claims = entity.get("claims")
        values = claims.get(property_id, []) if isinstance(claims, Mapping) else []
        if not isinstance(values, list):
            return []
        ordered = sorted(
            (value for value in values if isinstance(value, Mapping)),
            key=lambda value: {"preferred": 0, "normal": 1, "deprecated": 2}.get(
                str(value.get("rank")), 1
            ),
        )
        result: list[str] = []
        for claim in ordered:
            mainsnak = claim.get("mainsnak")
            datavalue = (
                mainsnak.get("datavalue") if isinstance(mainsnak, Mapping) else None
            )
            value = datavalue.get("value") if isinstance(datavalue, Mapping) else None
            qid = value.get("id") if isinstance(value, Mapping) else None
            if isinstance(qid, str) and qid.startswith("Q"):
                result.append(qid)
        return result

    @staticmethod
    def _coordinate(entity: Mapping[str, Any]) -> tuple[float, float] | None:
        claims = entity.get("claims")
        values = claims.get("P625", []) if isinstance(claims, Mapping) else []
        if not isinstance(values, list):
            return None
        for claim in values:
            mainsnak = claim.get("mainsnak") if isinstance(claim, Mapping) else None
            datavalue = (
                mainsnak.get("datavalue") if isinstance(mainsnak, Mapping) else None
            )
            value = datavalue.get("value") if isinstance(datavalue, Mapping) else None
            if not isinstance(value, Mapping):
                continue
            try:
                latitude = float(value["latitude"])
                longitude = float(value["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                math.isfinite(latitude)
                and math.isfinite(longitude)
                and -90 <= latitude <= 90
                and -180 <= longitude <= 180
            ):
                return latitude, longitude
        return None

    @staticmethod
    def _label(entity: Mapping[str, Any]) -> str | None:
        labels = entity.get("labels")
        english = labels.get("en") if isinstance(labels, Mapping) else None
        value = english.get("value") if isinstance(english, Mapping) else None
        return str(value)[:150] if isinstance(value, str) and value.strip() else None
