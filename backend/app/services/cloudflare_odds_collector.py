from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import time
from datetime import datetime
from typing import Any, Mapping

import httpx

from app.core.config import settings


class CloudflareOddsCollectorError(RuntimeError):
    """Raised when the remote collector cannot safely satisfy a request."""


class CloudflareOddsCollectorClient:
    def __init__(self) -> None:
        self.base_url = settings.CLOUDFLARE_ODDS_COLLECTOR_URL.rstrip("/")
        self.timeout = settings.CLOUDFLARE_ODDS_COLLECTOR_TIMEOUT_SECONDS

    @property
    def enabled(self) -> bool:
        return bool(
            settings.CLOUDFLARE_ODDS_COLLECTOR_ENABLED
            and self.base_url
            and settings.API_FOOTBALL_KEY
            and settings.API_FOOTBALL_KEY != "DEMO_KEY"
        )

    async def track_fixture(
        self,
        *,
        fixture_id: int,
        fixture_source: str | None,
        provider_fixture_id: str | None,
        league_id: int,
        home_team: str,
        away_team: str,
        kickoff: datetime,
        preferred_bookmaker: str | None,
        capture_entry: bool,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        api_fixture_id = self._api_fixture_id(
            fixture_id, fixture_source, provider_fixture_id
        )
        payload: dict[str, object] = {
            "id": f"fixture:{fixture_id}",
            "league_id": league_id,
            "home_team": home_team,
            "away_team": away_team,
            "kickoff": kickoff.isoformat(),
            "capture_entry": capture_entry,
        }
        if api_fixture_id is not None:
            payload["api_fixture_id"] = api_fixture_id
        if preferred_bookmaker:
            payload["preferred_bookmaker"] = preferred_bookmaker
        return await self._request("POST", "/v1/tracked-fixtures", payload=payload)

    async def snapshots(self, *, after: int = 0, limit: int = 250) -> list[dict]:
        if not self.enabled:
            return []
        result = await self._request(
            "GET", f"/v1/snapshots?after={max(0, after)}&limit={min(250, limit)}"
        )
        rows = result.get("snapshots", []) if isinstance(result, Mapping) else []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, Any]:
        body = (
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            if payload is not None
            else ""
        )
        headers = self._signed_headers(method, path, body)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    content=body.encode("utf-8") if body else None,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise CloudflareOddsCollectorError("collector_transport_error") from exc
        if response.status_code >= 400:
            raise CloudflareOddsCollectorError(f"collector_http_{response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            raise CloudflareOddsCollectorError("collector_invalid_json") from exc
        if not isinstance(result, dict):
            raise CloudflareOddsCollectorError("collector_invalid_payload")
        return result

    @classmethod
    def _signed_headers(
        cls,
        method: str,
        path: str,
        body: str,
        *,
        timestamp: str | None = None,
        nonce: str | None = None,
    ) -> dict[str, str]:
        issued_at = timestamp or str(int(time.time()))
        request_nonce = nonce or secrets.token_urlsafe(24)
        body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        message = (
            f"{issued_at}.{request_nonce}.{method.upper()}.{path}.{body_hash}"
        ).encode("utf-8")
        signature = hmac.new(cls._client_secret(), message, hashlib.sha256).hexdigest()
        return {
            "x-betai-timestamp": issued_at,
            "x-betai-nonce": request_nonce,
            "x-betai-signature": signature,
            "content-type": "application/json",
        }

    @staticmethod
    def valid_entry(value: object) -> dict[str, object] | None:
        if not isinstance(value, Mapping) or value.get("recorded") is not True:
            return None
        raw = value.get("raw_odds")
        if not isinstance(raw, Mapping):
            return None
        odds: dict[str, float] = {}
        for outcome in ("HOME_WIN", "DRAW", "AWAY_WIN"):
            try:
                odd = float(raw[outcome])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(odd) or not 1.0 < odd <= 1000.0:
                return None
            odds[outcome] = odd
        captured_at = value.get("captured_at")
        bookmaker = value.get("bookmaker")
        if not isinstance(captured_at, str) or not isinstance(bookmaker, str):
            return None
        return {
            "raw_odds": odds,
            "captured_at": captured_at,
            "bookmaker": bookmaker,
            "payload_sha256": value.get("payload_sha256"),
            "api_fixture_id": value.get("api_fixture_id"),
        }

    @staticmethod
    def _api_fixture_id(
        fixture_id: int,
        fixture_source: str | None,
        provider_fixture_id: str | None,
    ) -> int | None:
        if fixture_source != "api_football":
            return None
        for value in (provider_fixture_id, fixture_id):
            try:
                parsed = int(str(value))
            except (TypeError, ValueError):
                continue
            if 0 < parsed < 100_000_000:
                return parsed
        return None

    @staticmethod
    def _client_secret() -> bytes:
        material = ("bet-ai-cloudflare-hmac-v1:" + settings.API_FOOTBALL_KEY).encode(
            "utf-8"
        )
        return hashlib.sha256(material).hexdigest().encode("ascii")


cloudflare_odds_collector = CloudflareOddsCollectorClient()
