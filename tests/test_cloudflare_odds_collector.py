import hashlib
import hmac

from app.core.config import settings
from app.services.cloudflare_odds_collector import CloudflareOddsCollectorClient


def test_client_secret_is_domain_separated_and_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(settings, "API_FOOTBALL_KEY", "test-api-key")

    first = CloudflareOddsCollectorClient._client_secret()
    second = CloudflareOddsCollectorClient._client_secret()

    assert first == second
    assert len(first) == 64
    assert first != b"test-api-key"


def test_signed_headers_bind_a_single_use_nonce(monkeypatch) -> None:
    monkeypatch.setattr(settings, "API_FOOTBALL_KEY", "test-api-key")
    body = '{"fixture_id":42}'
    headers = CloudflareOddsCollectorClient._signed_headers(
        "POST",
        "/v1/tracked-fixtures",
        body,
        timestamp="1787256000",
        nonce="nonce_abcdefghijklmnopqrstuvwxyz",
    )
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    message = (
        "1787256000.nonce_abcdefghijklmnopqrstuvwxyz.POST."
        f"/v1/tracked-fixtures.{body_hash}"
    ).encode()
    expected = hmac.new(
        CloudflareOddsCollectorClient._client_secret(),
        message,
        hashlib.sha256,
    ).hexdigest()

    assert headers["x-betai-nonce"] == "nonce_abcdefghijklmnopqrstuvwxyz"
    assert headers["x-betai-signature"] == expected


def test_valid_entry_accepts_complete_market() -> None:
    entry = CloudflareOddsCollectorClient.valid_entry(
        {
            "recorded": True,
            "raw_odds": {"HOME_WIN": 2.1, "DRAW": 3.2, "AWAY_WIN": 3.6},
            "captured_at": "2026-08-13T18:00:00.000Z",
            "bookmaker": "Bet365",
            "payload_sha256": "a" * 64,
        }
    )

    assert entry is not None
    assert entry["bookmaker"] == "Bet365"
    assert entry["raw_odds"] == {
        "HOME_WIN": 2.1,
        "DRAW": 3.2,
        "AWAY_WIN": 3.6,
    }


def test_valid_entry_rejects_incomplete_or_invalid_market() -> None:
    assert CloudflareOddsCollectorClient.valid_entry({"recorded": False}) is None
    assert (
        CloudflareOddsCollectorClient.valid_entry(
            {
                "recorded": True,
                "raw_odds": {"HOME_WIN": 1.0, "DRAW": 3.2, "AWAY_WIN": 3.6},
                "captured_at": "2026-08-13T18:00:00.000Z",
                "bookmaker": "Bet365",
            }
        )
        is None
    )
