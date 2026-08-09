from __future__ import annotations

import hashlib

MAX_PROVIDER_ID = 249_999_999
"""Largest raw provider id accepted in any namespaced range.

All namespaced ids (= OFFSET + raw id, with raw id in 1..MAX) stay inside
PostgreSQL's signed 32-bit integer range when OFFSET <= 2_000_000_000.
"""

OPENLIGADB_ID_OFFSET = 500_000_000
SOURCE_ID_OFFSETS = {
    "thesportsdb": 1_000_000_000,
    "sportmonks": 1_250_000_000,
    "football_data_org": 1_500_000_000,
    "fixture_download": 1_750_000_000,
}

# Natural-key (hash) providers keep disjoint positive ranges so the historical
# space and the prediction/odds space share one canonical identity per match.
FOOTBALL_DATA_FIXTURE_OFFSET = 2_000_000_000
FOOTBALL_DATA_TEAM_OFFSET = 2_050_000_000
OPENFOOTBALL_FIXTURE_OFFSET = 2_100_000_000
OPENFOOTBALL_TEAM_OFFSET = 2_150_000_000
FIXTURE_DOWNLOAD_TEAM_OFFSET = 2_200_000_000

# Canonical id for the fixture_download source: nothing is re-hashed, the
# completed-fixture feed and the aggregator share this single namespaced range.
FIXTURE_DOWNLOAD_FIXTURE_OFFSET = SOURCE_ID_OFFSETS["fixture_download"]

STATSBOMB_FIXTURE_OFFSET = 8_000_000_000
STATSBOMB_TEAM_OFFSET = 8_500_000_000
STATSBOMB_PLAYER_OFFSET = 9_000_000_000


def namespaced_id(source: str, raw_id: object) -> int | None:
    """Map a raw provider id into a source-specific positive integer range."""
    try:
        provider_id = int(str(raw_id))
    except (TypeError, ValueError):
        return None
    if provider_id <= 0 or provider_id > MAX_PROVIDER_ID:
        return None
    return SOURCE_ID_OFFSETS[source] + provider_id


def hashed_id(namespace: str, natural_key: str, offset: int) -> int:
    """Deterministically derive a positive namespaced id from a natural key.

    The digest is folded into 1..MAX_PROVIDER_ID so the id stays inside the
    provider range bound; the same natural key always yields the same id, in
    every layer that ingests the source.
    """
    digest = hashlib.blake2b(
        f"{namespace}:{natural_key}".encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, byteorder="big") % MAX_PROVIDER_ID
    return offset + (value or 1)
