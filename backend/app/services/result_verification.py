from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping

from app.core.team_identity import normalize_team_name
from app.db.models import HistoricalFixture, MatchPrediction
from app.providers.openligadb import ID_OFFSET as OPENLIGADB_ID_OFFSET

_FINAL_STATUSES = frozenset({"FT", "AET", "PEN"})
_SCORE_PATTERN = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
_SOURCE_ALIASES = {
    "api-football": "api_football",
    "api_football": "api_football",
    "openligadb": "openligadb",
}


@dataclass(frozen=True)
class VerifiedResult:
    actual_result: str
    home_score: int
    away_score: int
    source: str
    provider_fixture_id: str
    verified_at: datetime


@dataclass(frozen=True)
class ResultVerificationDecision:
    status: str
    reason: str | None = None
    result: VerifiedResult | None = None


def canonical_result_source(value: object) -> str | None:
    return _SOURCE_ALIASES.get(str(value or "").strip().lower())


def provider_request_fixture_id(prediction: MatchPrediction) -> int | None:
    """Return the provider-specific request ID without guessing from global ranges."""
    source = canonical_result_source(prediction.fixture_source)
    raw_id = str(prediction.provider_fixture_id or "").strip()
    if source is None or not raw_id.isdecimal():
        return None
    provider_id = int(raw_id)
    if provider_id <= 0:
        return None
    return OPENLIGADB_ID_OFFSET + provider_id if source == "openligadb" else provider_id


class ResultVerificationService:
    """Validate provider responses before they become labels or audit outcomes."""

    @staticmethod
    def verify(
        prediction: MatchPrediction,
        fixture: Mapping[str, object] | None,
    ) -> ResultVerificationDecision:
        source = canonical_result_source(prediction.fixture_source)
        request_id = provider_request_fixture_id(prediction)
        if source is None:
            return ResultVerificationDecision("rejected", "unsupported_fixture_source")
        if request_id is None:
            return ResultVerificationDecision("rejected", "invalid_provider_fixture_id")
        if not fixture:
            return ResultVerificationDecision("pending", "fixture_not_available")
        if fixture.get("status") not in _FINAL_STATUSES:
            return ResultVerificationDecision("pending", "fixture_not_final")

        try:
            response_fixture_id = int(str(fixture.get("fixture_id")))
        except (TypeError, ValueError):
            return ResultVerificationDecision("rejected", "missing_response_fixture_id")
        if response_fixture_id != request_id:
            return ResultVerificationDecision(
                "conflict", "provider_fixture_id_mismatch"
            )

        identity_error = ResultVerificationService._identity_error(prediction, fixture)
        if identity_error:
            return ResultVerificationDecision("conflict", identity_error)

        score_match = _SCORE_PATTERN.fullmatch(str(fixture.get("score") or ""))
        if score_match is None:
            return ResultVerificationDecision("rejected", "invalid_final_score")
        home_score, away_score = (int(value) for value in score_match.groups())
        actual_result = (
            "HOME_WIN"
            if home_score > away_score
            else "AWAY_WIN" if home_score < away_score else "DRAW"
        )
        return ResultVerificationDecision(
            "verified",
            result=VerifiedResult(
                actual_result=actual_result,
                home_score=home_score,
                away_score=away_score,
                source=source,
                provider_fixture_id=str(prediction.provider_fixture_id),
                verified_at=datetime.now(UTC),
            ),
        )

    @staticmethod
    def verify_historical(
        prediction: MatchPrediction,
        fixture: HistoricalFixture | None,
    ) -> ResultVerificationDecision:
        """Verify an exact namespaced fixture already ingested into local history."""
        if fixture is None:
            return ResultVerificationDecision("pending", "historical_fixture_not_found")
        if prediction.fixture_id is None or fixture.fixture_id != prediction.fixture_id:
            return ResultVerificationDecision(
                "conflict", "historical_fixture_id_mismatch"
            )
        if fixture.status not in _FINAL_STATUSES:
            return ResultVerificationDecision("pending", "historical_fixture_not_final")

        identity = {
            "league_id": fixture.league_id,
            "home_team_id": fixture.home_team_id,
            "away_team_id": fixture.away_team_id,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
        }
        identity_error = ResultVerificationService._identity_error(prediction, identity)
        if identity_error:
            return ResultVerificationDecision("conflict", identity_error)

        actual_result = (
            "HOME_WIN"
            if fixture.home_goals > fixture.away_goals
            else "AWAY_WIN" if fixture.home_goals < fixture.away_goals else "DRAW"
        )
        if fixture.actual_result != actual_result:
            return ResultVerificationDecision(
                "conflict", "historical_result_score_mismatch"
            )
        return ResultVerificationDecision(
            "verified",
            result=VerifiedResult(
                actual_result=actual_result,
                home_score=fixture.home_goals,
                away_score=fixture.away_goals,
                source=f"historical:{fixture.data_source}"[:50],
                provider_fixture_id=str(
                    prediction.provider_fixture_id or fixture.fixture_id
                ),
                verified_at=datetime.now(UTC),
            ),
        )

    @staticmethod
    def _identity_error(
        prediction: MatchPrediction,
        fixture: Mapping[str, object],
    ) -> str | None:
        for field in ("league_id", "home_team_id", "away_team_id"):
            expected = getattr(prediction, field)
            received = fixture.get(field)
            if expected is not None and received not in (None, 0, "0"):
                try:
                    if int(str(received)) != expected:
                        return f"{field}_mismatch"
                except (TypeError, ValueError):
                    return f"invalid_{field}"

        for field in ("home_team", "away_team"):
            expected_name = normalize_team_name(str(getattr(prediction, field) or ""))
            received_name = normalize_team_name(str(fixture.get(field) or ""))
            if expected_name and received_name and expected_name != received_name:
                return f"{field}_mismatch"
        return None
