from datetime import UTC, datetime

from app.db.models import HistoricalFixture, MatchPrediction
from app.providers.openligadb import ID_OFFSET as OPENLIGADB_ID_OFFSET
from app.services.result_verification import (
    ResultVerificationService,
    provider_request_fixture_id,
)


def _prediction(**overrides: object) -> MatchPrediction:
    values: dict[str, object] = {
        "fixture_id": 123,
        "fixture_source": "api_football",
        "provider_fixture_id": "123",
        "league_id": 39,
        "home_team_id": 10,
        "away_team_id": 20,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "training_eligible": True,
    }
    values.update(overrides)
    return MatchPrediction(**values)


def _fixture(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "fixture_id": 123,
        "status": "FT",
        "score": "2 - 1",
        "league_id": 39,
        "home_team_id": 10,
        "away_team_id": 20,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
    }
    values.update(overrides)
    return values


def test_verified_provider_result_produces_training_label() -> None:
    decision = ResultVerificationService.verify(_prediction(), _fixture())

    assert decision.status == "verified"
    assert decision.result is not None
    assert decision.result.actual_result == "HOME_WIN"
    assert decision.result.home_score == 2
    assert decision.result.away_score == 1
    assert decision.result.source == "api_football"


def test_provider_identity_mismatch_is_quarantined() -> None:
    decision = ResultVerificationService.verify(
        _prediction(), _fixture(home_team_id=999)
    )

    assert decision.status == "conflict"
    assert decision.reason == "home_team_id_mismatch"
    assert decision.result is None


def test_non_final_fixture_stays_pending() -> None:
    decision = ResultVerificationService.verify(_prediction(), _fixture(status="2H"))

    assert decision.status == "pending"
    assert decision.reason == "fixture_not_final"


def test_invalid_score_is_rejected() -> None:
    decision = ResultVerificationService.verify(
        _prediction(), _fixture(score="cancelled")
    )

    assert decision.status == "rejected"
    assert decision.reason == "invalid_final_score"


def test_source_is_not_inferred_from_global_fixture_id() -> None:
    prediction = _prediction(
        fixture_id=OPENLIGADB_ID_OFFSET + 44,
        fixture_source=None,
        provider_fixture_id="44",
    )

    assert provider_request_fixture_id(prediction) is None
    decision = ResultVerificationService.verify(prediction, None)
    assert decision.status == "rejected"
    assert decision.reason == "unsupported_fixture_source"


def test_openligadb_provider_id_is_namespaced_only_for_request() -> None:
    prediction = _prediction(
        fixture_id=OPENLIGADB_ID_OFFSET + 44,
        fixture_source="openligadb",
        provider_fixture_id="44",
    )

    assert provider_request_fixture_id(prediction) == OPENLIGADB_ID_OFFSET + 44


def test_exact_historical_fixture_can_verify_non_api_source() -> None:
    prediction = _prediction(
        fixture_id=800_000_123,
        fixture_source="football_data_org",
        provider_fixture_id="E0-2026-123",
    )
    historical = HistoricalFixture(
        fixture_id=800_000_123,
        league_id=39,
        season=2026,
        kickoff=datetime(2026, 8, 1, tzinfo=UTC),
        home_team_id=10,
        away_team_id=20,
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=1,
        away_goals=1,
        actual_result="DRAW",
        status="FT",
        data_source="football_data_csv",
    )

    decision = ResultVerificationService.verify_historical(prediction, historical)

    assert decision.status == "verified"
    assert decision.result is not None
    assert decision.result.actual_result == "DRAW"
    assert decision.result.source == "historical:football_data_csv"


def test_historical_score_label_conflict_is_quarantined() -> None:
    historical = HistoricalFixture(
        fixture_id=123,
        league_id=39,
        season=2026,
        kickoff=datetime(2026, 8, 1, tzinfo=UTC),
        home_team_id=10,
        away_team_id=20,
        home_team="Arsenal",
        away_team="Chelsea",
        home_goals=3,
        away_goals=0,
        actual_result="AWAY_WIN",
        status="FT",
        data_source="api_football",
    )

    decision = ResultVerificationService.verify_historical(_prediction(), historical)

    assert decision.status == "conflict"
    assert decision.reason == "historical_result_score_mismatch"
