from app.core.config import settings
from app.prediction.eligibility import PredictionEligibilityPolicy


def quality_payload(**checks: bool) -> dict[str, object]:
    return {
        "score": 90.0,
        "checks": {
            "fixture_identified": True,
            "fixture_source_identified": True,
            "provider_fixture_identified": True,
            "league_identified": True,
            "kickoff_known": True,
            "market_available": True,
            "home_history_sufficient": True,
            "away_history_sufficient": True,
            **checks,
        },
        "manual_feature_override_count": 0,
    }


def test_complete_automatic_prediction_is_eligible() -> None:
    decision = PredictionEligibilityPolicy.evaluate(quality_payload())

    assert decision.eligible is True
    assert decision.status == "eligible"
    assert decision.reasons == ()


def test_missing_market_and_history_force_abstention() -> None:
    decision = PredictionEligibilityPolicy.evaluate(
        quality_payload(
            market_available=False,
            home_history_sufficient=False,
        )
    )

    assert decision.eligible is False
    assert decision.status == "abstain"
    assert "market_unavailable" in decision.reasons
    assert "home_history_insufficient" in decision.reasons


def test_missing_provider_identity_force_abstention() -> None:
    decision = PredictionEligibilityPolicy.evaluate(
        quality_payload(provider_fixture_identified=False)
    )

    assert decision.eligible is False
    assert "missing_provider_fixture_identified" in decision.reasons


def test_low_quality_and_manual_override_force_abstention(monkeypatch) -> None:
    monkeypatch.setattr(settings, "AUTO_PREDICTION_MIN_DATA_QUALITY_SCORE", 75.0)
    payload = quality_payload()
    payload["score"] = 50.0
    payload["manual_feature_override_count"] = 1

    decision = PredictionEligibilityPolicy.evaluate(payload)

    assert decision.eligible is False
    assert "data_quality_below_threshold" in decision.reasons
    assert "manual_override_not_automatic" in decision.reasons
