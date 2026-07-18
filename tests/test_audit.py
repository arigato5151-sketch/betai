from app.db.models import MatchPrediction
from app.prediction.audit import PredictionAuditor


def test_roi_returns_zero_for_incomplete_or_invalid_bet() -> None:
    assert PredictionAuditor.calculate_bet_roi(None, "HOME_WIN", 2.0) == 0.0
    assert PredictionAuditor.calculate_bet_roi("HOME_WIN", None, 2.0) == 0.0
    assert PredictionAuditor.calculate_bet_roi("HOME_WIN", "HOME_WIN", None) == 0.0
    assert PredictionAuditor.calculate_bet_roi("HOME_WIN", "HOME_WIN", 1.0) == 0.0


def test_roi_and_clv_known_values() -> None:
    assert PredictionAuditor.calculate_bet_roi("HOME_WIN", "HOME_WIN", 2.5) == 1.5
    assert PredictionAuditor.calculate_bet_roi("HOME_WIN", "DRAW", 2.5) == -1.0
    assert PredictionAuditor.calculate_clv(2.2, 2.0) == 0.1
    assert PredictionAuditor.calculate_clv(2.2, None) == 0.0


def test_audit_handles_legacy_resolved_row_with_nullable_bet_fields() -> None:
    prediction = MatchPrediction(
        actual_result="HOME_WIN",
        prediction=None,
        odd=None,
        prob_home=None,
        prob_draw=None,
        prob_away=None,
    )

    result = PredictionAuditor.audit_predictions([prediction])

    assert result["total_predictions"] == 1
    assert result["total_roi_pct"] == 0.0
