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


def test_closing_odd_uses_predicted_outcome_market() -> None:
    market = {"raw_odds": {"HOME_WIN": 1.8, "DRAW": 3.4, "AWAY_WIN": 4.6}}

    assert PredictionAuditor.select_closing_odd(market, "AWAY_WIN") == 4.6
    assert PredictionAuditor.select_closing_odd(market, "DRAW") == 3.4
    assert PredictionAuditor.select_closing_odd(market, None) is None


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
    assert result["total_bets"] == 0
    assert result["total_roi_pct"] == 0.0
    assert result["brier_score"] is None
    assert result["brier_samples"] == 0


def test_audit_normalizes_probabilities_and_reports_sample_counts() -> None:
    prediction = MatchPrediction(
        actual_result="HOME_WIN",
        prediction="HOME_WIN",
        odd=2.5,
        closing_odds=2.0,
        prob_home=60.0,
        prob_draw=25.0,
        prob_away=15.0,
    )

    result = PredictionAuditor.audit_predictions([prediction])

    assert result["correct_predictions"] == 1
    assert result["total_bets"] == 1
    assert result["total_profit_units"] == 1.5
    assert result["total_roi_pct"] == 150.0
    assert result["brier_score"] == 0.245
    assert result["brier_samples"] == 1
    assert result["avg_clv_pct"] == 25.0
    assert result["clv_samples"] == 1
    assert result["sample_status"] == "insufficient"
    assert result["decision_grade"] is False
    assert result["win_rate_confidence_interval_95"] == {
        "lower_pct": 20.65,
        "upper_pct": 100.0,
    }
    assert result["roi_confidence_interval_95_pct"] == {
        "lower_pct": 150.0,
        "upper_pct": 150.0,
    }


def test_audit_groups_resolved_predictions_by_league() -> None:
    predictions = [
        MatchPrediction(
            league_id=39,
            actual_result="HOME_WIN",
            prediction="HOME_WIN",
            odd=2.0,
            prob_home=60,
            prob_draw=25,
            prob_away=15,
        ),
        MatchPrediction(
            league_id=203,
            actual_result="DRAW",
            prediction="AWAY_WIN",
            odd=3.0,
            prob_home=30,
            prob_draw=30,
            prob_away=40,
        ),
        MatchPrediction(actual_result="DRAW", prediction="DRAW", odd=3.0),
    ]

    result = PredictionAuditor.audit_by_league(predictions)

    assert result["overall"]["total_predictions"] == 3
    assert result["unassigned_predictions"] == 1
    assert [row["league_id"] for row in result["leagues"]] == [39, 203]
    assert result["leagues"][0]["league_name"] == "Premier League"


def test_wilson_interval_rejects_invalid_samples() -> None:
    assert PredictionAuditor.wilson_interval(0, 0) is None
    assert PredictionAuditor.wilson_interval(2, 1) is None


def test_reliable_sample_is_explicit(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "AUDIT_MIN_RELIABLE_SAMPLES", 2)
    predictions = [
        MatchPrediction(
            actual_result="HOME_WIN",
            prediction="HOME_WIN",
            odd=2.0,
            prob_home=60,
            prob_draw=25,
            prob_away=15,
        ),
        MatchPrediction(
            actual_result="DRAW",
            prediction="AWAY_WIN",
            odd=3.0,
            prob_home=30,
            prob_draw=30,
            prob_away=40,
        ),
    ]

    result = PredictionAuditor.audit_predictions(predictions)

    assert result["sample_status"] == "reliable"
    assert result["decision_grade"] is True
    assert result["minimum_reliable_samples"] == 2
