from datetime import UTC, datetime, timedelta

from app.db.models import MatchPrediction
from app.prediction.audit import FinancialRecommendationPolicy, PredictionAuditor


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
    kickoff = datetime(2026, 8, 9, 18, tzinfo=UTC)
    prediction = MatchPrediction(
        actual_result="HOME_WIN",
        prediction="HOME_WIN",
        odd=2.5,
        closing_odds=2.0,
        kickoff=kickoff,
        closing_odds_snapshot_at=kickoff - timedelta(hours=2),
        closing_odds_snapshot_id=11,
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
    assert result["closing_bets"] == 1
    assert result["closing_roi_pct"] == 100.0
    assert result["closing_roi_confidence_interval_95_pct"] == 100.0
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


def test_financial_recommendations_require_positive_roi_and_clv(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "AUDIT_MIN_RELIABLE_SAMPLES", 30)
    decision = FinancialRecommendationPolicy.evaluate(
        {
            "decision_grade": True,
            "total_bets": 40,
            "roi_confidence_interval_95_pct": {
                "lower_pct": 1.2,
                "upper_pct": 8.4,
            },
            "clv_samples": 35,
            "avg_clv_pct": 2.1,
            "closing_bets": 220,
            "closing_roi_confidence_interval_95_pct": 2.5,
        }
    )

    assert decision["eligible"] is True
    assert decision["reasons"] == []
    assert decision["closing_gate"]["closing_bets"] == 220
    assert decision["closing_gate"]["closing_roi_lower_95_pct"] == 2.5


def test_financial_recommendations_fail_closed_on_weak_evidence(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "AUDIT_MIN_RELIABLE_SAMPLES", 30)
    decision = FinancialRecommendationPolicy.evaluate(
        {
            "decision_grade": True,
            "total_bets": 40,
            "roi_confidence_interval_95_pct": {
                "lower_pct": -3.5,
                "upper_pct": 9.0,
            },
            "clv_samples": 12,
            "avg_clv_pct": -0.4,
            "closing_bets": 150,
            "closing_roi_confidence_interval_95_pct": -5.0,
        }
    )

    assert decision["eligible"] is False
    assert decision["reasons"] == [
        "roi_confidence_interval_not_positive",
        "insufficient_closing_odds_samples",
        "average_clv_not_positive",
        "insufficient_closing_odds_bets",
        "closing_roi_below_threshold",
    ]


def test_closing_roi_never_pushed_as_zero_without_evidence() -> None:
    prediction = MatchPrediction(
        actual_result="HOME_WIN",
        prediction="HOME_WIN",
        odd=2.5,
    )

    result = PredictionAuditor.audit_predictions([prediction])

    assert result["closing_bets"] == 0
    assert result["closing_roi_pct"] is None
    assert result["closing_vs_opening_roi_delta_pct"] is None
    assert result["closing_roi_confidence_interval_95_pct"] is None


def test_financial_recommendations_require_closing_roi_samples_and_gain() -> None:
    subject = {
        "decision_grade": True,
        "total_bets": 40,
        "roi_confidence_interval_95_pct": {
            "lower_pct": 1.2,
            "upper_pct": 8.4,
        },
        "clv_samples": 35,
        "avg_clv_pct": 2.1,
    }

    too_few = FinancialRecommendationPolicy.evaluate(
        {**subject, "closing_bets": 199, "closing_roi_confidence_interval_95_pct": 2.5}
    )
    assert too_few["reasons"] == ["insufficient_closing_odds_bets"]

    no_gain = FinancialRecommendationPolicy.evaluate(
        {**subject, "closing_bets": 250, "closing_roi_confidence_interval_95_pct": 1.9}
    )
    assert no_gain["reasons"] == ["closing_roi_below_threshold"]


def test_stale_closing_odds_are_excluded_from_closing_evidence() -> None:
    kickoff = datetime(2026, 8, 9, 18, tzinfo=UTC)
    stale = MatchPrediction(
        actual_result="HOME_WIN",
        prediction="HOME_WIN",
        odd=2.5,
        closing_odds=2.0,
        kickoff=kickoff,
        closing_odds_snapshot_at=kickoff - timedelta(hours=30),
        closing_odds_snapshot_id=7,
    )

    result = PredictionAuditor.audit_predictions([stale])

    assert result["closing_bets"] == 0
    assert result["closing_roi_pct"] is None
    assert result["closing_vs_opening_roi_delta_pct"] is None
    assert result["stale_closing_odds_bets"] == 1
    assert result["closing_provenance_missing_bets"] == 0


def test_fresh_closing_odds_count_as_evidence_with_provenance() -> None:
    kickoff = datetime(2026, 8, 9, 18, tzinfo=UTC)
    fresh = MatchPrediction(
        actual_result="HOME_WIN",
        prediction="HOME_WIN",
        odd=2.5,
        closing_odds=2.0,
        kickoff=kickoff,
        closing_odds_snapshot_at=kickoff - timedelta(hours=2),
        closing_odds_snapshot_id=9,
    )

    result = PredictionAuditor.audit_predictions([fresh])

    assert result["closing_bets"] == 1
    assert result["closing_roi_pct"] == 100.0
    assert result["stale_closing_odds_bets"] == 0
    assert result["closing_provenance_missing_bets"] == 0


def test_legacy_closing_odds_without_provenance_are_not_closing_evidence() -> None:
    legacy = MatchPrediction(
        actual_result="HOME_WIN",
        prediction="HOME_WIN",
        odd=2.5,
        closing_odds=2.0,
    )

    result = PredictionAuditor.audit_predictions([legacy])

    # Unverifiable legacy prices are missing evidence, never evidence: they
    # must not inflate closing_bets/closing_roi_pct with a 0.0 or a fake value.
    assert result["closing_bets"] == 0
    assert result["closing_roi_pct"] is None
    assert result["closing_vs_opening_roi_delta_pct"] is None
    assert result["stale_closing_odds_bets"] == 0
    assert result["closing_provenance_missing_bets"] == 1


def test_audit_reports_opening_and_closing_roi_delta() -> None:
    kickoff = datetime(2026, 8, 9, 18, tzinfo=UTC)
    predictions = [
        MatchPrediction(
            actual_result="HOME_WIN",
            prediction="HOME_WIN",
            odd=3.0,
            closing_odds=2.2,
            kickoff=kickoff,
            closing_odds_snapshot_at=kickoff - timedelta(hours=2),
            closing_odds_snapshot_id=13,
        ),
        MatchPrediction(
            actual_result="AWAY_WIN",
            prediction="HOME_WIN",
            odd=3.0,
            closing_odds=2.2,
            kickoff=kickoff,
            closing_odds_snapshot_at=kickoff - timedelta(hours=1),
            closing_odds_snapshot_id=14,
        ),
    ]

    result = PredictionAuditor.audit_predictions(predictions)

    assert result["opening_roi_pct"] == 50.0
    assert result["closing_roi_pct"] == 10.0
    assert result["closing_vs_opening_roi_delta_pct"] == -40.0


def test_closing_roi_erosion_blocks_financial_activation() -> None:
    subject = {
        "decision_grade": True,
        "total_bets": 300,
        "roi_confidence_interval_95_pct": {"lower_pct": 4.0, "upper_pct": 12.0},
        "clv_samples": 250,
        "avg_clv_pct": 3.0,
        "closing_bets": 220,
        "closing_roi_confidence_interval_95_pct": 2.4,
        "opening_roi_pct": 14.0,
        "closing_roi_pct": 6.0,
    }

    decision = FinancialRecommendationPolicy.evaluate(subject)

    assert decision["eligible"] is False
    assert "closing_roi_erosion" in decision["reasons"]


def test_matching_closing_roi_is_not_erosion() -> None:
    subject = {
        "decision_grade": True,
        "total_bets": 300,
        "roi_confidence_interval_95_pct": {"lower_pct": 4.0, "upper_pct": 12.0},
        "clv_samples": 250,
        "avg_clv_pct": 3.0,
        "closing_bets": 220,
        "closing_roi_confidence_interval_95_pct": 2.4,
        "opening_roi_pct": 7.0,
        "closing_roi_pct": 6.0,
    }

    decision = FinancialRecommendationPolicy.evaluate(subject)

    assert "closing_roi_erosion" not in decision["reasons"]
