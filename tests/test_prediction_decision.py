from app.prediction.decision import PredictionDecisionPolicy


def test_clear_probability_distribution_is_decision_eligible() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 70.0,
                "DRAW": 20.0,
                "AWAY_WIN": 10.0,
            }
        }
    )

    assert decision["status"] == "eligible"
    assert decision["confidence_tier"] == "high"
    assert decision["probability_margin_pct"] == 50.0


def test_near_uniform_distribution_abstains_without_rejecting_forecast() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 34.0,
                "DRAW": 33.0,
                "AWAY_WIN": 33.0,
            }
        }
    )

    assert decision["status"] == "abstain"
    assert "probability_margin_too_low" in decision["reasons"]
    assert decision["confidence_tier"] == "low"


def test_weak_consensus_abstains_when_sources_materially_diverge() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 42.0,
                "DRAW": 34.0,
                "AWAY_WIN": 24.0,
            },
            "ensemble": {
                "components": {
                    "stats": {
                        "HOME_WIN": 80.0,
                        "DRAW": 10.0,
                        "AWAY_WIN": 10.0,
                    },
                    "ml": {
                        "HOME_WIN": 10.0,
                        "DRAW": 20.0,
                        "AWAY_WIN": 70.0,
                    },
                }
            },
        }
    )

    assert decision["status"] == "abstain"
    assert "prediction_sources_diverge" in decision["reasons"]
    assert decision["max_source_js_divergence"] > 0.15


def test_invalid_probability_payload_fails_closed() -> None:
    assert PredictionDecisionPolicy.evaluate({}) == {
        "status": "abstain",
        "reasons": ["invalid_probabilities"],
        "confidence_tier": "insufficient",
    }


def test_decision_requiring_market_stays_research_without_price() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 70.0,
                "DRAW": 20.0,
                "AWAY_WIN": 10.0,
            }
        },
        require_market=True,
    )

    assert decision["status"] == "research"
    assert "market_unavailable" in decision["reasons"]
    assert decision["market_validation"]["present"] is False


def test_decision_requiring_market_demotes_clear_forecast_without_edge() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 70.0,
                "DRAW": 20.0,
                "AWAY_WIN": 10.0,
            }
        },
        market_edge_pct=2.0,
        market_implied_pct=68.0,
        require_market=True,
    )

    assert decision["status"] == "research"
    assert "market_edge_insufficient" in decision["reasons"]
    assert decision["market_validation"]["passed"] is False


def test_decision_requiring_market_clears_positive_edge() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 70.0,
                "DRAW": 20.0,
                "AWAY_WIN": 10.0,
            }
        },
        market_edge_pct=6.5,
        market_implied_pct=60.0,
        require_market=True,
    )

    assert decision["status"] == "eligible"
    assert decision["market_validation"]["passed"] is True
    assert decision["market_validation"]["edge_pct"] == 6.5


def test_conditional_status_when_marginal_probabilities() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 35.0,
                "DRAW": 32.5,
                "AWAY_WIN": 32.5,
            }
        }
    )

    assert decision["status"] == "conditional"
    assert decision["confidence_tier"] == "conditional"
    assert decision["reasons"] == []
    assert decision["top_probability_pct"] == 35.0


def test_abstain_still_triggers_for_truly_uncertain() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 33.4,
                "DRAW": 33.3,
                "AWAY_WIN": 33.3,
            }
        }
    )

    assert decision["status"] == "abstain"
    assert "probability_margin_too_low" in decision["reasons"]
    assert decision["confidence_tier"] == "low"


def test_contextual_thresholds_market_confirmed_relaxes() -> None:
    decision_without = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 27.0,
                "DRAW": 36.0,
                "AWAY_WIN": 37.0,
            }
        }
    )
    decision_with = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 27.0,
                "DRAW": 36.0,
                "AWAY_WIN": 37.0,
            }
        },
        market_confirmed=True,
        data_quality_score=75.0,
    )

    assert decision_without["status"] == "abstain"
    assert decision_with["status"] == "conditional"
    assert decision_with["confidence_tier"] == "conditional"


def test_contextual_thresholds_high_data_quality_only() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 28.0,
                "DRAW": 35.3,
                "AWAY_WIN": 36.7,
            }
        },
        data_quality_score=80.0,
    )

    assert decision["status"] == "conditional"
    assert decision["reasons"] == []
    assert decision["confidence_tier"] == "conditional"


def test_backward_compatible_evaluate_without_new_params() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 70.0,
                "DRAW": 20.0,
                "AWAY_WIN": 10.0,
            }
        }
    )

    assert decision["status"] == "eligible"
    assert decision["confidence_tier"] == "high"
    assert "market_validation" in decision


def test_eligible_when_both_prob_and_margin_clear() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 45.0,
                "DRAW": 30.0,
                "AWAY_WIN": 25.0,
            }
        }
    )

    assert decision["status"] == "eligible"
    assert decision["confidence_tier"] == "medium"


def test_abstain_when_prob_ok_but_margin_too_low() -> None:
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "HOME_WIN": 36.0,
                "DRAW": 34.6,
                "AWAY_WIN": 29.4,
            }
        }
    )

    assert decision["status"] == "abstain"
    assert "probability_margin_too_low" in decision["reasons"]
