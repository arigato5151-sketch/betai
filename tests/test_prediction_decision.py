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
    assert "top_probability_too_low" in decision["reasons"]
    assert "probability_margin_too_low" in decision["reasons"]
    assert "predictive_entropy_too_high" in decision["reasons"]


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
