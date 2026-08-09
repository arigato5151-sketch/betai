from __future__ import annotations

from app.prediction.ml.walkforward_policy import (
    DEFAULT_MAX_WALK_FORWARD_BRIER,
    DEFAULT_MAX_WALK_FORWARD_LOG_LOSS,
    walk_forward_gate,
)


def test_gate_passes_for_compliant_metrics() -> None:
    passed, failures = walk_forward_gate(
        {
            "evaluation_strategy": "walk_forward_temporal_holdout",
            "walk_forward_brier_score": 0.31,
            "walk_forward_log_loss": 0.98,
        }
    )
    assert passed is True
    assert failures == []


def test_gate_rejects_wrong_evaluation_strategy() -> None:
    passed, failures = walk_forward_gate(
        {
            "evaluation_strategy": "random_split",
            "walk_forward_brier_score": 0.31,
            "walk_forward_log_loss": 0.98,
        }
    )
    assert passed is False
    assert any("evaluation_strategy" in failure for failure in failures)


def test_gate_rejects_budget_overruns() -> None:
    passed, failures = walk_forward_gate(
        {
            "evaluation_strategy": "walk_forward_temporal_holdout",
            "walk_forward_brier_score": DEFAULT_MAX_WALK_FORWARD_BRIER + 0.2,
            "walk_forward_log_loss": DEFAULT_MAX_WALK_FORWARD_LOG_LOSS + 0.3,
        }
    )
    assert passed is False
    assert any("walk_forward_brier_score" in failure for failure in failures)
    assert any("walk_forward_log_loss" in failure for failure in failures)


def test_gate_requires_walk_forward_metrics() -> None:
    passed, failures = walk_forward_gate(
        {
            "evaluation_strategy": "walk_forward_temporal_holdout",
            "brier_score": 0.1,
        }
    )
    assert passed is False
    assert any("must be recorded" in failure for failure in failures)


def test_gate_rejects_tiered_promotion_failure() -> None:
    passed, failures = walk_forward_gate(
        {
            "evaluation_strategy": "walk_forward_temporal_holdout",
            "walk_forward_brier_score": 0.30,
            "walk_forward_log_loss": 0.97,
            "beats_opening_market": False,
            "promotion_failures": ["opening_market"],
        }
    )
    assert passed is False
    assert any("opening market" in failure for failure in failures)
    assert any("promotion rejected" in failure for failure in failures)


def test_gate_ignores_benchmarks_when_absent_for_single_model() -> None:
    passed, failures = walk_forward_gate(
        {
            "evaluation_strategy": "walk_forward_temporal_holdout",
            "walk_forward_brier_score": 0.32,
            "walk_forward_log_loss": 1.0,
        }
    )
    assert passed is True
    assert failures == []
