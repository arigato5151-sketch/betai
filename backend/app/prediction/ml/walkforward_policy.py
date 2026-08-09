"""Walk-forward evaluation and promotion gate policy.

CI runs this gate as the "walk-forward model evaluation" step.  It is a
pure function of the recorded metrics so it can be unit-tested cheaply and
reused by scripts/calibrate_constants.py style tooling.
"""

from __future__ import annotations

from typing import Mapping

DEFAULT_MAX_WALK_FORWARD_BRIER = 0.40
DEFAULT_MAX_WALK_FORWARD_LOG_LOSS = 1.10

_REQUIRED_STRATEGY = "walk_forward_temporal_holdout"


def walk_forward_gate(
    metrics: Mapping[str, object],
    *,
    max_brier: float = DEFAULT_MAX_WALK_FORWARD_BRIER,
    max_log_loss: float = DEFAULT_MAX_WALK_FORWARD_LOG_LOSS,
) -> tuple[bool, list[str]]:
    """Evaluate the promotion gate over recorded walk-forward metrics.

    Returns ``(passed, failures)`` where ``failures`` lists human readable
    reasons the candidate must not be promoted.
    """
    failures: list[str] = []
    if metrics.get("evaluation_strategy") != _REQUIRED_STRATEGY:
        failures.append(
            f"evaluation_strategy must be {_REQUIRED_STRATEGY!r}, "
            f"got {metrics.get('evaluation_strategy')!r}"
        )

    brier = metrics.get("walk_forward_brier_score")
    log_loss = metrics.get("walk_forward_log_loss")
    if not isinstance(brier, (int, float)) or not isinstance(log_loss, (int, float)):
        failures.append(
            "walk_forward_brier_score and walk_forward_log_loss must be recorded"
        )
    else:
        if brier > max_brier:
            failures.append(
                f"walk_forward_brier_score={brier:.4f} exceeds budget {max_brier:.4f}"
            )
        if log_loss > max_log_loss:
            failures.append(
                f"walk_forward_log_loss={log_loss:.4f} exceeds budget {max_log_loss:.4f}"
            )

    if metrics.get("beats_opening_market") is not True and (
        "beats_opening_market" in metrics
    ):
        failures.append("candidate does not beat the opening market benchmark")
    if metrics.get("promotion_failures"):
        failures.append(f"tiered promotion rejected: {metrics['promotion_failures']}")
    return (not failures, failures)
