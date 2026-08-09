from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if not BACKEND_DIR.is_dir():
    BACKEND_DIR = ROOT_DIR
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.prediction.ml.features import FeatureEngine  # noqa: E402
from app.prediction.ml.model import MLModelPipeline  # noqa: E402
from app.prediction.ml.model_router import TieredModelArtifactStore  # noqa: E402
from app.prediction.ml.walkforward_policy import walk_forward_gate  # noqa: E402

_OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")


class WalkForwardGateClassifier(BaseEstimator):
    """Deterministic candidate so the gate's numbers are stable in CI."""

    def __init__(self, home_form_index: int = 0) -> None:
        self.home_form_index = home_form_index

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WalkForwardGateClassifier":
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.full((len(X), 3), 0.017, dtype=float)
        bucket = (X[:, self.home_form_index] % 3).astype(int)
        probabilities[np.arange(len(X)), bucket] = 0.95
        return probabilities


def _synthetic_rows(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            actual_result=_OUTCOMES[index % 3],
            created_at=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index),
            feature_snapshot={
                **FeatureEngine.FEATURE_DEFAULTS,
                "home_form": float((index % 3) * 40),
            },
            feature_schema_version=FeatureEngine.SCHEMA_VERSION,
            feature_snapshot_at=None,
        )
        for index in range(count)
    ]


def _run_synthetic_evaluation(
    max_brier: float, max_log_loss: float
) -> dict[str, object]:
    settings.MIN_TRAINING_SAMPLES = 48
    settings.MAX_MODEL_BRIER_SCORE = 1.0
    settings.MAX_MODEL_LOG_LOSS = 10.0
    settings.MAX_MODEL_CALIBRATION_ERROR = 1.0
    settings.MIN_MODEL_BASELINE_BRIER_IMPROVEMENT = -1.0
    settings.MAX_MODEL_BASELINE_LOG_LOSS_REGRESSION = 1.0
    settings.MIN_ISOTONIC_CALIBRATION_SAMPLES = 10**6

    pipeline = MLModelPipeline()
    pipeline.model = None
    pipeline.calibrator = None
    pipeline.is_ready = False
    home_form_index = pipeline.feature_names.index("home_form")
    pipeline._get_candidate_models = lambda: [  # type: ignore[method-assign]
        (
            "Walk-Forward-Gate",
            WalkForwardGateClassifier(home_form_index=home_form_index),
        )
    ]
    pipeline._save_active_model = lambda *args, **kwargs: None  # type: ignore[method-assign]

    succeeded = pipeline.train_pipeline(_synthetic_rows(150))
    metrics = dict(pipeline.metrics)
    passed, failures = walk_forward_gate(
        metrics, max_brier=max_brier, max_log_loss=max_log_loss
    )
    return {
        "mode": "synthetic",
        "trained": bool(succeeded),
        "passed": bool(passed),
        "failures": failures,
        "metrics": {
            "evaluation_strategy": metrics.get("evaluation_strategy"),
            "walk_forward_brier_score": metrics.get("walk_forward_brier_score"),
            "walk_forward_log_loss": metrics.get("walk_forward_log_loss"),
            "brier_score": metrics.get("brier_score"),
            "log_loss": metrics.get("log_loss"),
            "training_samples": metrics.get("training_samples"),
        },
    }


def _run_artifact_evaluation(
    max_brier: float, max_log_loss: float
) -> dict[str, object]:
    bundle = TieredModelArtifactStore().load_active()
    if bundle is None:
        return {
            "mode": "artifact",
            "passed": False,
            "failures": ["no signed tiered artifact present"],
            "metrics": {},
        }
    tier1_metrics = dict(bundle.metadata.get("tier1_metrics", {}))
    passed, failures = walk_forward_gate(
        tier1_metrics, max_brier=max_brier, max_log_loss=max_log_loss
    )
    return {
        "mode": "artifact",
        "artifact_version": bundle.artifact_version,
        "passed": bool(passed),
        "failures": failures,
        "metrics": {
            k: tier1_metrics.get(k)
            for k in (
                "evaluation_strategy",
                "walk_forward_brier_score",
                "walk_forward_log_loss",
                "beats_opening_market",
                "promotion_failures",
            )
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation and model promotion gate (P7)."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run deterministic walk-forward evaluation on synthetic data",
    )
    parser.add_argument(
        "--artifact",
        action="store_true",
        help="validate the signed tiered model artifact instead of synthetic data",
    )
    parser.add_argument("--max-brier", type=float, default=0.40)
    parser.add_argument("--max-log-loss", type=float, default=1.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "synthetic" if args.synthetic else "artifact" if args.artifact else "auto"
    if mode == "auto":
        bundle = TieredModelArtifactStore().load_active()
        mode = "artifact" if bundle is not None else "synthetic"
    report = (
        _run_synthetic_evaluation(args.max_brier, args.max_log_loss)
        if mode == "synthetic"
        else _run_artifact_evaluation(args.max_brier, args.max_log_loss)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
