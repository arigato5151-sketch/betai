from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.config import settings
from app.prediction.ml.calibrate import MultiClassCalibrator
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.model import MLModelPipeline


class ProbabilityModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.probabilities for _ in range(len(features))], dtype=float)


def training_row(result: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        actual_result=result,
        home_form=60,
        home_attack=65,
        home_defense=62,
        home_xg=1.5,
        away_form=55,
        away_attack=58,
        away_defense=57,
        away_xg=1.2,
    )


def test_not_ready_pipeline_returns_safe_fallback() -> None:
    pipeline = MLModelPipeline()

    assert pipeline.feature_names == FeatureEngine.FEATURE_NAMES
    assert pipeline.predict_match({}) == {"ready": False, "probabilities": None}


def test_inference_normalizes_three_class_probabilities() -> None:
    pipeline = MLModelPipeline()
    pipeline.model = ProbabilityModel([1.0, 2.0, 1.0])
    pipeline.is_ready = True
    pipeline.active_model_name = "Test Model"

    result = pipeline.predict_match({name: 1.0 for name in pipeline.feature_names})

    assert result["ready"] is True
    assert result["prediction"] == "DRAW"
    assert result["all_probabilities"] == {
        "HOME_WIN": 25.0,
        "DRAW": 50.0,
        "AWAY_WIN": 25.0,
    }
    assert sum(result["all_probabilities"].values()) == 100.0
    assert pipeline.runtime_stats == {"inference_success": 1, "inference_failure": 0}


@pytest.mark.parametrize(
    "probabilities",
    [[0.5, 0.5], [0.5, float("nan"), 0.5], [0.5, -0.1, 0.6], [0.0, 0.0, 0.0]],
)
def test_inference_rejects_invalid_model_output(probabilities: list[float]) -> None:
    pipeline = MLModelPipeline()
    pipeline.model = ProbabilityModel(probabilities)
    pipeline.is_ready = True

    assert pipeline.predict_match({}) == {"ready": False, "probabilities": None}


def test_inference_rejects_non_finite_features() -> None:
    pipeline = MLModelPipeline()
    pipeline.model = ProbabilityModel([0.3, 0.4, 0.3])
    pipeline.is_ready = True

    assert pipeline.predict_match({"home_form": float("nan")}) == {
        "ready": False,
        "probabilities": None,
    }
    assert pipeline.runtime_stats["inference_failure"] == 1


def test_training_rejects_insufficient_or_unbalanced_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MLModelPipeline()
    monkeypatch.setattr(settings, "MIN_TRAINING_SAMPLES", 6)

    assert pipeline.train_pipeline([training_row("HOME_WIN")] * 5) is False
    assert pipeline.train_pipeline([training_row("HOME_WIN")] * 6) is False
    assert (
        pipeline.train_pipeline(
            [
                training_row("HOME_WIN"),
                training_row("HOME_WIN"),
                training_row("DRAW"),
                training_row("DRAW"),
                training_row("AWAY_WIN"),
                training_row("INVALID"),
            ]
        )
        is False
    )


def test_load_model_clears_stale_state_when_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MLModelPipeline()
    pipeline.model = ProbabilityModel([0.3, 0.4, 0.3])
    pipeline.calibrator = object()
    pipeline.is_ready = True
    monkeypatch.setattr("app.prediction.ml.model.os.path.exists", lambda _: False)

    assert pipeline.load_active_model() is False
    assert pipeline.model is None
    assert pipeline.calibrator is None
    assert pipeline.is_ready is False


def test_load_valid_model_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = MLModelPipeline()
    model = ProbabilityModel([0.2, 0.6, 0.2])
    monkeypatch.setattr("app.prediction.ml.model.os.path.exists", lambda _: True)
    monkeypatch.setattr(
        "app.prediction.ml.model.joblib.load",
        lambda _: {
            "model": model,
            "calibrator": None,
            "feature_names": ["home_form"],
            "model_name": "Artifact Model",
            "metrics": {"brier_score": 0.12},
        },
    )

    assert pipeline.load_active_model() is True
    assert pipeline.model is model
    assert pipeline.active_model_name == "Artifact Model"
    assert pipeline.metrics["brier_score"] == 0.12


def test_calibrator_falls_back_to_base_probabilities_for_zero_rows() -> None:
    base = ProbabilityModel([0.2, 0.3, 0.5])
    calibrator = MultiClassCalibrator(base)
    calibrator.calibrators = [
        Mock(predict=Mock(return_value=np.array([0.0]))) for _ in range(3)
    ]

    probabilities = calibrator.predict_proba(np.array([[1.0]]))[0]

    assert probabilities == pytest.approx([0.2, 0.3, 0.5])
    assert probabilities.sum() == pytest.approx(1.0)


def test_versioned_artifacts_support_validated_rollback(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    artifacts_dir = tmp_path / "models"
    active_path = artifacts_dir / "active_model.pkl"
    monkeypatch.setattr(settings, "MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(active_path))
    pipeline = MLModelPipeline()

    pipeline._save_active_model(
        ProbabilityModel([0.7, 0.2, 0.1]), None, "First", {"brier_score": 0.2}
    )
    first_version = pipeline.artifact_version
    pipeline._save_active_model(
        ProbabilityModel([0.1, 0.2, 0.7]), None, "Second", {"brier_score": 0.1}
    )

    assert pipeline.artifact_version != first_version
    assert pipeline.status()["rollback_available"] is True
    assert len(list((artifacts_dir / "versions").glob("model_*.pkl"))) == 2
    assert pipeline.rollback() is True
    assert pipeline.active_model_name == "First"
    assert pipeline.artifact_version == first_version


def test_rollback_requires_active_and_previous_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(tmp_path / "active.pkl"))

    assert MLModelPipeline().rollback() is False
