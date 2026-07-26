from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

from app.core.config import settings
from app.prediction.ml.calibrate import MultiClassCalibrator
from app.prediction.ml.categorical import NativeCategoricalBoostingClassifier
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.model import (
    CATBOOST_AVAILABLE,
    LGBM_AVAILABLE,
    MLModelPipeline,
)


class ProbabilityModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.array([self.probabilities for _ in range(len(features))], dtype=float)


class RecordingClassifier(BaseEstimator):
    fit_ranges: list[tuple[float, float]] = []

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.__class__.fit_ranges.append(
            (float(features[:, 0].min()), float(features[:, 0].max()))
        )
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.4, 0.3, 0.3]), (len(features), 1))


class RecordingCalibrator:
    fit_range: tuple[float, float] | None = None
    predict_range: tuple[float, float] | None = None

    def __init__(self, base_clf: RecordingClassifier) -> None:
        self.base_clf = base_clf

    def fit(self, features: np.ndarray, labels: np.ndarray):
        self.__class__.fit_range = (
            float(features[:, 0].min()),
            float(features[:, 0].max()),
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.__class__.predict_range = (
            float(features[:, 0].min()),
            float(features[:, 0].max()),
        )
        return self.base_clf.predict_proba(features)


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


def temporal_training_row(index: int) -> SimpleNamespace:
    outcomes = ("HOME_WIN", "DRAW", "AWAY_WIN")
    return SimpleNamespace(
        actual_result=outcomes[index % len(outcomes)],
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index),
        feature_snapshot={
            **FeatureEngine.FEATURE_DEFAULTS,
            "home_form": float(index),
        },
        feature_schema_version=FeatureEngine.SCHEMA_VERSION,
        feature_snapshot_at=None,
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


def test_candidate_pool_registers_enabled_native_boosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.prediction.ml import model

    monkeypatch.setattr(model, "CATBOOST_AVAILABLE", True)
    monkeypatch.setattr(model, "LGBM_AVAILABLE", True)
    monkeypatch.setattr(settings, "ENABLE_CATBOOST_CANDIDATE", True)
    monkeypatch.setattr(settings, "ENABLE_LIGHTGBM_CANDIDATE", True)

    candidates = dict(MLModelPipeline()._get_candidate_models())

    assert isinstance(candidates["CatBoost"], NativeCategoricalBoostingClassifier)
    assert candidates["CatBoost"].backend == "catboost"
    assert isinstance(candidates["LightGBM"], NativeCategoricalBoostingClassifier)
    assert candidates["LightGBM"].backend == "lightgbm"
    assert candidates["CatBoost"].categorical_feature_names == (
        "league_id",
        "home_team_id",
        "away_team_id",
    )


def test_candidate_pool_skips_unavailable_native_boosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.prediction.ml import model

    monkeypatch.setattr(model, "CATBOOST_AVAILABLE", False)
    monkeypatch.setattr(model, "LGBM_AVAILABLE", False)

    candidates = dict(MLModelPipeline()._get_candidate_models())

    assert "CatBoost" not in candidates
    assert "LightGBM" not in candidates
    assert "Regularized Logistic Regression" in candidates
    assert "Random Forest" in candidates


@pytest.mark.parametrize(
    "backend,available",
    [
        pytest.param("catboost", CATBOOST_AVAILABLE, id="catboost"),
        pytest.param("lightgbm", LGBM_AVAILABLE, id="lightgbm"),
    ],
)
def test_native_categorical_booster_clone_fit_and_predict(
    backend: str, available: bool
) -> None:
    if not available:
        pytest.skip(f"{backend} is not installed")
    feature_names = ("numeric", "league_id", "home_team_id", "away_team_id")
    estimator = NativeCategoricalBoostingClassifier(
        backend=backend,
        feature_names=feature_names,
        categorical_feature_names=feature_names[1:],
        n_estimators=8,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=1,
    )
    labels = np.asarray([0, 1, 2] * 8)
    features = np.asarray(
        [
            [
                float(index % 5),
                float(203 if index % 2 else 39),
                float(100 + index % 4),
                float(200 + index % 5),
            ]
            for index in range(len(labels))
        ]
    )

    fitted = clone(estimator).fit(features, labels)
    probabilities = fitted.predict_proba(features[:3])

    assert probabilities.shape == (3, 3)
    assert np.all(np.isfinite(probabilities))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(3))
    assert fitted.classes_.tolist() == [0, 1, 2]


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


def test_training_uses_disjoint_temporal_calibration_and_test_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.prediction.ml import calibrate

    RecordingClassifier.fit_ranges = []
    RecordingCalibrator.fit_range = None
    RecordingCalibrator.predict_range = None
    pipeline = MLModelPipeline()
    monkeypatch.setattr(settings, "MIN_TRAINING_SAMPLES", 30)
    monkeypatch.setattr(settings, "MIN_MODEL_BASELINE_BRIER_IMPROVEMENT", -1.0)
    monkeypatch.setattr(settings, "MAX_MODEL_BASELINE_LOG_LOSS_REGRESSION", 1.0)
    monkeypatch.setattr(
        pipeline,
        "_get_candidate_models",
        lambda: [("Recording", RecordingClassifier())],
    )
    monkeypatch.setattr(calibrate, "MultiClassCalibrator", RecordingCalibrator)
    monkeypatch.setattr(pipeline, "_save_active_model", Mock())
    rows = [temporal_training_row(index) for index in reversed(range(60))]

    assert pipeline.train_pipeline(rows) is True

    # Final model sees only the oldest training window.
    assert RecordingClassifier.fit_ranges[-1] == (0.0, 38.0)
    # Calibration and test windows are later and mutually disjoint.
    assert RecordingCalibrator.fit_range == (39.0, 42.0)
    assert RecordingCalibrator.predict_range == (43.0, 47.0)
    assert pipeline.metrics["calibration_applied"] is False
    assert pipeline.metrics["evaluation_strategy"] == ("walk_forward_temporal_holdout")
    assert pipeline.metrics["training_samples"] == 39.0
    assert pipeline.metrics["calibration_samples"] == 9.0
    assert pipeline.metrics["test_samples"] == 12.0


def test_training_rejects_candidate_that_does_not_beat_naive_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.prediction.ml import calibrate

    pipeline = MLModelPipeline()
    monkeypatch.setattr(settings, "MIN_TRAINING_SAMPLES", 30)
    monkeypatch.setattr(
        pipeline,
        "_get_candidate_models",
        lambda: [("Recording", RecordingClassifier())],
    )
    monkeypatch.setattr(calibrate, "MultiClassCalibrator", RecordingCalibrator)
    save = Mock()
    monkeypatch.setattr(pipeline, "_save_active_model", save)
    rows = [temporal_training_row(index) for index in range(60)]

    assert pipeline.train_pipeline(rows) is False
    save.assert_not_called()
    assert pipeline.is_ready is False


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


def test_load_valid_signed_model_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    artifacts_dir = tmp_path / "models"
    active_path = artifacts_dir / "active_model.pkl"
    monkeypatch.setattr(settings, "MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(active_path))
    model = ProbabilityModel([0.2, 0.6, 0.2])
    writer = MLModelPipeline()
    writer._save_active_model(
        model,
        None,
        "Artifact Model",
        {"brier_score": 0.12},
    )
    pipeline = MLModelPipeline()

    assert pipeline.load_active_model() is True
    assert pipeline.active_model_name == "Artifact Model"
    assert pipeline.metrics["brier_score"] == 0.12
    assert active_path.with_name("active_model.pkl.sig").is_file()


@pytest.mark.parametrize(
    "backend,available",
    [
        pytest.param("catboost", CATBOOST_AVAILABLE, id="catboost"),
        pytest.param("lightgbm", LGBM_AVAILABLE, id="lightgbm"),
    ],
)
def test_signed_native_booster_artifact_roundtrip(
    backend: str,
    available: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    if not available:
        pytest.skip(f"{backend} is not installed")
    artifacts_dir = tmp_path / backend
    active_path = artifacts_dir / "active_model.pkl"
    monkeypatch.setattr(settings, "MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(active_path))
    feature_names = ("numeric", "league_id", "home_team_id", "away_team_id")
    labels = np.asarray([0, 1, 2] * 8)
    features = np.asarray(
        [
            [
                float(index % 5),
                float(203 if index % 2 else 39),
                float(100 + index % 4),
                float(200 + index % 5),
            ]
            for index in range(len(labels))
        ]
    )
    model = NativeCategoricalBoostingClassifier(
        backend=backend,
        feature_names=feature_names,
        categorical_feature_names=feature_names[1:],
        n_estimators=8,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=1,
    ).fit(features, labels)
    writer = MLModelPipeline()
    writer.feature_names = list(feature_names)
    writer._save_active_model(model, None, backend, {"brier_score": 0.2})

    loaded = MLModelPipeline()

    assert loaded.load_active_model() is True
    result = loaded.predict_match(
        {
            "numeric": 2.0,
            "league_id": 203.0,
            "home_team_id": 101.0,
            "away_team_id": 202.0,
        }
    )
    assert result["ready"] is True
    assert sum(result["all_probabilities"].values()) == pytest.approx(100.0, abs=0.02)
    assert active_path.with_name("active_model.pkl.sig").is_file()


def test_load_rejects_tampered_model_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog
) -> None:
    artifacts_dir = tmp_path / "models"
    active_path = artifacts_dir / "active_model.pkl"
    monkeypatch.setattr(settings, "MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(active_path))
    writer = MLModelPipeline()
    writer._save_active_model(
        ProbabilityModel([0.2, 0.6, 0.2]),
        None,
        "Artifact Model",
        {"brier_score": 0.12},
    )
    with active_path.open("ab") as artifact:
        artifact.write(b"tampered")

    pipeline = MLModelPipeline()

    assert pipeline.load_active_model() is False
    assert pipeline.is_ready is False
    assert pipeline.model is None
    assert "signature verification failed" in caplog.text


def test_failed_artifact_reload_preserves_in_memory_champion(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    artifacts_dir = tmp_path / "models"
    active_path = artifacts_dir / "active_model.pkl"
    monkeypatch.setattr(settings, "MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(active_path))
    writer = MLModelPipeline()
    writer._save_active_model(
        ProbabilityModel([0.2, 0.6, 0.2]),
        None,
        "Artifact Model",
        {"brier_score": 0.12},
    )
    with active_path.open("ab") as artifact:
        artifact.write(b"tampered")

    champion = ProbabilityModel([0.7, 0.2, 0.1])
    pipeline = MLModelPipeline()
    pipeline.model = champion
    pipeline.is_ready = True
    pipeline.active_model_name = "In-memory champion"

    assert pipeline.load_active_model() is False
    assert pipeline.model is champion
    assert pipeline.is_ready is True
    assert pipeline.active_model_name == "In-memory champion"


def test_calibrator_falls_back_to_base_probabilities_for_zero_rows() -> None:
    base = ProbabilityModel([0.2, 0.3, 0.5])
    calibrator = MultiClassCalibrator(base)
    calibrator.calibrators = [
        Mock(predict=Mock(return_value=np.array([0.0]))) for _ in range(3)
    ]

    probabilities = calibrator.predict_proba(np.array([[1.0]]))[0]

    assert probabilities == pytest.approx([0.2, 0.3, 0.5])
    assert probabilities.sum() == pytest.approx(1.0)


def test_multiclass_calibration_error_uses_prediction_confidence() -> None:
    labels = np.array([0, 1, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )

    error = MLModelPipeline._multiclass_calibration_error(labels, probabilities)

    assert error == pytest.approx(0.2)


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


def test_rollback_rejects_tampered_previous_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog
) -> None:
    artifacts_dir = tmp_path / "models"
    active_path = artifacts_dir / "active_model.pkl"
    previous_path = artifacts_dir / "previous_model.pkl"
    monkeypatch.setattr(settings, "MODEL_ARTIFACTS_DIR", str(artifacts_dir))
    monkeypatch.setattr(settings, "ACTIVE_MODEL_PATH", str(active_path))
    pipeline = MLModelPipeline()
    pipeline._save_active_model(
        ProbabilityModel([0.7, 0.2, 0.1]), None, "First", {"brier_score": 0.2}
    )
    pipeline._save_active_model(
        ProbabilityModel([0.1, 0.2, 0.7]), None, "Second", {"brier_score": 0.1}
    )
    with previous_path.open("ab") as artifact:
        artifact.write(b"tampered")

    assert pipeline.rollback() is False
    assert pipeline.load_active_model() is True
    assert pipeline.active_model_name == "Second"
    assert "signature verification failed" in caplog.text
