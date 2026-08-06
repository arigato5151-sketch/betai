import hashlib
import hmac
import math
import os
import platform
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from typing import Any, Dict, List, Optional, Tuple, cast
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.config import settings
from app.core.logging_config import logger
from app.prediction.ml.categorical import NativeCategoricalBoostingClassifier
from app.prediction.ml.features import FeatureEngine

# Try imports with robust fallbacks
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except (ImportError, OSError):
    XGB_AVAILABLE = False

try:
    import lightgbm

    LGBM_AVAILABLE = True
    LIGHTGBM_VERSION: str | None = lightgbm.__version__
except (ImportError, OSError):
    LGBM_AVAILABLE = False
    LIGHTGBM_VERSION = None

try:
    import catboost

    CATBOOST_AVAILABLE = True
    CATBOOST_VERSION: str | None = catboost.__version__
except (ImportError, OSError):
    CATBOOST_AVAILABLE = False
    CATBOOST_VERSION = None


class MLModelPipeline:
    LABEL_MAP = {"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}
    INV_LABEL_MAP = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}

    def __init__(self):
        self.model: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.feature_names = list(FeatureEngine.FEATURE_NAMES)
        self.is_ready = False
        self.active_model_name: Optional[str] = None
        self.metrics: Dict[str, object] = {}
        self.artifact_version: Optional[str] = None
        self.runtime_stats = {"inference_success": 0, "inference_failure": 0}
        self._artifact_mtime_ns: int | None = None
        self._reload_lock = threading.Lock()

    @staticmethod
    def _previous_model_path() -> Path:
        return Path(settings.ACTIVE_MODEL_PATH).with_name("previous_model.pkl")

    @staticmethod
    def _signature_path(artifact_path: Path) -> Path:
        return artifact_path.with_name(f"{artifact_path.name}.sig")

    @classmethod
    def _artifact_signature(cls, artifact_path: Path) -> str:
        digest = hmac.new(
            settings.MODEL_SIGNING_KEY.encode("utf-8"),
            digestmod=hashlib.sha256,
        )
        with artifact_path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _write_signature(cls, artifact_path: Path, signature_path: Path) -> None:
        signature_path.write_text(
            cls._artifact_signature(artifact_path),
            encoding="ascii",
        )

    @classmethod
    def _verify_artifact(cls, artifact_path: Path) -> bool:
        signature_path = cls._signature_path(artifact_path)
        try:
            expected_signature = signature_path.read_text(encoding="ascii").strip()
            actual_signature = cls._artifact_signature(artifact_path)
        except OSError as exc:
            logger.error(
                "Model artifact signature could not be read for %s: %s",
                artifact_path,
                exc,
            )
            return False

        if not hmac.compare_digest(expected_signature, actual_signature):
            logger.error(
                "Model artifact signature verification failed for %s", artifact_path
            )
            return False
        return True

    def status(self) -> Dict[str, Any]:
        self._refresh_artifact_if_changed()
        previous_path = self._previous_model_path()
        return {
            "ready": self.is_ready,
            "model_name": self.active_model_name,
            "artifact_version": self.artifact_version,
            "metrics": dict(self.metrics),
            "runtime": dict(self.runtime_stats),
            "rollback_available": (
                previous_path.is_file()
                and self._signature_path(previous_path).is_file()
            ),
        }

    def _refresh_artifact_if_changed(self) -> None:
        active_path = Path(settings.ACTIVE_MODEL_PATH)
        try:
            current_mtime = active_path.stat().st_mtime_ns
        except OSError:
            return
        if current_mtime == self._artifact_mtime_ns:
            return
        with self._reload_lock:
            try:
                current_mtime = active_path.stat().st_mtime_ns
            except OSError:
                return
            if current_mtime != self._artifact_mtime_ns:
                self.load_active_model()

    def load_active_model(self) -> bool:
        """Loads model artifact from disk if it exists."""
        if not os.path.exists(settings.ACTIVE_MODEL_PATH):
            logger.info("No active ML model found on disk.")
            self.model = None
            self.calibrator = None
            self.is_ready = False
            self.active_model_name = None
            self.artifact_version = None
            self._artifact_mtime_ns = None
            return False

        previous_state = (
            self.model,
            self.calibrator,
            list(self.feature_names),
            self.active_model_name,
            dict(self.metrics),
            self.artifact_version,
            self.is_ready,
            self._artifact_mtime_ns,
        )
        try:
            active_path = Path(settings.ACTIVE_MODEL_PATH)
            if not self._verify_artifact(active_path):
                raise ValueError("Model artifact integrity verification failed")
            payload = joblib.load(settings.ACTIVE_MODEL_PATH)
            self._validate_loaded_payload(payload)
            self.model = payload["model"]
            self.calibrator = payload.get("calibrator")
            self.feature_names = payload.get("feature_names", self.feature_names)
            self.active_model_name = payload.get("model_name", "Unknown")
            self.metrics = payload.get("metrics", {})
            self.artifact_version = payload.get("artifact_version", "legacy")
            self.is_ready = True
            try:
                self._artifact_mtime_ns = (
                    Path(settings.ACTIVE_MODEL_PATH).stat().st_mtime_ns
                )
            except OSError:
                self._artifact_mtime_ns = None
            logger.info(
                f"Loaded active model: {self.active_model_name} with validation Brier Score: {self.metrics.get('brier_score', 'N/A')}"
            )
            return True
        except Exception as exc:
            logger.error("Failed to load ML model artifact from disk: %s", exc)
            if previous_state[6] and previous_state[0] is not None:
                (
                    self.model,
                    self.calibrator,
                    self.feature_names,
                    self.active_model_name,
                    self.metrics,
                    self.artifact_version,
                    self.is_ready,
                    self._artifact_mtime_ns,
                ) = previous_state
            else:
                self.model = None
                self.calibrator = None
                self.is_ready = False
                self.active_model_name = None
                self.artifact_version = None
                self._artifact_mtime_ns = None
            return False

    @classmethod
    def _validate_loaded_payload(cls, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("Model artifact payload must be a mapping")
        if payload.get("schema_version", 1) not in {1, 2}:
            raise ValueError("Unsupported model artifact schema")

        feature_names = payload.get("feature_names")
        if (
            not isinstance(feature_names, list)
            or not feature_names
            or any(not isinstance(name, str) or not name for name in feature_names)
            or len(feature_names) != len(set(feature_names))
        ):
            raise ValueError("Model artifact contains invalid feature names")

        feature_schema = payload.get("feature_schema_version")
        if (
            feature_schema is not None
            and feature_schema not in FeatureEngine.COMPATIBLE_SNAPSHOT_VERSIONS
        ):
            raise ValueError("Unsupported model feature schema")

        model = payload.get("model")
        predictor = payload.get("calibrator") or model
        predict_proba = getattr(predictor, "predict_proba", None)
        if model is None or predictor is None or not callable(predict_proba):
            raise ValueError("Model artifact has no probability predictor")
        classes = getattr(model, "classes_", None)
        if classes is not None and list(classes) != list(cls.INV_LABEL_MAP):
            raise ValueError("Model artifact class order is incompatible")

        smoke_features = np.asarray(
            [[FeatureEngine.FEATURE_DEFAULTS.get(name, 0.0) for name in feature_names]],
            dtype=float,
        )
        probabilities: np.ndarray = np.asarray(
            cast(Any, predict_proba)(smoke_features), dtype=np.float64
        )
        if (
            probabilities.shape != (1, len(cls.LABEL_MAP))
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or float(probabilities.sum()) <= 0
        ):
            raise ValueError("Model artifact failed probability smoke validation")

    @staticmethod
    def _runtime_dependency_versions() -> dict[str, str]:
        versions = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        }
        if CATBOOST_VERSION:
            versions["catboost"] = CATBOOST_VERSION
        if LIGHTGBM_VERSION:
            versions["lightgbm"] = LIGHTGBM_VERSION
        return versions

    def _save_active_model(
        self, model: Any, calibrator: Any, model_name: str, metrics: Dict[str, object]
    ) -> None:
        artifacts_dir = Path(settings.MODEL_ARTIFACTS_DIR)
        versions_dir = artifacts_dir / "versions"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        versions_dir.mkdir(parents=True, exist_ok=True)
        active_path = Path(settings.ACTIVE_MODEL_PATH)
        previous_path = self._previous_model_path()
        active_signature_path = self._signature_path(active_path)
        previous_signature_path = self._signature_path(previous_path)
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "schema_version": 2,
            "artifact_version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "model": model,
            "calibrator": calibrator,
            "feature_names": self.feature_names,
            "feature_schema_version": FeatureEngine.SCHEMA_VERSION,
            "model_name": model_name,
            "metrics": metrics,
            "runtime_dependencies": self._runtime_dependency_versions(),
        }
        temporary_path = active_path.with_suffix(".tmp")
        temporary_signature_path = self._signature_path(temporary_path)
        joblib.dump(payload, temporary_path)
        self._write_signature(temporary_path, temporary_signature_path)
        if active_path.is_file() and self._verify_artifact(active_path):
            shutil.copy2(active_path, previous_path)
            shutil.copy2(active_signature_path, previous_signature_path)
        elif active_path.is_file():
            logger.error(
                "Existing active model failed integrity verification; "
                "it will not be preserved as rollback candidate."
            )
        os.replace(temporary_path, active_path)
        os.replace(temporary_signature_path, active_signature_path)
        version_path = versions_dir / f"model_{version}.pkl"
        shutil.copy2(active_path, version_path)
        shutil.copy2(active_signature_path, self._signature_path(version_path))
        self.artifact_version = version
        self._artifact_mtime_ns = active_path.stat().st_mtime_ns
        logger.info(
            "Serialized model %s as artifact %s to %s",
            model_name,
            version,
            settings.ACTIVE_MODEL_PATH,
        )

    def rollback(self) -> bool:
        """Atomically swap the active and previous validated model artifacts."""
        active_path = Path(settings.ACTIVE_MODEL_PATH)
        previous_path = self._previous_model_path()
        active_signature_path = self._signature_path(active_path)
        previous_signature_path = self._signature_path(previous_path)
        required_paths = (
            active_path,
            active_signature_path,
            previous_path,
            previous_signature_path,
        )
        if not all(path.is_file() for path in required_paths):
            return False

        try:
            if not self._verify_artifact(active_path) or not self._verify_artifact(
                previous_path
            ):
                return False
            previous_payload = joblib.load(previous_path)
            if "model" not in previous_payload:
                raise ValueError("Previous artifact has no model")
            swap_path = active_path.with_suffix(".rollback.tmp")
            swap_signature_path = self._signature_path(swap_path)
            os.replace(active_path, swap_path)
            os.replace(active_signature_path, swap_signature_path)
            os.replace(previous_path, active_path)
            os.replace(previous_signature_path, active_signature_path)
            os.replace(swap_path, previous_path)
            os.replace(swap_signature_path, previous_signature_path)
            if self.load_active_model():
                logger.warning(
                    "Rolled back active ML model to %s", self.artifact_version
                )
                return True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("ML model rollback failed: %s", exc)
        return False

    def _numeric_candidate(self, estimator: Any, *, scale: bool = False) -> Pipeline:
        """Keep raw categorical IDs away from estimators that treat them as ordinal."""
        numeric_indices = [
            index
            for index, name in enumerate(self.feature_names)
            if name not in FeatureEngine.CATEGORICAL_FEATURE_NAMES
        ]
        transformer: Any = StandardScaler() if scale else "passthrough"
        preprocessor = ColumnTransformer(
            [("numeric", transformer, numeric_indices)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

    def _get_candidate_models(self) -> List[Tuple[str, Any]]:
        candidates = [
            (
                "Regularized Logistic Regression",
                self._numeric_candidate(
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                    scale=True,
                ),
            )
        ]

        if XGB_AVAILABLE:
            candidates.append(
                (
                    "XGBoost",
                    self._numeric_candidate(
                        XGBClassifier(
                            n_estimators=150,
                            max_depth=5,
                            learning_rate=0.08,
                            random_state=42,
                            eval_metric="mlogloss",
                        )
                    ),
                )
            )
        else:
            logger.warning(
                "XGBoost is not installed. Using Gradient Boosting fallback."
            )
            candidates.append(
                (
                    "Gradient Boosting",
                    self._numeric_candidate(
                        GradientBoostingClassifier(random_state=42)
                    ),
                )
            )

        if settings.ENABLE_LIGHTGBM_CANDIDATE and LGBM_AVAILABLE:
            candidates.append(
                (
                    "LightGBM",
                    NativeCategoricalBoostingClassifier(
                        backend="lightgbm",
                        feature_names=tuple(self.feature_names),
                        categorical_feature_names=(
                            FeatureEngine.CATEGORICAL_FEATURE_NAMES
                        ),
                        n_estimators=settings.ML_BOOSTER_TREES,
                        max_depth=settings.ML_BOOSTER_MAX_DEPTH,
                        learning_rate=settings.ML_BOOSTER_LEARNING_RATE,
                        random_state=42,
                        n_jobs=settings.ML_BOOSTER_THREADS,
                    ),
                )
            )
        elif settings.ENABLE_LIGHTGBM_CANDIDATE:
            logger.warning("LightGBM is unavailable; candidate skipped.")

        if settings.ENABLE_CATBOOST_CANDIDATE and CATBOOST_AVAILABLE:
            candidates.append(
                (
                    "CatBoost",
                    NativeCategoricalBoostingClassifier(
                        backend="catboost",
                        feature_names=tuple(self.feature_names),
                        categorical_feature_names=(
                            FeatureEngine.CATEGORICAL_FEATURE_NAMES
                        ),
                        n_estimators=settings.ML_BOOSTER_TREES,
                        max_depth=settings.ML_BOOSTER_MAX_DEPTH,
                        learning_rate=settings.ML_BOOSTER_LEARNING_RATE,
                        random_state=42,
                        n_jobs=settings.ML_BOOSTER_THREADS,
                    ),
                )
            )
        elif settings.ENABLE_CATBOOST_CANDIDATE:
            logger.warning("CatBoost is unavailable; candidate skipped.")

        # Always add Random Forest as a standard robust baseline
        candidates.append(
            (
                "Random Forest",
                self._numeric_candidate(
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                    )
                ),
            )
        )

        return candidates

    @staticmethod
    def _training_sort_key(row: Any) -> float:
        timestamp = getattr(row, "feature_snapshot_at", None) or getattr(
            row, "created_at", None
        )
        if not isinstance(timestamp, datetime):
            return 0.0
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.timestamp()

    @classmethod
    def _has_all_outcomes(cls, labels: np.ndarray) -> bool:
        return set(np.unique(labels)) == set(cls.INV_LABEL_MAP)

    @classmethod
    def _probability_metrics(
        cls, labels: np.ndarray, probabilities: np.ndarray
    ) -> tuple[float, float]:
        expected_shape = (len(labels), len(cls.LABEL_MAP))
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"Probability matrix must have shape {expected_shape}, got {probabilities.shape}"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Probability matrix contains invalid values")
        row_sums = probabilities.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Probability rows must have a positive sum")
        normalized = probabilities / row_sums
        one_hot = np.zeros_like(normalized)
        one_hot[np.arange(len(labels)), labels] = 1.0
        brier = float(np.mean(np.sum((normalized - one_hot) ** 2, axis=1)))
        loss = float(
            log_loss(labels, normalized, labels=list(range(len(cls.LABEL_MAP))))
        )
        return brier, loss

    @classmethod
    def _multiclass_calibration_error(
        cls,
        labels: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Confidence-based ECE for mutually exclusive multiclass predictions."""
        if not len(labels):
            return 0.0
        normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
        confidence = normalized.max(axis=1)
        predicted = normalized.argmax(axis=1)
        correct = (predicted == labels).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        error = 0.0
        for index in range(n_bins):
            lower, upper = edges[index], edges[index + 1]
            mask = (confidence >= lower) & (
                (confidence < upper) | ((index == n_bins - 1) & (confidence == upper))
            )
            if not np.any(mask):
                continue
            error += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) - float(np.mean(correct[mask]))
            )
        return error

    def _feature_importance_summary(
        self, model: Any, limit: int = 10
    ) -> list[dict[str, float | str]]:
        feature_names = list(self.feature_names)
        estimator = model
        if isinstance(model, Pipeline):
            estimator = model.named_steps["classifier"]
            preprocessor = model.named_steps["preprocessor"]
            try:
                numeric_indices = preprocessor.transformers_[0][2]
                feature_names = [
                    self.feature_names[int(index)] for index in numeric_indices
                ]
            except (AttributeError, IndexError, TypeError, ValueError):
                return []
        raw_importance = getattr(estimator, "feature_importances_", None)
        if raw_importance is None:
            coefficients = getattr(estimator, "coef_", None)
            if coefficients is not None:
                raw_importance = np.mean(np.abs(coefficients), axis=0)
        if raw_importance is None:
            return []
        importance = np.asarray(raw_importance, dtype=float).reshape(-1)
        if len(importance) != len(feature_names):
            return []
        total = float(importance.sum())
        if not math.isfinite(total) or total <= 0:
            return []
        normalized = importance / total
        indices = np.argsort(normalized)[::-1][:limit]
        return [
            {
                "feature": feature_names[int(index)],
                "importance": round(float(normalized[index]), 6),
            }
            for index in indices
        ]

    def train_pipeline(self, rows: List[Any]) -> bool:
        """
        Gathers database features, splits into train/validation,
        evaluates candidate models, performs isotonic calibration on the winner,
        and saves the best-performing model to disk.
        """
        if len(rows) < settings.MIN_TRAINING_SAMPLES:
            logger.warning(
                f"Insufficient training samples: {len(rows)}/{settings.MIN_TRAINING_SAMPLES}. Skipping training."
            )
            return False

        champion_predictor = self.calibrator or self.model
        champion_feature_names = list(self.feature_names)

        # Retraining upgrades legacy artifacts to the current inference schema.
        self.feature_names = list(FeatureEngine.FEATURE_NAMES)

        # Sort first so no future match can enter an earlier training window.
        chronological_rows = sorted(rows, key=self._training_sort_key)

        # Extract features and targets from DB rows.
        X_list = []
        y_list = []
        source_counts: dict[str, int] = {}
        leagues: set[int] = set()
        sample_timestamps: list[float] = []

        for row in chronological_rows:
            if not row.actual_result:
                continue

            feature_dict = FeatureEngine.build_training_features(row)
            feats = [feature_dict[name] for name in self.feature_names]
            label = self.LABEL_MAP.get(row.actual_result)
            if label is None or not all(math.isfinite(value) for value in feats):
                continue
            X_list.append(feats)
            y_list.append(label)
            source = str(getattr(row, "training_source", "labeled_prediction"))
            source_counts[source] = source_counts.get(source, 0) + 1
            league_id = getattr(row, "league_id", None)
            if isinstance(league_id, int):
                leagues.add(league_id)
            sample_timestamps.append(self._training_sort_key(row))

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        if len(X) < settings.MIN_TRAINING_SAMPLES:
            return False

        outcome_count = len(self.LABEL_MAP)
        test_size = max(outcome_count, math.ceil(len(y) * 0.2))
        calibration_size = max(outcome_count * 2, math.ceil(len(y) * 0.15))
        train_end = len(y) - calibration_size - test_size
        calibration_end = len(y) - test_size
        if train_end < outcome_count * 2:
            logger.warning(
                "Training skipped: temporal train/calibration/test windows are too small."
            )
            return False

        X_train, y_train = X[:train_end], y[:train_end]
        X_calibration, y_calibration = (
            X[train_end:calibration_end],
            y[train_end:calibration_end],
        )
        X_test, y_test = X[calibration_end:], y[calibration_end:]
        calibration_fit_size = max(outcome_count, len(y_calibration) // 2)
        X_calibration_fit = X_calibration[:calibration_fit_size]
        y_calibration_fit = y_calibration[:calibration_fit_size]
        X_calibration_validation = X_calibration[calibration_fit_size:]
        y_calibration_validation = y_calibration[calibration_fit_size:]
        if not all(
            self._has_all_outcomes(labels)
            for labels in (
                y_train,
                y_calibration_fit,
                y_calibration_validation,
                y_test,
            )
        ):
            logger.warning(
                "Training skipped: every temporal train/calibration/test window "
                "must contain all outcomes."
            )
            return False

        candidates = self._get_candidate_models()
        best_model_name = None
        best_template = None
        best_brier = float("inf")
        best_cv_loss = float("inf")
        splitter = TimeSeriesSplit(n_splits=3)

        for name, template in candidates:
            try:
                fold_metrics: list[tuple[float, float]] = []
                for fold_train_indices, fold_validation_indices in splitter.split(
                    X_train
                ):
                    fold_y_train = y_train[fold_train_indices]
                    fold_y_validation = y_train[fold_validation_indices]
                    if not self._has_all_outcomes(
                        fold_y_train
                    ) or not self._has_all_outcomes(fold_y_validation):
                        continue

                    fold_model = clone(template)
                    fold_model.fit(X_train[fold_train_indices], fold_y_train)
                    fold_probabilities = np.asarray(
                        fold_model.predict_proba(X_train[fold_validation_indices]),
                        dtype=float,
                    )
                    fold_metrics.append(
                        self._probability_metrics(fold_y_validation, fold_probabilities)
                    )

                if not fold_metrics:
                    logger.warning(
                        "Model candidate %s has no valid temporal folds.", name
                    )
                    continue

                brier = float(np.mean([metric[0] for metric in fold_metrics]))
                loss = float(np.mean([metric[1] for metric in fold_metrics]))

                logger.info(
                    "Model Candidate: %s | Walk-forward Brier: %.4f | Log Loss: %.4f",
                    name,
                    brier,
                    loss,
                )

                if brier < best_brier:
                    best_brier = brier
                    best_cv_loss = loss
                    best_template = template
                    best_model_name = name
            except Exception as e:
                logger.error(f"Failed training candidate model {name}: {e}")

        if best_template is None or not best_model_name:
            logger.error("No model candidates trained successfully.")
            return False

        # Calibration stage utilizing Isotonic Calibration from calibrate service
        from app.prediction.ml.calibrate import MultiClassCalibrator

        logger.info(
            "Selected best model: %s (walk-forward Brier: %.4f). "
            "Starting holdout calibration.",
            best_model_name,
            best_brier,
        )

        best_model = clone(best_template)
        best_model.fit(X_train, y_train)
        calibrator = MultiClassCalibrator(best_model)
        calibrator.fit(X_calibration_fit, y_calibration_fit)
        raw_selection_probabilities = np.asarray(
            best_model.predict_proba(X_calibration_validation), dtype=float
        )
        calibrated_selection_probabilities = np.asarray(
            calibrator.predict_proba(X_calibration_validation), dtype=float
        )
        raw_selection_brier, raw_selection_loss = self._probability_metrics(
            y_calibration_validation, raw_selection_probabilities
        )
        calibrated_selection_brier, calibrated_selection_loss = (
            self._probability_metrics(
                y_calibration_validation, calibrated_selection_probabilities
            )
        )
        calibration_applied = (
            len(y_calibration_fit) >= settings.MIN_ISOTONIC_CALIBRATION_SAMPLES
            and calibrated_selection_brier <= raw_selection_brier
            and raw_selection_loss - calibrated_selection_loss
            >= settings.MIN_CALIBRATION_LOG_LOSS_IMPROVEMENT
        )
        selected_calibrator = calibrator if calibration_applied else None
        test_predictor = selected_calibrator or best_model
        test_probabilities = np.asarray(
            test_predictor.predict_proba(X_test), dtype=float
        )
        test_brier, test_loss = self._probability_metrics(y_test, test_probabilities)
        test_calibration_error = self._multiclass_calibration_error(
            y_test, test_probabilities
        )
        test_predictions = test_probabilities.argmax(axis=1)
        test_accuracy = float(accuracy_score(y_test, test_predictions))
        test_macro_f1 = float(
            f1_score(y_test, test_predictions, average="macro", zero_division=0)
        )
        class_counts = np.bincount(y_train, minlength=outcome_count).astype(float)
        class_priors = class_counts / class_counts.sum()
        baseline_probabilities = np.tile(class_priors, (len(y_test), 1))
        baseline_brier, baseline_loss = self._probability_metrics(
            y_test, baseline_probabilities
        )
        champion_brier: float | None = None
        champion_loss: float | None = None
        champion_feature_indices = (
            [self.feature_names.index(name) for name in champion_feature_names]
            if champion_feature_names
            and all(name in self.feature_names for name in champion_feature_names)
            else None
        )
        if champion_predictor is not None and champion_feature_indices is not None:
            try:
                champion_probabilities = np.asarray(
                    champion_predictor.predict_proba(
                        X_test[:, champion_feature_indices]
                    ),
                    dtype=float,
                )
                champion_brier, champion_loss = self._probability_metrics(
                    y_test, champion_probabilities
                )
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning("Champion comparison skipped: %s", exc)

        label_distribution = {
            self.INV_LABEL_MAP[index]: int(count)
            for index, count in enumerate(
                np.bincount(y, minlength=outcome_count).tolist()
            )
        }
        best_metrics: Dict[str, object] = {
            "brier_score": test_brier,
            "log_loss": test_loss,
            "calibration_error": test_calibration_error,
            "accuracy": test_accuracy,
            "macro_f1": test_macro_f1,
            "calibration_applied": calibration_applied,
            "calibration_minimum_samples": (settings.MIN_ISOTONIC_CALIBRATION_SAMPLES),
            "calibration_selection_raw_brier": raw_selection_brier,
            "calibration_selection_calibrated_brier": calibrated_selection_brier,
            "calibration_selection_raw_log_loss": raw_selection_loss,
            "calibration_selection_calibrated_log_loss": calibrated_selection_loss,
            "baseline_brier_score": baseline_brier,
            "baseline_log_loss": baseline_loss,
            "brier_improvement_vs_baseline": baseline_brier - test_brier,
            "champion_brier_score": champion_brier,
            "champion_log_loss": champion_loss,
            "brier_improvement_vs_champion": (
                champion_brier - test_brier if champion_brier is not None else None
            ),
            "walk_forward_brier_score": best_brier,
            "walk_forward_log_loss": best_cv_loss,
            "samples": float(len(X)),
            "training_samples": float(len(X_train)),
            "calibration_samples": float(len(X_calibration)),
            "calibration_fit_samples": float(len(X_calibration_fit)),
            "calibration_validation_samples": float(len(X_calibration_validation)),
            "test_samples": float(len(X_test)),
            "training_sources": source_counts,
            "league_count": len(leagues),
            "label_distribution": label_distribution,
            "top_features": self._feature_importance_summary(best_model),
            "sample_start_timestamp": min(sample_timestamps),
            "sample_end_timestamp": max(sample_timestamps),
            "evaluation_strategy": "walk_forward_temporal_holdout",
        }
        guard_failures = {
            "brier_score": (
                test_brier,
                settings.MAX_MODEL_BRIER_SCORE,
            ),
            "log_loss": (
                test_loss,
                settings.MAX_MODEL_LOG_LOSS,
            ),
            "calibration_error": (
                test_calibration_error,
                settings.MAX_MODEL_CALIBRATION_ERROR,
            ),
        }
        rejected_metrics = {
            name: {"actual": actual, "maximum": maximum}
            for name, (actual, maximum) in guard_failures.items()
            if actual > maximum
        }
        if baseline_brier - test_brier < settings.MIN_MODEL_BASELINE_BRIER_IMPROVEMENT:
            rejected_metrics["baseline_brier_improvement"] = {
                "actual": baseline_brier - test_brier,
                "minimum": settings.MIN_MODEL_BASELINE_BRIER_IMPROVEMENT,
            }
        if test_loss - baseline_loss > settings.MAX_MODEL_BASELINE_LOG_LOSS_REGRESSION:
            rejected_metrics["baseline_log_loss"] = {
                "actual": test_loss,
                "maximum": (
                    baseline_loss + settings.MAX_MODEL_BASELINE_LOG_LOSS_REGRESSION
                ),
            }
        if champion_brier is not None and champion_loss is not None:
            brier_improved = (
                champion_brier - test_brier
                >= settings.MIN_MODEL_CHAMPION_BRIER_IMPROVEMENT
            )
            log_loss_tradeoff_accepted = (
                champion_loss - test_loss
                >= settings.MIN_MODEL_CHAMPION_LOG_LOSS_IMPROVEMENT
                and test_brier - champion_brier
                <= settings.MAX_MODEL_CHAMPION_BRIER_REGRESSION
            )
            if not brier_improved and not log_loss_tradeoff_accepted:
                rejected_metrics["champion_comparison"] = {
                    "brier_improvement": champion_brier - test_brier,
                    "log_loss_improvement": champion_loss - test_loss,
                }
        if rejected_metrics:
            logger.warning(
                "Candidate model rejected by quality guard; active model preserved: %s",
                rejected_metrics,
            )
            return False

        # Persist first; an artifact write failure must leave the champion active.
        self._save_active_model(
            best_model, selected_calibrator, best_model_name, best_metrics
        )
        self.model = best_model
        self.calibrator = selected_calibrator
        self.active_model_name = best_model_name
        self.metrics = best_metrics
        self.is_ready = True
        return True

    def predict_match(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Runs model inference and returns calibrated probabilities."""
        self._refresh_artifact_if_changed()
        if not self.is_ready or not self.model:
            return {"ready": False, "probabilities": None}

        try:
            features = np.array(
                [[feature_dict.get(k, 0.0) for k in self.feature_names]], dtype=float
            )
            if not np.all(np.isfinite(features)):
                raise ValueError("Feature vector contains non-finite values")

            predictor = self.calibrator or self.model
            probs = np.asarray(predictor.predict_proba(features)[0], dtype=float)
            if (
                probs.shape != (len(self.INV_LABEL_MAP),)
                or not np.all(np.isfinite(probs))
                or np.any(probs < 0)
                or float(probs.sum()) <= 0
            ):
                raise ValueError("Model returned invalid class probabilities")
            probs = probs / probs.sum()
        except (TypeError, ValueError, IndexError, AttributeError) as exc:
            self.runtime_stats["inference_failure"] += 1
            logger.error(f"ML inference rejected invalid model output: {exc}")
            return {"ready": False, "probabilities": None}

        # Structure output labels
        result_probs = {
            self.INV_LABEL_MAP[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(probs))
        }

        # Predict final outcome
        prediction = max(result_probs, key=lambda outcome: result_probs[outcome])
        self.runtime_stats["inference_success"] += 1

        raw_training_samples = self.metrics.get("samples", 0)
        training_samples = (
            int(raw_training_samples)
            if isinstance(raw_training_samples, (int, float))
            and not isinstance(raw_training_samples, bool)
            and math.isfinite(raw_training_samples)
            and raw_training_samples >= 0
            else 0
        )
        return {
            "ready": True,
            "prediction": prediction,
            "probability": result_probs[prediction],
            "all_probabilities": result_probs,
            "model_name": self.active_model_name,
            "artifact_version": self.artifact_version,
            "training_samples": training_samples,
        }


# Global active model instance
ml_pipeline = MLModelPipeline()
