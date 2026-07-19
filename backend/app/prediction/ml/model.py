import os
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from app.core.config import settings
from app.core.logging_config import logger
from app.prediction.ml.features import FeatureEngine

# Try imports with robust fallbacks
try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class MLModelPipeline:
    LABEL_MAP = {"HOME_WIN": 0, "DRAW": 1, "AWAY_WIN": 2}
    INV_LABEL_MAP = {0: "HOME_WIN", 1: "DRAW", 2: "AWAY_WIN"}

    def __init__(self):
        self.model: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.feature_names = list(FeatureEngine.FEATURE_NAMES)
        self.is_ready = False
        self.active_model_name: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self.artifact_version: Optional[str] = None
        self.runtime_stats = {"inference_success": 0, "inference_failure": 0}
        self._artifact_mtime_ns: int | None = None

    @staticmethod
    def _previous_model_path() -> Path:
        return Path(settings.ACTIVE_MODEL_PATH).with_name("previous_model.pkl")

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.is_ready,
            "model_name": self.active_model_name,
            "artifact_version": self.artifact_version,
            "metrics": dict(self.metrics),
            "runtime": dict(self.runtime_stats),
            "rollback_available": self._previous_model_path().is_file(),
        }

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

        try:
            payload = joblib.load(settings.ACTIVE_MODEL_PATH)
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
        except Exception as e:
            logger.error(f"Failed to load ML model artifact from disk: {e}")
            self.model = None
            self.calibrator = None
            self.is_ready = False
            self.active_model_name = None
            self.artifact_version = None
            self._artifact_mtime_ns = None
            return False

    def _save_active_model(
        self, model: Any, calibrator: Any, model_name: str, metrics: Dict[str, float]
    ) -> None:
        artifacts_dir = Path(settings.MODEL_ARTIFACTS_DIR)
        versions_dir = artifacts_dir / "versions"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        versions_dir.mkdir(parents=True, exist_ok=True)
        active_path = Path(settings.ACTIVE_MODEL_PATH)
        previous_path = self._previous_model_path()
        version = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        payload = {
            "schema_version": 1,
            "artifact_version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "model": model,
            "calibrator": calibrator,
            "feature_names": self.feature_names,
            "feature_schema_version": FeatureEngine.SCHEMA_VERSION,
            "model_name": model_name,
            "metrics": metrics,
        }
        temporary_path = active_path.with_suffix(".tmp")
        joblib.dump(payload, temporary_path)
        if active_path.is_file():
            shutil.copy2(active_path, previous_path)
        os.replace(temporary_path, active_path)
        shutil.copy2(active_path, versions_dir / f"model_{version}.pkl")
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
        if not active_path.is_file() or not previous_path.is_file():
            return False

        try:
            previous_payload = joblib.load(previous_path)
            if "model" not in previous_payload:
                raise ValueError("Previous artifact has no model")
            swap_path = active_path.with_suffix(".rollback.tmp")
            os.replace(active_path, swap_path)
            os.replace(previous_path, active_path)
            os.replace(swap_path, previous_path)
            if self.load_active_model():
                logger.warning(
                    "Rolled back active ML model to %s", self.artifact_version
                )
                return True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("ML model rollback failed: %s", exc)
        return False

    def _get_candidate_models(self) -> List[Tuple[str, Any]]:
        candidates = []

        if XGB_AVAILABLE:
            candidates.append(
                (
                    "XGBoost",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=5,
                        learning_rate=0.08,
                        random_state=42,
                        eval_metric="mlogloss",
                    ),
                )
            )
        else:
            logger.warning(
                "XGBoost is not installed. Using Gradient Boosting fallback."
            )
            candidates.append(
                ("Gradient Boosting", GradientBoostingClassifier(random_state=42))
            )

        if LGBM_AVAILABLE:
            candidates.append(
                (
                    "LightGBM",
                    LGBMClassifier(
                        n_estimators=150,
                        max_depth=5,
                        learning_rate=0.08,
                        random_state=42,
                        verbosity=-1,
                    ),
                )
            )

        if CATBOOST_AVAILABLE:
            candidates.append(
                (
                    "CatBoost",
                    CatBoostClassifier(
                        iterations=150,
                        depth=5,
                        learning_rate=0.08,
                        random_seed=42,
                        verbose=False,
                    ),
                )
            )

        # Always add Random Forest as a standard robust baseline
        candidates.append(
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
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

        # Retraining upgrades legacy artifacts to the current inference schema.
        self.feature_names = list(FeatureEngine.FEATURE_NAMES)

        # Sort first so no future match can enter an earlier training window.
        chronological_rows = sorted(rows, key=self._training_sort_key)

        # Extract features and targets from DB rows.
        X_list = []
        y_list = []

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

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        if len(X) < settings.MIN_TRAINING_SAMPLES:
            return False

        outcome_count = len(self.LABEL_MAP)
        test_size = max(outcome_count, math.ceil(len(y) * 0.2))
        calibration_size = max(outcome_count, math.ceil(len(y) * 0.15))
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
        if not all(
            self._has_all_outcomes(labels)
            for labels in (y_train, y_calibration, y_test)
        ):
            logger.warning(
                "Training skipped: every temporal window must contain all outcomes."
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
        calibrator.fit(X_calibration, y_calibration)
        test_probabilities = np.asarray(calibrator.predict_proba(X_test), dtype=float)
        test_brier, test_loss = self._probability_metrics(y_test, test_probabilities)
        best_metrics = {
            "brier_score": test_brier,
            "log_loss": test_loss,
            "walk_forward_brier_score": best_brier,
            "walk_forward_log_loss": best_cv_loss,
            "samples": float(len(X)),
            "training_samples": float(len(X_train)),
            "calibration_samples": float(len(X_calibration)),
            "test_samples": float(len(X_test)),
            "evaluation_strategy": "walk_forward_temporal_holdout",
        }

        # Save active model
        self.model = best_model
        self.calibrator = calibrator
        self.active_model_name = best_model_name
        self.metrics = best_metrics
        self.is_ready = True

        self._save_active_model(best_model, calibrator, best_model_name, best_metrics)
        return True

    def predict_match(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Runs model inference and returns calibrated probabilities."""
        active_path = Path(settings.ACTIVE_MODEL_PATH)
        try:
            current_mtime = active_path.stat().st_mtime_ns
            if current_mtime != self._artifact_mtime_ns:
                self.load_active_model()
        except OSError:
            pass
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

        return {
            "ready": True,
            "prediction": prediction,
            "probability": result_probs[prediction],
            "all_probabilities": result_probs,
            "model_name": self.active_model_name,
            "artifact_version": self.artifact_version,
        }


# Global active model instance
ml_pipeline = MLModelPipeline()
