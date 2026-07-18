import os
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from app.core.config import settings
from app.core.logging_config import logger

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
        self.feature_names = [
            "home_form",
            "home_attack",
            "home_defense",
            "home_xg",
            "away_form",
            "away_attack",
            "away_defense",
            "away_xg",
            "home_form_ema",
            "away_form_ema",
            "rest_days_diff",
            "home_clean_sheet_streak",
            "away_clean_sheet_streak",
            "home_scoring_streak",
            "away_scoring_streak",
            "h2h_home_win_rate",
            "h2h_draw_rate",
            "h2h_home_loss_rate",
            "home_elo",
            "away_elo",
        ]
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

        # Extract features and targets from DB rows
        X_list = []
        y_list = []

        for row in rows:
            if not row.actual_result:
                continue

            # Build feature array in strict order
            feats = [
                float(row.home_form or 50),
                float(row.home_attack or 50),
                float(row.home_defense or 50),
                float(row.home_xg or 1.2),
                float(row.away_form or 50),
                float(row.away_attack or 50),
                float(row.away_defense or 50),
                float(row.away_xg or 1.2),
                float(row.home_form or 50),  # home_form_ema fallback
                float(row.away_form or 50),  # away_form_ema fallback
                0.0,  # rest_days_diff fallback
                0.0,  # streaks
                0.0,
                0.0,
                0.0,
                0.33,  # H2H rates
                0.33,
                0.34,
                1500.0,  # Elo rates
                1500.0,
            ]
            label = self.LABEL_MAP.get(row.actual_result)
            if label is None or not all(math.isfinite(value) for value in feats):
                continue
            X_list.append(feats)
            y_list.append(label)

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)

        if len(X) < settings.MIN_TRAINING_SAMPLES:
            return False

        class_counts = np.bincount(y, minlength=len(self.LABEL_MAP))
        validation_size = math.ceil(len(y) * 0.2)
        if np.any(class_counts < 2) or validation_size < len(self.LABEL_MAP):
            logger.warning(
                "Training skipped: each outcome needs at least two samples and the validation split must contain all classes."
            )
            return False

        # Train/Validation split (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        candidates = self._get_candidate_models()
        best_model_name = None
        best_model = None
        best_brier = float("inf")
        best_metrics = {}

        for name, clf in candidates:
            try:
                clf.fit(X_train, y_train)
                val_probs = clf.predict_proba(X_val)

                # Multi-class Brier score = (1/N) * sum((f_i - o_i)^2)
                # Compute brier manually for multiclass
                o_onehot = np.zeros_like(val_probs)
                o_onehot[np.arange(len(y_val)), y_val] = 1.0
                brier = float(np.mean(np.sum((val_probs - o_onehot) ** 2, axis=1)))
                loss = log_loss(y_val, val_probs)

                logger.info(
                    f"Model Candidate: {name} | Brier: {round(brier, 4)} | Log Loss: {round(loss, 4)}"
                )

                if brier < best_brier:
                    best_brier = brier
                    best_model = clf
                    best_model_name = name
                    best_metrics = {
                        "brier_score": brier,
                        "log_loss": loss,
                        "samples": float(len(X)),
                    }
            except Exception as e:
                logger.error(f"Failed training candidate model {name}: {e}")

        if not best_model or not best_model_name:
            logger.error("No model candidates trained successfully.")
            return False

        # Calibration stage utilizing Isotonic Calibration from calibrate service
        from app.prediction.ml.calibrate import MultiClassCalibrator

        logger.info(
            f"Selected best model: {best_model_name} (Brier: {round(best_brier, 4)}). Starting Isotonic calibration..."
        )

        calibrator = MultiClassCalibrator(best_model)
        calibrator.fit(X_train, y_train)

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
