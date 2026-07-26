from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class NativeCategoricalBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible adapter for native CatBoost/LightGBM categories."""

    def __init__(
        self,
        *,
        backend: Literal["catboost", "lightgbm"],
        feature_names: tuple[str, ...],
        categorical_feature_names: tuple[str, ...],
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        n_jobs: int = 2,
    ) -> None:
        self.backend = backend
        self.feature_names = feature_names
        self.categorical_feature_names = categorical_feature_names
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.n_jobs = n_jobs

    @staticmethod
    def _category_token(value: object) -> str:
        if value is None:
            return "__MISSING__"
        try:
            numeric = float(str(value))
        except (TypeError, ValueError):
            token = str(value).strip()
            return token or "__MISSING__"
        if not math.isfinite(numeric):
            return "__MISSING__"
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".17g")

    def _as_frame(self, features: Any, *, fitting: bool) -> pd.DataFrame:
        matrix = np.asarray(features)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                "Feature matrix must be two-dimensional and match feature_names"
            )

        frame = pd.DataFrame(matrix, columns=list(self.feature_names))
        categorical_names = set(self.categorical_feature_names)
        for name in self.feature_names:
            if name in categorical_names:
                tokens = frame[name].map(self._category_token)
                if self.backend == "lightgbm":
                    categories: list[str] | None
                    if fitting:
                        categories = sorted(set(cast(list[str], tokens.tolist())))
                        self.category_levels_[name] = categories
                    else:
                        categories = self.category_levels_.get(name)
                        if categories is None:
                            raise ValueError(
                                f"Missing fitted categories for feature {name}"
                            )
                    frame[name] = pd.Categorical(tokens, categories=categories)
                else:
                    frame[name] = tokens.astype(str)
                continue

            numeric = pd.to_numeric(frame[name], errors="coerce").astype(float)
            if not np.all(np.isfinite(numeric.to_numpy())):
                raise ValueError(f"Feature {name} contains non-finite values")
            frame[name] = numeric
        return frame

    def fit(self, features: Any, labels: Any) -> "NativeCategoricalBoostingClassifier":
        self.category_levels_: dict[str, list[str]] = {}
        frame = self._as_frame(features, fitting=True)
        target = np.asarray(labels, dtype=int)
        if target.ndim != 1 or len(target) != len(frame):
            raise ValueError("Labels must be one-dimensional and match feature rows")

        if self.backend == "catboost":
            from catboost import CatBoostClassifier

            estimator: Any = CatBoostClassifier(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                loss_function="MultiClass",
                random_seed=self.random_state,
                thread_count=self.n_jobs,
                auto_class_weights="Balanced",
                allow_writing_files=False,
                verbose=False,
            )
            estimator.fit(frame, target)
        elif self.backend == "lightgbm":
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight="balanced",
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            estimator.fit(
                frame,
                target,
                categorical_feature=list(self.categorical_feature_names),
            )
        else:  # pragma: no cover - Literal protects normal construction.
            raise ValueError(f"Unsupported categorical booster backend: {self.backend}")

        self.estimator_ = estimator
        self.classes_ = np.asarray(estimator.classes_, dtype=int)
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        frame = self._as_frame(features, fitting=False)
        return np.asarray(self.estimator_.predict_proba(frame), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.feature_importances_, dtype=float)
