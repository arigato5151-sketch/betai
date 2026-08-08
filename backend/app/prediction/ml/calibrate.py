import numpy as np
from typing import Any, List
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class MultiClassCalibrator:
    """
    Fits one-vs-rest probability calibrators over a multi-class classifier's raw
    probabilities. ``method="isotonic"`` uses Isotonic Regression (default, keeps
    backward compatibility); ``method="platt"`` uses a logistic (sigmoid) fit per
    class. Ensures that the calibrated probabilities sum to 1.0.
    """

    def __init__(self, base_clf: Any, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.base_clf = base_clf
        self.method = method
        self.calibrators: List[Any] = []
        self.n_classes: int = 3  # HOME_WIN, DRAW, AWAY_WIN

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiClassCalibrator":
        # Get raw probabilities from the fitted base classifier
        raw_probs = self.base_clf.predict_proba(X)
        self.calibrators = []

        for i in range(self.n_classes):
            # Target binary label for class i
            y_bin = (y == i).astype(int)

            # Extract raw predictions for class i
            x_raw = raw_probs[:, i]

            if self.method == "platt":
                sigmoid = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
                sigmoid.fit(x_raw.reshape(-1, 1), y_bin)
                self.calibrators.append(sigmoid)
            else:
                # Fit isotonic regression mapping x_raw -> y_bin
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(x_raw, y_bin)
                self.calibrators.append(ir)

        return self

    def apply(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply the fitted per-class transforms to a raw probability matrix."""
        calibrated_probs = np.zeros_like(raw_probs)

        for i in range(self.n_classes):
            x_raw = raw_probs[:, i]
            if self.method == "platt":
                calibrated_probs[:, i] = self.calibrators[i].predict_proba(
                    x_raw.reshape(-1, 1)
                )[:, 1]
            else:
                calibrated_probs[:, i] = self.calibrators[i].predict(x_raw)

        # Normalize probabilities so that each row sums to 1.0
        calibrated_probs = np.clip(calibrated_probs, 0.0, 1.0)
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0
        if np.any(zero_rows):
            # Isotonic models can all return zero on sparse edges; retain base probabilities.
            calibrated_probs[zero_rows] = raw_probs[zero_rows]
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        return calibrated_probs / row_sums

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Get raw predictions, then calibrate them with the fitted transforms
        raw_probs = np.asarray(self.base_clf.predict_proba(X), dtype=float)
        if not self.calibrators:
            return raw_probs
        return self.apply(raw_probs)
