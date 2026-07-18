import numpy as np
from typing import List, Any
from sklearn.isotonic import IsotonicRegression


class MultiClassCalibrator:
    """
    Fits one-vs-rest Isotonic Regression calibrators over a multi-class classifier's raw probabilities.
    Ensures that class probabilities sum to 1.0 after calibration.
    """

    def __init__(self, base_clf: Any):
        self.base_clf = base_clf
        self.calibrators: List[IsotonicRegression] = []
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

            # Fit isotonic regression mapping x_raw -> y_bin
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(x_raw, y_bin)
            self.calibrators.append(ir)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Get raw predictions
        raw_probs = self.base_clf.predict_proba(X)
        calibrated_probs = np.zeros_like(raw_probs)

        # Apply isotonic regression for each class
        for i in range(self.n_classes):
            calibrated_probs[:, i] = self.calibrators[i].predict(raw_probs[:, i])

        # Normalize probabilities so that each row sums to 1.0
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        zero_rows = row_sums[:, 0] <= 0
        if np.any(zero_rows):
            # Isotonic models can all return zero on sparse edges; retain base probabilities.
            calibrated_probs[zero_rows] = raw_probs[zero_rows]
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        return calibrated_probs / row_sums
