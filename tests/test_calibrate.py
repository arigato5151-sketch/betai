import numpy as np
import pytest

from app.prediction.ml.calibrate import MultiClassCalibrator


class _FakeClassifier:
    """Deterministic raw predictor whose probabilities the calibrator reshapes."""

    def __init__(self, raw: np.ndarray) -> None:
        self._raw = raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._raw


def _toy_tables(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Three perfectly separable 1X2 classes with distinct raw probabilities."""
    rng = np.random.RandomState(7)
    classes = []
    raw = []
    centers = np.linspace(0.45, 0.92, 3)
    for label, center in enumerate(centers):
        for _ in range(n_per_class):
            proba_home = np.clip(center + rng.normal(0.0, 0.08), 0.05, 0.95)
            remaining = 1.0 - proba_home
            draw = remaining * rng.uniform(0.2, 0.8)
            if label == 0:
                proba = np.array([proba_home, draw, remaining - draw])
            elif label == 1:
                proba = np.array([draw, proba_home, remaining - draw])
            else:
                proba = np.array([draw, remaining - draw, proba_home])
            raw.append(proba / max(proba.sum(), 1e-9))
            classes.append(label)
    return np.asarray(raw), np.asarray(classes)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
@pytest.mark.parametrize("n_per_class", [40, 200])
def test_multiclass_calibrator_applies_method_and_renormalizes(
    method: str, n_per_class: int
) -> None:
    raw, labels = _toy_tables(n_per_class)
    calibrator = MultiClassCalibrator(_FakeClassifier(raw), method=method)

    calibrator.fit(raw, labels)
    out = calibrator.apply(raw)

    assert out.shape == raw.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_delegates_to_apply_over_base_output() -> None:
    raw, labels = _toy_tables(30)
    calibrator = MultiClassCalibrator(_FakeClassifier(raw), method="isotonic")
    calibrator.fit(raw, labels)

    assert np.allclose(calibrator.predict_proba(raw), calibrator.apply(raw), atol=1e-12)


def test_multiclass_calibrator_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        MultiClassCalibrator(_FakeClassifier(np.zeros((1, 3))), method="spline")
