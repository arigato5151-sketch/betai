from types import SimpleNamespace

import numpy as np

from backend.app.prediction.ml import explain
from backend.app.prediction.ml.explain import ExplainabilityService


def test_feature_importance_fallback_is_normalized_and_sorted(monkeypatch):
    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)
    model = SimpleNamespace(feature_importances_=np.array([0.6, 0.3, 0.1]))

    result = ExplainabilityService.generate_explanation(
        model, {}, ["home_attack", "away_form", "home_elo"]
    )

    assert result == [
        "Tahmin sebebi: Ev Sahibi Hücum Gücü (+%60), "
        "Deplasman Formu (+%30), Ev Sahibi ELO Derecesi (+%10)"
    ]


def test_unknown_feature_uses_raw_name(monkeypatch):
    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)
    model = SimpleNamespace(feature_importances_=[1.0])

    result = ExplainabilityService.generate_explanation(model, {}, ["weather_index"])

    assert result == ["Tahmin sebebi: weather_index (+%100)"]


def test_missing_importances_use_uniform_weights(monkeypatch):
    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)

    result = ExplainabilityService.generate_explanation(
        object(), {}, ["first", "second", "third", "fourth"]
    )

    assert result == ["Tahmin sebebi: first (+%25), second (+%25), third (+%25)"]


def test_empty_feature_list_returns_generic_explanation(monkeypatch):
    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)

    result = ExplainabilityService.generate_explanation(object(), {}, [])

    assert result == [ExplainabilityService.GENERIC_EXPLANATION]


def test_invalid_importances_fall_back_to_uniform_weights(monkeypatch):
    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)
    model = SimpleNamespace(feature_importances_=[np.nan, -1.0, np.inf])

    result = ExplainabilityService.generate_explanation(
        model, {}, ["first", "second", "third"]
    )

    assert result == ["Tahmin sebebi: first (+%33), second (+%33), third (+%33)"]


def test_shap_runtime_failure_uses_importance_fallback(monkeypatch):
    class BrokenExplainer:
        def __init__(self, model):
            raise RuntimeError("unsupported model")

    monkeypatch.setattr(explain, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        explain,
        "shap",
        SimpleNamespace(TreeExplainer=BrokenExplainer),
        raising=False,
    )
    model = SimpleNamespace(feature_importances_=[0.8, 0.2])

    result = ExplainabilityService.generate_explanation(
        model, {}, ["home_form", "away_form"]
    )

    assert result == ["Tahmin sebebi: Ev Sahibi Formu (+%80), Deplasman Formu (+%20)"]


def test_multiclass_shap_values_are_aggregated(monkeypatch):
    class FakeExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, values):
            assert values.shape == (1, 3)
            return [
                np.array([[0.6, 0.2, 0.1]]),
                np.array([[-0.3, 0.1, 0.1]]),
                np.array([[0.3, -0.3, 0.1]]),
            ]

    monkeypatch.setattr(explain, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        explain,
        "shap",
        SimpleNamespace(TreeExplainer=FakeExplainer),
        raising=False,
    )

    result = ExplainabilityService.generate_explanation(
        object(),
        {"home_attack": 1.2, "away_form": 0.4, "home_elo": 1500.0},
        ["home_attack", "away_form", "home_elo"],
    )

    assert result == [
        "Tahmin sebebi: Ev Sahibi Hücum Gücü (+%57), "
        "Deplasman Formu (+%29), Ev Sahibi ELO Derecesi (+%14)"
    ]


def test_shap_shape_mismatch_uses_importance_fallback(monkeypatch):
    class ShortExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, values):
            return np.array([[0.9]])

    monkeypatch.setattr(explain, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(
        explain,
        "shap",
        SimpleNamespace(TreeExplainer=ShortExplainer),
        raising=False,
    )
    model = SimpleNamespace(feature_importances_=[0.25, 0.75])

    result = ExplainabilityService.generate_explanation(
        model, {}, ["home_form", "away_form"]
    )

    assert result == ["Tahmin sebebi: Deplasman Formu (+%75), Ev Sahibi Formu (+%25)"]
