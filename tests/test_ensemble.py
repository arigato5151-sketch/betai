import pytest

from app.core.config import settings
from app.prediction.ensemble import ProbabilityEnsembler


def stats_analysis(probabilities: dict[str, float]) -> dict:
    prediction = max(probabilities, key=probabilities.get) if probabilities else "DRAW"
    return {
        "model": "poisson_dixon_coles_v3",
        "prediction": prediction,
        "probability": probabilities.get(prediction, 0.0),
        "all_probabilities": probabilities,
        "confidence_gap": 0.0,
        "confidence_tier": "DUSUK",
    }


@pytest.fixture(autouse=True)
def deterministic_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ENSEMBLE_STATS_WEIGHT", 0.4)
    monkeypatch.setattr(settings, "ENSEMBLE_ML_WEIGHT", 0.2)
    monkeypatch.setattr(settings, "ENSEMBLE_MARKET_WEIGHT", 0.4)


def test_three_source_ensemble_blends_normalized_probabilities() -> None:
    result = ProbabilityEnsembler.apply(
        stats_analysis({"HOME_WIN": 50.0, "DRAW": 30.0, "AWAY_WIN": 20.0}),
        ml_result={
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": 20.0,
                "DRAW": 30.0,
                "AWAY_WIN": 50.0,
            },
        },
        market={
            "fair_probability": {
                "HOME_WIN": 40.0,
                "DRAW": 30.0,
                "AWAY_WIN": 30.0,
            }
        },
    )

    assert result["all_probabilities"] == {
        "HOME_WIN": 40.0,
        "DRAW": 30.0,
        "AWAY_WIN": 30.0,
    }
    assert result["prediction"] == "HOME_WIN"
    assert result["confidence_gap"] == 10.0
    assert result["confidence_tier"] == "ORTA"
    assert result["model"] == ProbabilityEnsembler.VERSION
    assert result["ensemble"]["weights"] == {
        "stats": 0.4,
        "ml": 0.2,
        "market": 0.4,
    }


def test_missing_market_renormalizes_available_source_weights() -> None:
    result = ProbabilityEnsembler.apply(
        stats_analysis({"HOME_WIN": 60.0, "DRAW": 25.0, "AWAY_WIN": 15.0}),
        ml_result={
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": 30.0,
                "DRAW": 30.0,
                "AWAY_WIN": 40.0,
            },
        },
    )

    assert result["ensemble"]["weights"] == {
        "stats": pytest.approx(2 / 3, abs=1e-6),
        "ml": pytest.approx(1 / 3, abs=1e-6),
    }
    assert sum(result["all_probabilities"].values()) == 100.0


def test_learned_weight_metadata_is_applied_and_snapshotted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.prediction import ensemble

    monkeypatch.setattr(
        ensemble.ensemble_weight_manager,
        "get_active_weights",
        lambda: (
            {"stats": 0.1, "ml": 0.2, "market": 0.7},
            {"source": "learned", "artifact_version": "weights-1"},
        ),
    )

    result = ProbabilityEnsembler.apply(
        stats_analysis({"HOME_WIN": 60.0, "DRAW": 20.0, "AWAY_WIN": 20.0}),
        ml_result={
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": 30.0,
                "DRAW": 30.0,
                "AWAY_WIN": 40.0,
            },
        },
        market={
            "fair_probability": {
                "HOME_WIN": 20.0,
                "DRAW": 30.0,
                "AWAY_WIN": 50.0,
            }
        },
    )

    assert result["ensemble"]["weights"] == {
        "stats": 0.1,
        "ml": 0.2,
        "market": 0.7,
    }
    assert result["ensemble"]["weight_metadata"] == {
        "source": "learned",
        "artifact_version": "weights-1",
    }
    assert result["all_probabilities"] == {
        "HOME_WIN": 26.0,
        "DRAW": 29.0,
        "AWAY_WIN": 45.0,
    }


@pytest.mark.parametrize(
    "optional_source",
    [
        {"ready": False},
        {
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": float("nan"),
                "DRAW": 50.0,
                "AWAY_WIN": 50.0,
            },
        },
    ],
    ids=["ml-not-ready", "invalid-ml-probability"],
)
def test_single_valid_source_preserves_stats_result(optional_source: dict) -> None:
    original = stats_analysis({"HOME_WIN": 33.33, "DRAW": 33.34, "AWAY_WIN": 33.33})

    result = ProbabilityEnsembler.apply(original, ml_result=optional_source)

    assert result["all_probabilities"] == original["all_probabilities"]
    assert result["prediction"] == original["prediction"]
    assert result["model"] == original["model"]
    assert result["ensemble"]["applied"] is False


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"HOME_WIN": 0.0, "DRAW": 0.0, "AWAY_WIN": 0.0},
        {"HOME_WIN": -1.0, "DRAW": 50.0, "AWAY_WIN": 51.0},
    ],
    ids=["missing-outcomes", "zero-total", "negative-probability"],
)
def test_invalid_stats_probabilities_are_rejected(
    probabilities: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="valid 1X2 probabilities"):
        ProbabilityEnsembler.apply(stats_analysis(probabilities))
