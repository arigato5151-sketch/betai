import pytest

from app.prediction.input_catalog import AnalysisInputCatalog
from app.prediction.ml.features import FeatureEngine


def test_feature_override_validation_rejects_unknown_and_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="Unsupported feature override"):
        AnalysisInputCatalog.validate_overrides({"unknown_feature": 1})

    with pytest.raises(ValueError, match="must be between"):
        AnalysisInputCatalog.validate_overrides({"fatigue_index": 1.5})


def test_catalog_marks_missing_inputs_and_preserves_manual_overrides() -> None:
    calculated = dict(FeatureEngine.FEATURE_DEFAULTS)
    rows = AnalysisInputCatalog.build(
        calculated,
        {"fatigue_index": 0.4},
        {
            "checks": {
                "kickoff_known": True,
                "home_history_sufficient": True,
                "away_history_sufficient": True,
                "travel_context_available": False,
                "h2h_available": False,
            }
        },
    )
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == FeatureEngine.FEATURE_NAMES
    assert by_name["fatigue_index"]["value"] == 0.4
    assert by_name["fatigue_index"]["availability"] == "manual"
    assert by_name["h2h_home_win_rate"]["availability"] == "missing"
    assert by_name["h2h_home_win_rate"]["missing_reason"]
