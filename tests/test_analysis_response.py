from app.api.endpoints import _build_analysis_response
from app.core.config import settings


def test_insufficient_ml_response_reports_sample_gap() -> None:
    labeled_samples = 17
    response = _build_analysis_response(
        record_id=1,
        home_team="Home",
        away_team="Away",
        analysis={
            "prediction": "DRAW",
            "probability": 34.0,
            "all_probabilities": {
                "HOME_WIN": 33.0,
                "DRAW": 34.0,
                "AWAY_WIN": 33.0,
            },
        },
        value_data={"value_bet": False},
        ml_result={"ready": False},
        insights=[],
        labeled_samples_count=labeled_samples,
    )

    assert response["ml_safety_trigger"] == "INSUFFICIENT_DATA"
    assert response["labeled_samples_count"] == labeled_samples
    assert response["remaining_to_threshold"] == (
        settings.MIN_TRAINING_SAMPLES - labeled_samples
    )
