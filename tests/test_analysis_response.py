from unittest.mock import AsyncMock

import pandas as pd
import pytest

from app.api.endpoints import (
    AnalysisRequest,
    _build_analysis_response,
    _compute_analysis,
)
from app.core.config import settings
from app.prediction.ml.features import FeatureEngine


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


@pytest.mark.asyncio
async def test_analysis_collects_feature_snapshot_before_first_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    home_matches = pd.DataFrame(
        [
            {
                "match_date": pd.Timestamp("2026-07-17T18:00:00Z"),
                "points": 3,
                "result": "W",
                "clean_sheet": 1,
                "scoring": 1,
                "goals_for": 2,
                "goals_against": 0,
            }
        ]
    )
    away_matches = home_matches.assign(match_date=pd.Timestamp("2026-07-16T18:00:00Z"))
    monkeypatch.setattr(endpoints.ml_pipeline, "is_ready", False)
    monkeypatch.setattr(
        endpoints.football_api,
        "get_team_last_matches_df",
        AsyncMock(side_effect=[home_matches, away_matches]),
    )
    monkeypatch.setattr(
        endpoints.football_api,
        "get_h2h",
        AsyncMock(
            return_value={
                "home_win_rate": 0.6,
                "draw_rate": 0.2,
                "home_loss_rate": 0.2,
            }
        ),
    )
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_team_id=1,
        away_team_id=2,
        kickoff="2026-07-20T18:00:00Z",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.1,
    )

    computed = await _compute_analysis(payload)

    assert list(computed["feature_vector"]) == FeatureEngine.FEATURE_NAMES
    assert computed["feature_vector"]["home_form_ema"] == 100.0
    assert computed["feature_vector"]["rest_days_diff"] == -1.0
    assert computed["feature_vector"]["h2h_home_win_rate"] == 0.6
    assert computed["ml_result"] == {"ready": False}
