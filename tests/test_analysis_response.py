from unittest.mock import AsyncMock

import pandas as pd
import pytest

from app.api.endpoints import (
    AnalysisRequest,
    _build_analysis_response,
    _compute_analysis,
    _fetch_ml_match_data,
)
from app.core.config import settings
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import HistoricalFeatureContext
from app.prediction.value_calc import ValueCalc


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


@pytest.mark.asyncio
async def test_analysis_prefers_complete_point_in_time_history_over_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    match_dates = pd.date_range("2026-07-01T18:00:00Z", periods=5, freq="3D", tz="UTC")
    home_matches = pd.DataFrame(
        {
            "match_date": match_dates,
            "points": [3.0, 3.0, 1.0, 3.0, 3.0],
            "result": ["W", "W", "D", "W", "W"],
            "clean_sheet": [1, 0, 0, 1, 1],
            "scoring": [1, 1, 1, 1, 1],
            "goals_for": [2, 3, 1, 2, 1],
            "goals_against": [0, 1, 1, 0, 0],
        }
    )
    away_matches = home_matches.assign(
        points=[0.0, 1.0, 0.0, 0.0, 1.0],
        result=["L", "D", "L", "L", "D"],
    )
    context = HistoricalFeatureContext(
        home_elo=1580.0,
        away_elo=1460.0,
        h2h_rates={
            "home_win_rate": 0.6,
            "draw_rate": 0.2,
            "home_loss_rate": 0.2,
        },
        h2h_matches=[{"home_goals": 2, "away_goals": 0}],
        home_matches_df=home_matches,
        away_matches_df=away_matches,
        home_previous_starting_xi=list(range(1, 12)),
        away_previous_starting_xi=list(range(20, 31)),
    )
    monkeypatch.setattr(endpoints, "_get_historical_feature_context", lambda _: context)
    monkeypatch.setattr(endpoints.ml_pipeline, "is_ready", False)
    home_api = AsyncMock(side_effect=AssertionError("form API should not be called"))
    h2h_api = AsyncMock(side_effect=AssertionError("H2H API should not be called"))
    availability_api = AsyncMock(
        return_value={
            "home_missing_players": 2,
            "away_missing_players": 1,
            "home_questionable_players": 0,
            "away_questionable_players": 1,
            "availability_report_present": 1,
        }
    )
    lineups_api = AsyncMock(
        return_value={
            "home_starting_xi": list(range(1, 10)) + [12, 13],
            "away_starting_xi": list(range(20, 31)),
            "source": "api_football_lineups",
        }
    )
    monkeypatch.setattr(endpoints.football_api, "get_team_last_matches_df", home_api)
    monkeypatch.setattr(endpoints.football_api, "get_h2h", h2h_api)
    monkeypatch.setattr(
        endpoints.football_api, "get_fixture_availability", availability_api
    )
    monkeypatch.setattr(endpoints.football_api, "get_fixture_lineups", lineups_api)
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_team_id=1,
        away_team_id=2,
        fixture_id=999,
        league_id=203,
        season=2026,
        kickoff="2026-07-20T18:00:00Z",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.1,
    )

    computed = await _compute_analysis(payload)

    assert computed["feature_vector"]["home_elo"] == 1580.0
    assert computed["feature_vector"]["away_elo"] == 1460.0
    assert computed["feature_vector"]["home_gf_last5"] == 1.8
    assert computed["feature_vector"]["h2h_avg_goals_home"] == 2.0
    assert computed["feature_vector"]["home_missing_players"] == 2.0
    assert computed["feature_vector"]["away_questionable_players"] == 1.0
    assert computed["feature_vector"]["availability_report_present"] == 1.0
    assert computed["feature_vector"]["home_lineup_continuity"] == pytest.approx(
        9 / 11, abs=1e-4
    )
    assert computed["feature_vector"]["away_lineup_continuity"] == 1.0
    home_api.assert_not_awaited()
    h2h_api.assert_not_awaited()
    availability_api.assert_awaited_once_with(999, 1, 2)
    lineups_api.assert_awaited_once_with(999, 1, 2)


@pytest.mark.asyncio
async def test_stale_point_in_time_form_uses_api_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    stale_frame = pd.DataFrame(
        {
            "match_date": pd.date_range("2026-01-01", periods=5, freq="7D", tz="UTC"),
            "points": [3.0] * 5,
        }
    )
    api_frame = pd.DataFrame(
        {
            "match_date": [pd.Timestamp("2026-07-17T18:00:00Z")],
            "points": [1.0],
        }
    )
    context = HistoricalFeatureContext(
        h2h_rates={
            "home_win_rate": 0.4,
            "draw_rate": 0.3,
            "home_loss_rate": 0.3,
        },
        home_matches_df=stale_frame,
        away_matches_df=stale_frame,
    )
    form_api = AsyncMock(return_value=api_frame)
    h2h_api = AsyncMock(side_effect=AssertionError("local H2H should be used"))
    monkeypatch.setattr(endpoints.football_api, "get_team_last_matches_df", form_api)
    monkeypatch.setattr(endpoints.football_api, "get_h2h", h2h_api)
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

    home_matches, away_matches, h2h_rates, availability, lineups = (
        await _fetch_ml_match_data(payload, context)
    )

    assert home_matches is api_frame
    assert away_matches is api_frame
    assert h2h_rates is context.h2h_rates
    assert availability is None
    assert lineups is None
    assert form_api.await_count == 2
    h2h_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_value_evaluation_uses_ensemble_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    monkeypatch.setattr(settings, "ENSEMBLE_STATS_WEIGHT", 0.4)
    monkeypatch.setattr(settings, "ENSEMBLE_ML_WEIGHT", 0.2)
    monkeypatch.setattr(settings, "ENSEMBLE_MARKET_WEIGHT", 0.4)
    monkeypatch.setattr(
        endpoints.StatsEngine,
        "analyze_match",
        lambda *_args, **_kwargs: {
            "model": "stats_test",
            "prediction": "DRAW",
            "probability": 40.0,
            "all_probabilities": {
                "HOME_WIN": 20.0,
                "DRAW": 40.0,
                "AWAY_WIN": 40.0,
            },
            "confidence_gap": 0.0,
            "confidence_tier": "DUSUK",
        },
    )
    monkeypatch.setattr(endpoints.ml_pipeline, "is_ready", True)
    monkeypatch.setattr(
        endpoints.ml_pipeline,
        "predict_match",
        lambda _features: {
            "ready": True,
            "prediction": "HOME_WIN",
            "probability": 80.0,
            "all_probabilities": {
                "HOME_WIN": 80.0,
                "DRAW": 10.0,
                "AWAY_WIN": 10.0,
            },
        },
    )
    monkeypatch.setattr(
        endpoints.ExplainabilityService,
        "generate_explanation",
        lambda *_args, **_kwargs: [],
    )
    market = ValueCalc.devig_1x2(2.0, 2.0, 2.0)
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.0,
        market_1x2=market,
    )

    computed = await _compute_analysis(payload)

    assert computed["analysis"]["prediction"] == "HOME_WIN"
    assert computed["analysis"]["all_probabilities"]["HOME_WIN"] == 37.34
    assert computed["value_data"]["edge"] == -25.32
    assert computed["analysis"]["ensemble"]["applied"] is True
