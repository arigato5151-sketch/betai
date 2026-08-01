from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.endpoints import (
    AnalysisRequest,
    _apply_external_travel_fallback,
    _assess_ml_safety,
    _build_analysis_response,
    _apply_external_elo_fallback,
    _compute_analysis,
    _derive_reference_lineup,
    _fetch_ml_match_data,
    _fetch_player_rating_data,
    _select_reference_lineup,
)
from app.core.config import settings
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import HistoricalFeatureContext
from app.prediction.value_calc import ValueCalc
from app.providers.base import ExternalDataPoint


@pytest.mark.asyncio
async def test_external_elo_fallback_updates_missing_historical_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    captured_at = datetime(2026, 7, 30, 10, tzinfo=UTC)

    async def external_elo(**kwargs: object) -> ExternalDataPoint:
        value = 1742.0 if kwargs["canonical_team_id"] == 611 else 1710.0
        return ExternalDataPoint(
            value=value,
            source="clubelo",
            captured_at=captured_at,
            expires_at=captured_at + timedelta(hours=24),
            confidence=0.8,
        )

    monkeypatch.setattr(
        endpoints.external_feature_service,
        "get_team_elo",
        external_elo,
    )
    payload = AnalysisRequest(
        home_team="Fenerbahçe",
        away_team="Galatasaray",
        home_team_id=611,
        away_team_id=645,
        league_id=203,
        kickoff=datetime(2026, 7, 30, 18, tzinfo=UTC),
        home_stats={"form": 70, "attack": 70, "defense": 70, "xg": 1.5},
        away_stats={"form": 70, "attack": 70, "defense": 70, "xg": 1.5},
        odd=2.0,
    )

    context = await _apply_external_elo_fallback(
        payload,
        HistoricalFeatureContext(),
    )

    assert context.home_elo == 1742.0
    assert context.away_elo == 1710.0
    assert context.home_elo_available is True
    assert context.feature_provenance["home_elo"] == {
        "source": "clubelo",
        "captured_at": captured_at.isoformat(),
        "confidence": 0.8,
        "is_fallback": True,
    }


@pytest.mark.asyncio
async def test_external_travel_fallback_updates_missing_historical_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    captured_at = datetime(2026, 7, 30, 10, tzinfo=UTC)
    get_distance = AsyncMock(
        return_value=ExternalDataPoint(
            value=1346.74,
            source="geonames_city",
            captured_at=captured_at,
            confidence=0.75,
            is_fallback=True,
        )
    )
    monkeypatch.setattr(
        endpoints.travel_context_service,
        "get_away_travel_distance",
        get_distance,
    )
    payload = AnalysisRequest(
        home_team="Ajax",
        away_team="Vojvodina",
        home_team_id=194,
        away_team_id=702,
        league_id=2,
        kickoff=datetime(2026, 7, 30, 18, tzinfo=UTC),
        home_stats={"form": 70, "attack": 70, "defense": 70, "xg": 1.5},
        away_stats={"form": 70, "attack": 70, "defense": 70, "xg": 1.5},
        odd=2.0,
    )

    context = await _apply_external_travel_fallback(
        payload,
        HistoricalFeatureContext(),
    )

    assert context.away_travel_distance_km == pytest.approx(1346.74)
    assert context.travel_context_available is True
    assert context.travel_provenance == {
        "source": "geonames_city",
        "captured_at": captured_at.isoformat(),
        "confidence": 0.75,
        "is_fallback": True,
    }
    get_distance.assert_awaited_once_with(
        home_team_id=194,
        away_team_id=702,
        home_team_name="Ajax",
        away_team_name="Vojvodina",
        client=endpoints.football_api,
    )


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
        data_quality={"score": 70.0},
    )
    assert response["ml_safety_trigger"] == "INSUFFICIENT_DATA"
    assert response["labeled_samples_count"] == labeled_samples
    assert response["remaining_to_threshold"] == (
        settings.MIN_TRAINING_SAMPLES - labeled_samples
    )


def test_ml_safety_uses_market_disagreement_for_upset_labels() -> None:
    assessment = _assess_ml_safety(
        ml_result={
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": 25.0,
                "DRAW": 20.0,
                "AWAY_WIN": 55.0,
            },
        },
        analysis={
            "all_probabilities": {
                "HOME_WIN": 30.0,
                "DRAW": 20.0,
                "AWAY_WIN": 50.0,
            },
            "ensemble": {
                "components": {
                    "stats": {
                        "HOME_WIN": 30.0,
                        "DRAW": 20.0,
                        "AWAY_WIN": 50.0,
                    },
                    "market": {
                        "HOME_WIN": 60.0,
                        "DRAW": 25.0,
                        "AWAY_WIN": 15.0,
                    },
                }
            },
        },
        data_quality={"score": 80.0},
    )

    assert assessment["trigger"] == "UPSET_CANDIDATE"
    assert assessment["market_favorite"] == "HOME_WIN"
    assert assessment["ml_prediction"] == "AWAY_WIN"


def test_ml_safety_does_not_call_away_favorite_an_upset() -> None:
    assessment = _assess_ml_safety(
        ml_result={
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": 20.0,
                "DRAW": 20.0,
                "AWAY_WIN": 60.0,
            },
        },
        analysis={
            "all_probabilities": {
                "HOME_WIN": 20.0,
                "DRAW": 20.0,
                "AWAY_WIN": 60.0,
            },
            "ensemble": {
                "components": {
                    "stats": {
                        "HOME_WIN": 20.0,
                        "DRAW": 20.0,
                        "AWAY_WIN": 60.0,
                    },
                    "market": {
                        "HOME_WIN": 25.0,
                        "DRAW": 25.0,
                        "AWAY_WIN": 50.0,
                    },
                }
            },
        },
        data_quality={"score": 80.0},
    )

    assert assessment["trigger"] == "HIGH_CONFIDENCE"


def test_ml_safety_marks_market_disagreement_risky_when_models_disagree() -> None:
    assessment = _assess_ml_safety(
        ml_result={
            "ready": True,
            "all_probabilities": {
                "HOME_WIN": 30.0,
                "DRAW": 22.0,
                "AWAY_WIN": 48.0,
            },
        },
        analysis={
            "all_probabilities": {
                "HOME_WIN": 46.0,
                "DRAW": 24.0,
                "AWAY_WIN": 30.0,
            },
            "ensemble": {
                "components": {
                    "stats": {
                        "HOME_WIN": 50.0,
                        "DRAW": 25.0,
                        "AWAY_WIN": 25.0,
                    },
                    "market": {
                        "HOME_WIN": 55.0,
                        "DRAW": 25.0,
                        "AWAY_WIN": 20.0,
                    },
                }
            },
        },
        data_quality={"score": 75.0},
    )

    assert assessment["trigger"] == "RISKY_UPSET"
    assert assessment["model_agreement"] is False


def test_prefill_payload_carries_automatic_odds_snapshots() -> None:
    from app.api.endpoints import _build_payload_from_prefill

    prefill = {
        "fixture": {
            "fixture_id": 10,
            "home_team_id": 101,
            "away_team_id": 202,
            "league_id": 203,
            "season": 2030,
            "kickoff": "2030-07-30T18:00:00+00:00",
        },
        "home_team": "Home",
        "away_team": "Away",
        "home_stats": {"form": 70, "attack": 70, "defense": 70, "xg": 1.5},
        "away_stats": {"form": 65, "attack": 65, "defense": 65, "xg": 1.2},
        "odd": 2.2,
        "market_1x2": {"raw_odds": {"HOME_WIN": 2.2}},
        "opening_odds_1x2": {
            "HOME_WIN": 2.4,
            "DRAW": 3.3,
            "AWAY_WIN": 3.1,
        },
        "current_odds_1x2": {
            "HOME_WIN": 2.2,
            "DRAW": 3.4,
            "AWAY_WIN": 3.3,
        },
        "opening_odds_at": "2030-07-29T10:00:00+00:00",
        "current_odds_at": "2030-07-30T10:00:00+00:00",
    }

    payload = _build_payload_from_prefill(prefill)

    assert payload.opening_odds_1x2 is not None
    assert payload.current_odds_1x2 is not None
    assert payload.opening_odds_1x2.home_win == 2.4
    assert payload.current_odds_1x2.home_win == 2.2


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
    captured_stats_kwargs: dict[str, object] = {}
    original_analyze_match = endpoints.StatsEngine.analyze_match

    def capture_stats_history(*args, **kwargs):
        captured_stats_kwargs.update(kwargs)
        return original_analyze_match(*args, **kwargs)

    monkeypatch.setattr(
        endpoints.StatsEngine,
        "analyze_match",
        capture_stats_history,
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
    assert computed["feature_vector"]["league_id"] == 0.0
    assert computed["feature_vector"]["home_team_id"] == 1.0
    assert computed["feature_vector"]["away_team_id"] == 2.0
    assert captured_stats_kwargs["home_match_history"] is home_matches
    assert captured_stats_kwargs["away_match_history"] is away_matches
    assert captured_stats_kwargs["as_of"] == payload.kickoff
    assert computed["ml_result"] == {"ready": False}
    assert computed["data_quality"]["analysis_outputs"] == {
        "expected_goals": computed["analysis"]["expected_goals"],
        "expected_score": computed["analysis"]["expected_score"],
        "score_band": computed["analysis"]["score_band"],
        "secondary_markets": computed["analysis"]["secondary_markets"],
        "match_profile": computed["analysis"]["match_profile"],
    }


@pytest.mark.asyncio
async def test_analysis_applies_validated_manual_feature_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    monkeypatch.setattr(endpoints.ml_pipeline, "is_ready", False)
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.0,
        feature_overrides={
            "fatigue_index": 0.4,
            "home_elo": 1625,
        },
    )

    computed = await _compute_analysis(payload)

    assert computed["calculated_feature_vector"]["fatigue_index"] == 0.0
    assert computed["feature_vector"]["fatigue_index"] == 0.4
    assert computed["feature_vector"]["home_elo"] == 1625.0
    assert computed["data_quality"]["manual_feature_overrides"] == [
        "fatigue_index",
        "home_elo",
    ]
    assert computed["data_quality"]["manual_feature_override_count"] == 2


@pytest.mark.asyncio
async def test_analysis_maps_validated_odds_snapshots_to_ml_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    monkeypatch.setattr(endpoints.ml_pipeline, "is_ready", False)
    kickoff = datetime(2030, 7, 20, 18, tzinfo=UTC)
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.0,
        kickoff=kickoff,
        opening_odds_1x2={
            "HOME_WIN": 2.5,
            "DRAW": 3.0,
            "AWAY_WIN": 4.0,
        },
        current_odds_1x2={
            "HOME_WIN": 2.0,
            "DRAW": 3.3,
            "AWAY_WIN": 3.6,
        },
        opening_odds_at=kickoff - timedelta(days=3),
        current_odds_at=kickoff - timedelta(hours=1),
    )

    computed = await _compute_analysis(payload)

    assert computed["feature_vector"]["odds_movement_home"] == -20.0
    assert computed["feature_vector"]["odds_movement_draw"] == 10.0
    assert computed["feature_vector"]["odds_movement_away"] == -10.0
    assert computed["data_quality"]["odds_snapshot"] == {
        "movement_features_used": True,
        "opening_captured_at": (kickoff - timedelta(days=3)).isoformat(),
        "current_captured_at": (kickoff - timedelta(hours=1)).isoformat(),
    }
    assert computed["data_quality"]["feature_provenance"]["odds_movement_home"] == {
        "source": "api_football_odds",
        "captured_at": (kickoff - timedelta(hours=1)).isoformat(),
        "confidence": settings.ODDS_SNAPSHOT_CONFIDENCE,
        "is_fallback": False,
    }


def test_analysis_rejects_incomplete_or_invalid_odds_snapshots() -> None:
    base = {
        "home_team": "Home",
        "away_team": "Away",
        "home_stats": {"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        "away_stats": {"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        "odd": 2.0,
    }

    with pytest.raises(ValidationError):
        AnalysisRequest(
            **base,
            opening_odds_1x2={"HOME_WIN": 2.5, "AWAY_WIN": 4.0},
            opening_odds_at="2030-07-17T18:00:00Z",
            kickoff="2030-07-20T18:00:00Z",
        )
    with pytest.raises(ValidationError):
        AnalysisRequest(
            **base,
            current_odds_1x2={
                "HOME_WIN": 1.0,
                "DRAW": 3.0,
                "AWAY_WIN": 4.0,
            },
            current_odds_at="2030-07-20T17:00:00Z",
            kickoff="2030-07-20T18:00:00Z",
        )


def test_analysis_rejects_unversioned_or_post_kickoff_odds_snapshots() -> None:
    base = {
        "home_team": "Home",
        "away_team": "Away",
        "home_stats": {"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        "away_stats": {"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        "odd": 2.0,
        "kickoff": "2030-07-20T18:00:00Z",
    }
    odds = {"HOME_WIN": 2.5, "DRAW": 3.0, "AWAY_WIN": 4.0}

    with pytest.raises(ValidationError, match="provided together"):
        AnalysisRequest(**base, current_odds_1x2=odds)
    with pytest.raises(ValidationError, match="before kickoff"):
        AnalysisRequest(
            **base,
            current_odds_1x2=odds,
            current_odds_at="2030-07-20T18:00:00Z",
        )
    with pytest.raises(ValidationError, match="cannot follow"):
        AnalysisRequest(
            **base,
            opening_odds_1x2=odds,
            opening_odds_at="2030-07-20T17:30:00Z",
            current_odds_1x2=odds,
            current_odds_at="2030-07-20T17:00:00Z",
        )


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
        home_schedule_df=home_matches,
        away_schedule_df=away_matches,
        home_player_ratings={
            player_id: {
                "rating": 9.5 if player_id == 11 else 7.0,
                "minutes": 1000.0,
            }
            for player_id in range(1, 14)
        },
        away_player_ratings={
            player_id: {"rating": 7.0, "minutes": 1000.0} for player_id in range(20, 33)
        },
        away_travel_distance_km=1200.0,
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
            "home_unavailable_players": [
                {
                    "player_id": 11,
                    "status": "missing",
                    "name": "Critical Starter",
                    "reason": "injury",
                }
            ],
            "away_unavailable_players": [
                {
                    "player_id": 30,
                    "status": "questionable",
                    "name": "Away Player",
                    "reason": "fitness",
                }
            ],
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
    assert computed["feature_vector"]["home_team_strength_ratio"] < 1.0
    # A confirmed starter supersedes the stale pre-match questionable status.
    assert computed["feature_vector"]["away_team_strength_ratio"] == 1.0
    assert computed["feature_vector"]["fatigue_index"] > 0.0
    assert computed["analysis"]["player_impact"]["home"][
        "critical_missing_player_ids"
    ] == [11]
    assert computed["data_quality"]["away_travel_distance_km"] == 1200.0
    home_api.assert_not_awaited()
    h2h_api.assert_not_awaited()
    availability_api.assert_awaited_once_with(999, 1, 2)
    lineups_api.assert_awaited_once_with(999, 1, 2)


@pytest.mark.asyncio
async def test_upcoming_match_uses_season_ratings_to_derive_typical_xi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    home_ratings = {
        player_id: {
            "rating": 6.0 + player_id / 100,
            "minutes": float(2000 - player_id),
            "appearances": 20.0,
        }
        for player_id in range(1, 14)
    }
    away_ratings = {
        player_id: {
            "rating": 7.0,
            "minutes": float(3000 - player_id),
            "appearances": 25.0,
        }
        for player_id in range(20, 33)
    }
    ratings_api = AsyncMock(side_effect=[home_ratings, away_ratings])
    monkeypatch.setattr(
        endpoints.football_api,
        "get_team_player_ratings",
        ratings_api,
    )
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_team_id=1,
        away_team_id=2,
        league_id=203,
        season=2098,
        kickoff="2099-07-20T18:00:00Z",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.1,
    )

    home, away = await _fetch_player_rating_data(
        payload,
        HistoricalFeatureContext(),
    )

    assert home == home_ratings
    assert away == away_ratings
    assert _derive_reference_lineup(home) == list(range(1, 12))
    assert ratings_api.await_count == 2


@pytest.mark.asyncio
async def test_current_season_roster_replaces_incomplete_stale_local_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import endpoints

    stale_local = {
        player_id: {
            "rating": 9.0,
            "minutes": 900.0,
            "appearances": 10.0,
        }
        for player_id in range(1, 8)
    }
    current_home = {
        player_id: {
            "rating": 7.0,
            "minutes": float(3000 - player_id),
            "appearances": 20.0,
        }
        for player_id in range(20, 33)
    }
    current_away = {
        player_id: {
            "rating": 7.0,
            "minutes": float(3000 - player_id),
            "appearances": 20.0,
        }
        for player_id in range(40, 53)
    }
    ratings_api = AsyncMock(side_effect=[current_home, current_away])
    monkeypatch.setattr(
        endpoints.football_api,
        "get_team_player_ratings",
        ratings_api,
    )
    payload = AnalysisRequest(
        home_team="Home",
        away_team="Away",
        home_team_id=1,
        away_team_id=2,
        league_id=203,
        season=2098,
        kickoff="2099-07-20T18:00:00Z",
        home_stats={"form": 70, "attack": 72, "defense": 68, "xg": 1.7},
        away_stats={"form": 62, "attack": 65, "defense": 64, "xg": 1.3},
        odd=2.1,
    )
    historical = HistoricalFeatureContext(
        home_previous_starting_xi=list(range(1, 12)),
        away_previous_starting_xi=list(range(40, 51)),
        home_player_ratings=stale_local,
    )

    home, away = await _fetch_player_rating_data(payload, historical)

    assert home == current_home
    assert away == current_away
    assert not set(stale_local).intersection(home)
    assert _select_reference_lineup(
        historical.home_previous_starting_xi,
        home,
    ) == list(range(20, 31))
    assert ratings_api.await_count == 2


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

    (
        home_matches,
        away_matches,
        h2h_rates,
        availability,
        lineups,
    ) = await _fetch_ml_match_data(payload, context)

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
