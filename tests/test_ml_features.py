from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.prediction.ml.features import FeatureEngine


NOW = pd.Timestamp(datetime.now(UTC)).normalize()


def matches_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_date": NOW - timedelta(days=6),
                "points": 0,
                "result": "L",
                "clean_sheet": 0,
                "scoring": 0,
                "goals_for": 0,
                "goals_against": 2,
            },
            {
                "match_date": NOW - timedelta(days=4),
                "points": np.nan,
                "result": "W",
                "clean_sheet": 1,
                "scoring": 1,
                "goals_for": "2",
                "goals_against": 0,
            },
            {
                "match_date": NOW - timedelta(days=2),
                "points": 3,
                "result": "W",
                "clean_sheet": 1,
                "scoring": 1,
                "goals_for": 3,
                "goals_against": "invalid",
            },
        ]
    )


def test_elo_is_chronological_and_ignores_invalid_results() -> None:
    ratings = FeatureEngine.calculate_elo_ratings(
        [
            {
                "created_at": "2026-01-02",
                "home_team_id": 1,
                "away_team_id": 2,
                "actual_result": "INVALID",
            },
            {
                "created_at": "2026-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "actual_result": "HOME_WIN",
            },
        ]
    )

    assert ratings[1] == pytest.approx(1516.0)
    assert ratings[2] == pytest.approx(1484.0)


def test_form_streak_and_goal_features_handle_nan_values() -> None:
    frame = matches_frame()

    assert FeatureEngine.compute_form_ema(frame) == pytest.approx(50.0)
    assert FeatureEngine.compute_streak(frame, "clean_sheet") == 2
    assert FeatureEngine.compute_streak(frame, "scoring") == 2
    assert FeatureEngine.compute_goals_avg(frame, "goals_for", "goals_against") == (
        1.67,
        0.67,
    )


def test_home_advantage_and_h2h_goals_are_bounded() -> None:
    frame = matches_frame()

    assert FeatureEngine.compute_home_advantage_coeff(frame) == 1.05
    assert FeatureEngine.compute_home_advantage_coeff(pd.DataFrame()) == 1.0
    assert FeatureEngine.compute_h2h_goals([]) == (1.2, 1.0)
    assert FeatureEngine.compute_h2h_goals(
        [
            {"home_goals": 2, "away_goals": 1},
            {"home_goals": "invalid", "away_goals": 3},
        ]
    ) == (2.0, 2.0)


def test_inference_feature_vector_has_stable_schema_and_rest_difference() -> None:
    home_frame = matches_frame()
    away_frame = matches_frame().copy()
    away_frame["match_date"] = away_frame["match_date"] - timedelta(days=3)

    features = FeatureEngine.build_inference_features(
        home_stats={"form": 70, "attack": 75, "defense": 68, "xg": 1.7},
        away_stats={"form": 60, "attack": 65, "defense": 62, "xg": 1.3},
        home_matches_df=home_frame,
        away_matches_df=away_frame,
        h2h_rates={"home_win_rate": 0.5, "draw_rate": 0.3, "home_loss_rate": 0.2},
        h2h_matches=[{"home_goals": 2, "away_goals": 1}],
        fixture_date=NOW,
    )

    assert list(features) == FeatureEngine.FEATURE_NAMES
    assert features["rest_days_diff"] == -3.0
    assert features["home_gf_last5"] == 1.67
    assert features["h2h_avg_goals_home"] == 2.0


def test_empty_feature_sources_use_documented_defaults() -> None:
    empty = pd.DataFrame()
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=empty,
        away_matches_df=empty,
        h2h_rates={},
        fixture_date=NOW,
    )

    assert features["home_form"] == 50.0
    assert features["away_xg"] == 1.2
    assert features["home_form_ema"] == 50.0
    assert features["rest_days_diff"] == 0.0
    assert features["home_elo"] == 1500.0


def test_training_uses_same_versioned_snapshot_schema_as_inference() -> None:
    snapshot = {
        **FeatureEngine.FEATURE_DEFAULTS,
        "home_form_ema": 82.5,
        "rest_days_diff": -2.0,
        "home_elo": 1612.0,
        "home_gf_last5": 2.4,
    }
    row = SimpleNamespace(
        feature_snapshot=snapshot,
        feature_schema_version=FeatureEngine.SCHEMA_VERSION,
    )

    features = FeatureEngine.build_training_features(row)

    assert list(features) == FeatureEngine.FEATURE_NAMES
    assert features == snapshot


def test_legacy_training_rows_receive_explicit_full_schema_defaults() -> None:
    row = SimpleNamespace(
        feature_snapshot=None,
        feature_schema_version=None,
        home_form=71.0,
        home_attack=68.0,
        home_defense=64.0,
        home_xg=1.6,
        away_form=59.0,
        away_attack=61.0,
        away_defense=63.0,
        away_xg=1.1,
    )

    features = FeatureEngine.build_training_features(row)

    assert list(features) == FeatureEngine.FEATURE_NAMES
    assert features["home_form"] == 71.0
    assert features["home_form_ema"] == 50.0
    assert features["home_elo"] == 1500.0
    assert features["h2h_avg_goals_away"] == 1.0
