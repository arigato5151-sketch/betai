from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.config import settings
from app.prediction.ml.features import FeatureEngine
from app.prediction.player_impact import PlayerImpactCalculator

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


def test_elo_home_advantage_reduces_expected_home_win_gain() -> None:
    ratings = FeatureEngine.calculate_elo_ratings(
        [
            {
                "created_at": "2026-01-01",
                "season": 2026,
                "home_team_id": 1,
                "away_team_id": 2,
                "actual_result": "HOME_WIN",
            }
        ],
        home_advantage_points=65.0,
    )

    assert 1500.0 < ratings[1] < 1516.0
    assert ratings[1] + ratings[2] == pytest.approx(3000.0)


def test_elo_regresses_existing_ratings_at_season_boundary() -> None:
    matches = [
        {
            "created_at": "2025-05-01",
            "season": 2024,
            "home_team_id": 1,
            "away_team_id": 2,
            "actual_result": "HOME_WIN",
        },
        {
            "created_at": "2025-08-01",
            "season": 2025,
            "home_team_id": 3,
            "away_team_id": 4,
            "actual_result": "DRAW",
        },
    ]

    without_regression = FeatureEngine.calculate_elo_ratings(matches)
    with_regression = FeatureEngine.calculate_elo_ratings(
        matches, season_regression=0.25
    )

    assert without_regression[1] == pytest.approx(1516.0)
    assert with_regression[1] == pytest.approx(1512.0)
    assert with_regression[2] == pytest.approx(1488.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"k_factor": 0}, "k_factor", id="non-positive-k-factor"),
        pytest.param(
            {"home_advantage_points": -1},
            "home_advantage_points",
            id="negative-home-advantage",
        ),
        pytest.param(
            {"season_regression": 1.1},
            "season_regression",
            id="invalid-season-regression",
        ),
    ],
)
def test_elo_rejects_invalid_configuration(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FeatureEngine.calculate_elo_ratings([], **kwargs)


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


@pytest.mark.parametrize(
    ("opening", "current", "expected"),
    [
        pytest.param(2.5, 2.0, -20.0, id="dropping-odds"),
        pytest.param(3.0, 3.3, 10.0, id="drifting-odds"),
        pytest.param(2.2, 2.2, 0.0, id="unchanged-odds"),
        pytest.param(None, 2.0, 0.0, id="missing-opening"),
        pytest.param(2.0, None, 0.0, id="missing-current"),
        pytest.param("invalid", 2.0, 0.0, id="non-numeric"),
        pytest.param(float("nan"), 2.0, 0.0, id="non-finite"),
        pytest.param(True, 2.0, 0.0, id="boolean"),
        pytest.param(1.0, 2.0, 0.0, id="invalid-decimal-odd"),
    ],
)
def test_odds_movement_is_percentage_change_with_safe_fallback(
    opening: object,
    current: object,
    expected: float,
) -> None:
    assert FeatureEngine.compute_odds_movement(opening, current) == expected


def test_inference_feature_vector_has_stable_schema_and_rest_difference() -> None:
    home_frame = matches_frame()
    away_frame = matches_frame().copy()
    away_frame["match_date"] = away_frame["match_date"] - timedelta(days=3)
    home_impact = PlayerImpactCalculator.assess(
        {player_id: (9.0 if player_id == 1 else 6.0) for player_id in range(1, 12)},
        reference_lineup=list(range(1, 12)),
        missing_player_ids=[1],
    )

    features = FeatureEngine.build_inference_features(
        home_stats={"form": 70, "attack": 75, "defense": 68, "xg": 1.7},
        away_stats={"form": 60, "attack": 65, "defense": 62, "xg": 1.3},
        home_matches_df=home_frame,
        away_matches_df=away_frame,
        h2h_rates={"home_win_rate": 0.5, "draw_rate": 0.3, "home_loss_rate": 0.2},
        h2h_matches=[{"home_goals": 2, "away_goals": 1}],
        availability={
            "home_missing_players": 2,
            "away_missing_players": 1,
            "home_questionable_players": 1,
            "away_questionable_players": 0,
            "availability_report_present": 1,
        },
        lineup_context={
            "home_starting_xi": list(range(1, 10)) + [12, 13],
            "away_starting_xi": list(range(20, 31)),
            "home_previous_starting_xi": list(range(1, 12)),
            "away_previous_starting_xi": list(range(20, 31)),
        },
        fixture_date=NOW,
        opening_odds={"HOME_WIN": 2.5, "DRAW": 3.0, "AWAY_WIN": 4.0},
        current_odds={"HOME_WIN": 2.0, "DRAW": 3.3, "AWAY_WIN": 3.6},
        home_player_impact=home_impact,
    )

    assert list(features) == FeatureEngine.FEATURE_NAMES
    assert features["rest_days_diff"] == -3.0
    assert -1.0 <= features["fatigue_index"] <= 1.0
    assert features["home_gf_last5"] == 1.67
    assert features["h2h_avg_goals_home"] == 2.0
    assert features["home_missing_players"] == 2.0
    assert features["away_missing_players"] == 1.0
    assert features["availability_report_present"] == 1.0
    assert features["home_lineup_confirmed"] == 1.0
    assert features["home_lineup_reference_available"] == 1.0
    assert features["home_lineup_continuity"] == pytest.approx(9 / 11, abs=1e-4)
    assert features["away_lineup_continuity"] == 1.0
    assert features["odds_movement_home"] == -20.0
    assert features["odds_movement_draw"] == 10.0
    assert features["odds_movement_away"] == -10.0
    assert features["home_team_strength_ratio"] < 1.0
    assert features["away_team_strength_ratio"] == 1.0


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
    assert features["fatigue_index"] == 0.0
    assert features["home_team_strength_ratio"] == 1.0
    assert features["away_team_strength_ratio"] == 1.0
    assert features["home_elo"] == 1500.0
    assert features["odds_movement_home"] == 0.0
    assert features["odds_movement_draw"] == 0.0
    assert features["odds_movement_away"] == 0.0


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


@pytest.mark.parametrize(
    "schema_version",
    [
        "ml_features_v1",
        "ml_features_v2",
        "ml_features_v3",
        "ml_features_v4",
        "ml_features_v5",
        "ml_features_v6",
        "ml_features_v7",
    ],
)
def test_older_snapshots_are_forward_compatible_with_new_defaults(
    schema_version: str,
) -> None:
    excluded_names = {name for name in FeatureEngine.FEATURE_NAMES if "lineup" in name}
    excluded_names.update(
        {
            "fatigue_index",
            "home_team_strength_ratio",
            "away_team_strength_ratio",
        }
    )
    if schema_version == "ml_features_v1":
        excluded_names.update(
            name for name in FeatureEngine.FEATURE_NAMES if "players" in name
        )
        excluded_names.add("availability_report_present")
    snapshot = {
        name: value
        for name, value in FeatureEngine.FEATURE_DEFAULTS.items()
        if name not in excluded_names
    }
    snapshot["home_form_ema"] = 77.0
    row = SimpleNamespace(
        feature_snapshot=snapshot,
        feature_schema_version=schema_version,
    )

    features = FeatureEngine.build_training_features(row)

    assert features["home_form_ema"] == 77.0
    assert features["home_missing_players"] == 0.0
    assert features["away_questionable_players"] == 0.0
    assert features["home_lineup_continuity"] == 0.0
    assert features["odds_movement_home"] == 0.0
    assert features["odds_movement_draw"] == 0.0
    assert features["odds_movement_away"] == 0.0
    assert features["fatigue_index"] == 0.0
    assert features["home_team_strength_ratio"] == 1.0
    assert features["away_team_strength_ratio"] == 1.0
    assert features["league_id"] == 0.0
    assert features["home_team_id"] == 0.0
    assert features["away_team_id"] == 0.0
    assert list(features) == FeatureEngine.FEATURE_NAMES


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [
        pytest.param(
            list(range(1, 12)),
            list(range(1, 12)),
            (1.0, 1.0, 1.0),
            id="identical-confirmed-lineups",
        ),
        pytest.param(
            list(range(1, 11)),
            list(range(1, 12)),
            (0.0, 1.0, 0.0),
            id="partial-current-lineup",
        ),
        pytest.param(None, None, (0.0, 0.0, 0.0), id="lineups-unavailable"),
    ],
)
def test_lineup_continuity_requires_complete_elevens(
    current: object, previous: object, expected: tuple[float, float, float]
) -> None:
    assert FeatureEngine.compute_lineup_continuity(current, previous) == expected


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
    assert features["odds_movement_home"] == 0.0
    assert features["fatigue_index"] == 0.0


def test_team_fatigue_uses_lookback_boundary_and_excludes_non_prior_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_LOOKBACK_DAYS", 14)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_REFERENCE_COUNT", 4)
    monkeypatch.setattr(settings, "FATIGUE_IDEAL_REST_DAYS", 7.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_REFERENCE_KM", 1000.0)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 1.0)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 0.0)
    schedule = pd.DataFrame(
        {
            "match_date": [
                NOW - timedelta(days=14),
                NOW,
                NOW + timedelta(days=1),
                "not-a-date",
            ]
        }
    )

    fatigue = FeatureEngine.compute_team_fatigue(schedule, NOW)

    assert fatigue == 0.25


def test_team_fatigue_normalizes_naive_and_aware_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_LOOKBACK_DAYS", 14)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_REFERENCE_COUNT", 4)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 1.0)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 0.0)
    naive_fixture_date = NOW.tz_localize(None)
    schedule = pd.DataFrame(
        {
            "match_date": [
                naive_fixture_date - timedelta(days=1),
                NOW - timedelta(days=2),
            ]
        }
    )

    fatigue = FeatureEngine.compute_team_fatigue(schedule, naive_fixture_date)

    assert fatigue == 0.5


def test_fatigue_index_combines_congestion_rest_and_away_travel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_LOOKBACK_DAYS", 14)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_REFERENCE_COUNT", 4)
    monkeypatch.setattr(settings, "FATIGUE_IDEAL_REST_DAYS", 7.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_REFERENCE_KM", 1000.0)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 0.4)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.35)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 0.25)
    home_schedule = pd.DataFrame(
        {"match_date": [NOW - timedelta(days=14), NOW - timedelta(days=8)]}
    )
    away_schedule = pd.DataFrame(
        {
            "match_date": [
                NOW - timedelta(days=13),
                NOW - timedelta(days=6),
                NOW - timedelta(days=3),
                NOW - timedelta(days=1),
            ]
        }
    )

    fatigue_index = FeatureEngine.compute_fatigue_index(
        home_schedule,
        away_schedule,
        NOW,
        away_travel_distance_km=1000.0,
    )

    assert fatigue_index == 0.75


def test_weather_features_are_normalized_with_safe_missing_defaults() -> None:
    assert FeatureEngine.weather_features(None) == {
        "weather_temperature_c": 15.0,
        "weather_precipitation_mm": 0.0,
        "weather_wind_speed_kmh": 0.0,
        "weather_available": 0.0,
    }
    assert FeatureEngine.weather_features(
        {
            "weather_temperature_c": 7.5,
            "weather_precipitation_mm": 2.0,
            "weather_wind_speed_kmh": 24.0,
        }
    ) == {
        "weather_temperature_c": 7.5,
        "weather_precipitation_mm": 2.0,
        "weather_wind_speed_kmh": 24.0,
        "weather_available": 1.0,
    }
    assert (
        FeatureEngine.weather_features(
            {
                "weather_temperature_c": 7.5,
                "weather_precipitation_mm": -1.0,
                "weather_wind_speed_kmh": 24.0,
            }
        )["weather_available"]
        == 0.0
    )


@pytest.mark.parametrize(
    "travel_distance",
    [None, -10, float("nan"), float("inf"), True, "invalid"],
)
def test_fatigue_handles_malformed_inputs_with_neutral_fallback(
    monkeypatch: pytest.MonkeyPatch,
    travel_distance: object,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 1.0)

    assert (
        FeatureEngine.compute_fatigue_index(
            pd.DataFrame({"unexpected": [1]}),
            pd.DataFrame({"match_date": ["not-a-date"]}),
            NOW,
            away_travel_distance_km=travel_distance,
        )
        == 0.0
    )
    assert (
        FeatureEngine.compute_team_fatigue(
            pd.DataFrame(),
            "not-a-date",
            travel_distance_km=500,
        )
        == 0.0
    )


def test_build_inference_features_uses_separate_schedule_and_travel_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_LOOKBACK_DAYS", 14)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_REFERENCE_COUNT", 4)
    monkeypatch.setattr(settings, "FATIGUE_IDEAL_REST_DAYS", 7.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_REFERENCE_KM", 1000.0)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 1.0)

    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame(),
        away_matches_df=pd.DataFrame(),
        h2h_rates={},
        fixture_date=NOW,
        home_schedule_df=pd.DataFrame({"match_date": [NOW - timedelta(days=2)]}),
        away_schedule_df=pd.DataFrame({"match_date": [NOW - timedelta(days=1)]}),
        away_travel_distance_km=500.0,
    )

    assert list(features) == FeatureEngine.FEATURE_NAMES
    assert features["fatigue_index"] == 0.5


def test_build_features_normalizes_naive_kickoff_for_rest_difference() -> None:
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame({"match_date": [NOW - timedelta(days=2)]}),
        away_matches_df=pd.DataFrame({"match_date": [NOW - timedelta(days=4)]}),
        h2h_rates={},
        fixture_date=NOW.tz_localize(None),
    )

    assert features["rest_days_diff"] == -2.0


def test_empty_explicit_schedule_falls_back_to_recent_match_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_MATCH_REFERENCE_COUNT", 4)
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 1.0)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 0.0)
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame({"match_date": [NOW - timedelta(days=5)]}),
        away_matches_df=pd.DataFrame(
            {
                "match_date": [
                    NOW - timedelta(days=5),
                    NOW - timedelta(days=2),
                ]
            }
        ),
        h2h_rates={},
        fixture_date=NOW,
        home_schedule_df=pd.DataFrame(),
        away_schedule_df=pd.DataFrame(),
    )

    assert features["fatigue_index"] == 0.25


def test_asymmetric_schedule_coverage_keeps_schedule_penalty_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FATIGUE_MATCH_WEIGHT", 1.0)
    monkeypatch.setattr(settings, "FATIGUE_REST_WEIGHT", 0.0)
    monkeypatch.setattr(settings, "FATIGUE_TRAVEL_WEIGHT", 0.0)
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame({"match_date": [NOW - timedelta(days=1)]}),
        away_matches_df=pd.DataFrame(),
        h2h_rates={},
        fixture_date=NOW,
    )

    assert features["rest_days_diff"] == 0.0
    assert features["fatigue_index"] == 0.0


def test_build_inference_features_keeps_fatigue_neutral_without_kickoff() -> None:
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame({"match_date": [NOW - timedelta(days=2)]}),
        away_matches_df=pd.DataFrame({"match_date": [NOW - timedelta(days=4)]}),
        h2h_rates={},
        fixture_date=None,
        away_schedule_df=pd.DataFrame(
            {"match_date": [pd.Timestamp.now(tz="UTC") - timedelta(days=1)]}
        ),
        away_travel_distance_km=3000.0,
    )

    assert features["fatigue_index"] == 0.0
    assert features["rest_days_diff"] == 0.0


def test_inference_preserves_raw_categorical_ids_for_native_boosters() -> None:
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame(),
        away_matches_df=pd.DataFrame(),
        h2h_rates={},
        fixture_date=NOW,
        league_id=203,
        home_team_id=101,
        away_team_id=202,
    )

    assert features["league_id"] == 203.0
    assert features["home_team_id"] == 101.0
    assert features["away_team_id"] == 202.0
    assert features["league_203"] == 1.0


@pytest.mark.parametrize("league_id", [2, 3, 848])
def test_uefa_competitions_have_distinct_one_hot_features(league_id: int) -> None:
    features = FeatureEngine.build_inference_features(
        home_stats={},
        away_stats={},
        home_matches_df=pd.DataFrame(),
        away_matches_df=pd.DataFrame(),
        h2h_rates={},
        fixture_date=NOW,
        league_id=league_id,
    )

    assert FeatureEngine.SCHEMA_VERSION == "ml_features_v9"
    assert features["league_id"] == float(league_id)
    assert features[f"league_{league_id}"] == 1.0
    assert sum(features[name] for name in FeatureEngine.LEAGUE_FEATURE_NAMES) == 1.0


def test_training_backfills_categories_from_row_for_v5_snapshot() -> None:
    snapshot = {
        name: value
        for name, value in FeatureEngine.FEATURE_DEFAULTS.items()
        if name not in FeatureEngine.CATEGORICAL_FEATURE_NAMES
    }
    row = SimpleNamespace(
        feature_snapshot=snapshot,
        feature_schema_version="ml_features_v5",
        league_id=203,
        home_team_id=101,
        away_team_id=202,
    )

    features = FeatureEngine.build_training_features(row)

    assert features["league_id"] == 203.0
    assert features["home_team_id"] == 101.0
    assert features["away_team_id"] == 202.0
