import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.prediction.stats_engine import (
    StatsEngine,
    build_team_profile,
    time_weighted_goal_averages,
)
from app.prediction.player_impact import PlayerImpactCalculator


def test_time_weighted_goals_favor_recent_matches() -> None:
    as_of = datetime(2026, 7, 20, tzinfo=UTC)
    history = pd.DataFrame(
        [
            {
                "match_date": as_of - timedelta(days=30),
                "goals_for": 0,
                "goals_against": 4,
            },
            {
                "match_date": as_of - timedelta(days=1),
                "goals_for": 4,
                "goals_against": 0,
            },
        ]
    )
    old_weight = math.exp(-0.05 * 30)
    recent_weight = math.exp(-0.05)

    goals_for, goals_against = time_weighted_goal_averages(
        history,
        as_of=as_of,
        decay_factor=0.05,
    )

    assert goals_for == pytest.approx(
        (4 * recent_weight) / (old_weight + recent_weight)
    )
    assert goals_against == pytest.approx(
        (4 * old_weight) / (old_weight + recent_weight)
    )
    assert goals_for > 2.0
    assert goals_against < 2.0


def test_zero_decay_preserves_arithmetic_goal_average() -> None:
    as_of = datetime(2026, 7, 20, tzinfo=UTC)
    history = pd.DataFrame(
        {
            "match_date": [
                as_of - timedelta(days=30),
                as_of - timedelta(days=1),
            ],
            "goals_for": [0, 4],
            "goals_against": [4, 0],
        }
    )

    assert time_weighted_goal_averages(
        history,
        as_of=as_of,
        decay_factor=0.0,
    ) == pytest.approx((2.0, 2.0))


def test_goal_decay_ignores_future_and_invalid_observations() -> None:
    as_of = datetime(2026, 7, 20, tzinfo=UTC)
    history = pd.DataFrame(
        [
            {
                "match_date": as_of - timedelta(days=1),
                "goals_for": 2,
                "goals_against": 1,
            },
            {
                "match_date": as_of + timedelta(days=1),
                "goals_for": 10,
                "goals_against": 10,
            },
            {
                "match_date": "invalid",
                "goals_for": 9,
                "goals_against": 9,
            },
            {
                "match_date": as_of - timedelta(days=2),
                "goals_for": -1,
                "goals_against": float("inf"),
            },
        ]
    )

    assert time_weighted_goal_averages(history, as_of=as_of) == (2.0, 1.0)
    for invalid_factor in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="decay_factor"):
            time_weighted_goal_averages(history, decay_factor=invalid_factor)


def test_team_profile_uses_weighted_goals_for_poisson_strengths() -> None:
    as_of = datetime(2026, 7, 20, tzinfo=UTC)
    history = pd.DataFrame(
        {
            "match_date": [
                as_of - timedelta(days=60),
                as_of - timedelta(days=1),
            ],
            "goals_for": [0, 4],
            "goals_against": [4, 0],
        }
    )
    aggregate = {
        "form": "LW",
        "goals": {
            "for": {"average": {"home": 2.0}},
            "against": {"average": {"home": 2.0}},
        },
        "fixtures": {"played": {"home": 2}},
    }

    profile = build_team_profile(
        aggregate,
        "home",
        match_history=history,
        as_of=as_of,
        decay_factor=0.05,
    )

    assert profile["method"] == "time_weighted_goal_decay"
    assert profile["goals_for_avg"] > 3.0
    assert profile["goals_against_avg"] < 1.0
    assert profile["attack_strength"] > profile["defense_strength"]


def test_poisson_pmf_matches_known_probability() -> None:
    # P(X=2), lambda=2: e^-2 * 2^2 / 2! = 0.270670566...
    assert StatsEngine._poisson_pmf(2.0, 2) == pytest.approx(0.2706705665, rel=1e-9)


def test_equal_expected_goals_produce_symmetric_outcomes() -> None:
    matrix = StatsEngine._score_probability_matrix(1.4, 1.4, rho=-0.13)
    probabilities = StatsEngine._result_probabilities(matrix)

    assert sum(map(sum, matrix)) == pytest.approx(1.0, abs=1e-12)
    assert probabilities["HOME_WIN"] == pytest.approx(35.81, abs=0.01)
    assert probabilities["DRAW"] == pytest.approx(28.39, abs=0.01)
    assert probabilities["AWAY_WIN"] == pytest.approx(35.81, abs=0.01)
    assert sum(probabilities.values()) == pytest.approx(100.0, abs=0.02)


def test_dixon_coles_adjusts_only_low_score_cells() -> None:
    home_lambda = 1.4
    away_lambda = 1.2
    rho = -0.13

    assert StatsEngine._dixon_coles_adjustment(
        0, 0, home_lambda, away_lambda, rho
    ) == pytest.approx(1 - home_lambda * away_lambda * rho)
    assert StatsEngine._dixon_coles_adjustment(
        1, 1, home_lambda, away_lambda, rho
    ) == pytest.approx(1 - rho)
    assert (
        StatsEngine._dixon_coles_adjustment(2, 1, home_lambda, away_lambda, rho) == 1.0
    )


@pytest.mark.parametrize("rate", [-1.0, float("nan"), float("inf")])
def test_poisson_rejects_invalid_rates(rate: float) -> None:
    with pytest.raises(ValueError):
        StatsEngine._poisson_pmf(rate, 0)


def test_poisson_rejects_negative_goal_count() -> None:
    with pytest.raises(ValueError):
        StatsEngine._poisson_pmf(1.2, -1)


def test_zero_goal_rates_collapse_to_nil_nil() -> None:
    matrix = StatsEngine._score_probability_matrix(0.0, 0.0)

    assert matrix[0][0] == 1.0
    assert StatsEngine._result_probabilities(matrix) == {
        "HOME_WIN": 0.0,
        "DRAW": 100.0,
        "AWAY_WIN": 0.0,
    }
    assert StatsEngine._over_under_probs(matrix) == {
        "over_2_5": 0.0,
        "under_2_5": 100.0,
        "over_1_5": 0.0,
    }
    assert StatsEngine._btts_probs(matrix) == {"yes": 0.0, "no": 100.0}
    assert StatsEngine._most_likely_score(matrix)["label"] == "0-0"


@pytest.mark.parametrize("rate", [0.35, 1.4, 3.4])
def test_symmetric_extreme_matrices_remain_normalized(rate: float) -> None:
    matrix = StatsEngine._score_probability_matrix(rate, rate)
    probabilities = StatsEngine._result_probabilities(matrix)

    assert len(matrix) == 8
    assert all(len(row) == 8 for row in matrix)
    assert min(cell for row in matrix for cell in row) >= 0
    assert sum(map(sum, matrix)) == pytest.approx(1.0, abs=1e-12)
    assert probabilities["HOME_WIN"] == pytest.approx(
        probabilities["AWAY_WIN"], abs=0.01
    )
    assert sum(probabilities.values()) == pytest.approx(100.0, abs=0.02)


def test_extreme_strength_gap_mirrors_home_and_away_outcomes() -> None:
    away_favorite = StatsEngine._result_probabilities(
        StatsEngine._score_probability_matrix(0.35, 3.4)
    )
    home_favorite = StatsEngine._result_probabilities(
        StatsEngine._score_probability_matrix(3.4, 0.35)
    )

    assert away_favorite["AWAY_WIN"] == pytest.approx(91.17, abs=0.01)
    assert home_favorite["HOME_WIN"] == pytest.approx(91.17, abs=0.01)
    assert away_favorite["DRAW"] == home_favorite["DRAW"]


def test_secondary_market_pairs_are_complementary() -> None:
    matrix = StatsEngine._score_probability_matrix(1.8, 1.1)
    totals = StatsEngine._over_under_probs(matrix)
    btts = StatsEngine._btts_probs(matrix)

    assert totals["over_2_5"] + totals["under_2_5"] == pytest.approx(100.0, abs=0.01)
    assert btts["yes"] + btts["no"] == pytest.approx(100.0, abs=0.01)
    assert all(
        0 <= probability <= 100 for probability in [*totals.values(), *btts.values()]
    )


def test_analysis_clamps_extreme_inputs_and_returns_consistent_markets() -> None:
    weak_team = {"form": 0, "attack": 0, "defense": 0, "xg": 0}
    strong_team = {"form": 100, "attack": 100, "defense": 100, "xg": 5}

    result = StatsEngine.analyze_match(weak_team, strong_team, league_id=203)

    assert result["expected_goals"]["home"] == 0.35
    assert 0.35 <= result["expected_goals"]["away"] <= 3.4
    assert sum(result["all_probabilities"].values()) == pytest.approx(100.0, abs=0.02)
    assert result["prediction"] in {"HOME_WIN", "DRAW", "AWAY_WIN"}
    assert result["expected_score"]["probability"] > 0
    assert result["score_band"] in {"0-2 Gol", "3-4 Gol", "5+ Gol"}
    assert len(result["secondary_markets"]) >= 3


def test_critical_player_absence_reduces_only_affected_team_xg() -> None:
    profile = {
        "form": 50,
        "attack_strength": 1.0,
        "defense_strength": 1.0,
        "goals_for_avg": 1.3,
        "goals_against_avg": 1.3,
    }
    ratings = {player_id: 6.0 for player_id in range(1, 12)}
    ratings[1] = 9.0
    impact = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=list(range(1, 12)),
        missing_player_ids=[1],
    )

    baseline = StatsEngine.analyze_match(profile, profile, league_id=203)
    penalized = StatsEngine.analyze_match(
        profile,
        profile,
        league_id=203,
        home_player_impact=impact,
    )

    assert impact.critical_missing_count == 1
    assert penalized["expected_goals"]["home"] < baseline["expected_goals"]["home"]
    assert penalized["expected_goals"]["away"] == baseline["expected_goals"]["away"]
    assert penalized["player_impact"]["home"]["data_available"] is True
    assert penalized["player_impact"]["away"]["team_strength_ratio"] == 1.0


def test_invalid_player_multiplier_is_neutral() -> None:
    profile = {
        "form": 50,
        "attack_strength": 1.0,
        "defense_strength": 1.0,
        "goals_for_avg": 1.3,
        "goals_against_avg": 1.3,
    }

    baseline = StatsEngine._expected_goals(profile, profile, is_home=True)

    assert StatsEngine._expected_goals(
        profile,
        profile,
        is_home=True,
        player_xg_multiplier=float("nan"),
    ) == pytest.approx(baseline)


@pytest.mark.parametrize("rho", [float("nan"), 1.0])
def test_score_matrix_rejects_invalid_dixon_coles_weights(rho: float) -> None:
    with pytest.raises(ValueError):
        StatsEngine._score_probability_matrix(3.4, 3.4, rho=rho)
