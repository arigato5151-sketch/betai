import pytest

from app.prediction.stats_engine import StatsEngine


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


@pytest.mark.parametrize("rho", [float("nan"), 1.0])
def test_score_matrix_rejects_invalid_dixon_coles_weights(rho: float) -> None:
    with pytest.raises(ValueError):
        StatsEngine._score_probability_matrix(3.4, 3.4, rho=rho)
