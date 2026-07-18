import pytest

from app.prediction.value_calc import ValueCalc


def test_devig_removes_overround_proportionally() -> None:
    market = ValueCalc.devig_1x2(home_odd=2.0, draw_odd=3.0, away_odd=4.0)

    assert market["overround_pct"] == pytest.approx(8.33, abs=0.01)
    assert market["fair_probability"] == {
        "HOME_WIN": 46.15,
        "DRAW": 30.77,
        "AWAY_WIN": 23.08,
    }
    assert sum(market["fair_probability"].values()) == pytest.approx(100.0)


def test_devig_preserves_fair_market_probabilities() -> None:
    market = ValueCalc.devig_1x2(home_odd=2.0, draw_odd=4.0, away_odd=4.0)

    assert market["overround_pct"] == 0.0
    assert market["fair_probability"] == {
        "HOME_WIN": 50.0,
        "DRAW": 25.0,
        "AWAY_WIN": 25.0,
    }


@pytest.mark.parametrize(
    ("probability", "odd", "expected_stake"),
    [
        (50.0, 2.0, 0.0),  # No positive edge means no stake.
        (60.0, 2.0, 3.0),  # Quarter Kelly is 5%, capped at 3% for odds <= 3.
        (25.0, 4.0, 0.0),
    ],
)
def test_fractional_kelly_stake(
    probability: float, odd: float, expected_stake: float
) -> None:
    assert ValueCalc._kelly_stake(probability, odd) == expected_stake


@pytest.mark.parametrize("invalid_odd", [1.0, 0.0, -2.0, float("nan"), float("inf")])
def test_devig_rejects_incomplete_or_invalid_market(invalid_odd: float) -> None:
    with pytest.raises(ValueError):
        ValueCalc.devig_1x2(2.0, 3.0, invalid_odd)


def test_api_market_parser_skips_incomplete_and_invalid_markets() -> None:
    incomplete = [{"name": "Match Winner", "values": [{"value": "Home", "odd": "2.0"}]}]
    invalid = [
        {
            "name": "Match Winner",
            "values": [
                {"value": "Home", "odd": "2.0"},
                {"value": "Draw", "odd": "3.0"},
                {"value": "Away", "odd": "1.0"},
            ],
        }
    ]

    assert ValueCalc.parse_from_api_bets(incomplete) is None
    assert ValueCalc.parse_from_api_bets(invalid) is None


def test_best_bookmaker_uses_lowest_overround_complete_market() -> None:
    bookmakers = [
        {
            "name": "High Margin",
            "bets": [
                {
                    "name": "Match Winner",
                    "values": [
                        {"value": "Home", "odd": "2.0"},
                        {"value": "Draw", "odd": "3.0"},
                        {"value": "Away", "odd": "4.0"},
                    ],
                }
            ],
        },
        {
            "name": "Low Margin",
            "bets": [
                {
                    "name": "Match Winner",
                    "values": [
                        {"value": "Home", "odd": "2.1"},
                        {"value": "Draw", "odd": "3.5"},
                        {"value": "Away", "odd": "3.6"},
                    ],
                }
            ],
        },
    ]

    market = ValueCalc.best_market_from_bookmakers(bookmakers)

    assert market is not None
    assert market["bookmaker"] == "Low Margin"
    assert market["overround_pct"] == pytest.approx(3.97, abs=0.01)


def test_default_market_adds_positive_overround_without_changing_model_shape() -> None:
    market = ValueCalc.default_market_from_model(
        {"HOME_WIN": 50, "DRAW": 25, "AWAY_WIN": 25}, overround_pct=5
    )

    assert market["overround_pct"] == pytest.approx(5.13, abs=0.01)
    assert market["fair_probability"]["HOME_WIN"] == pytest.approx(50.0, abs=0.1)
    assert market["fair_probability"]["DRAW"] == pytest.approx(25.0, abs=0.1)
    assert market["raw_odds"]["HOME_WIN"] < 2.0


def test_home_odd_hint_rebalances_other_outcomes_to_target_overround() -> None:
    market = ValueCalc.default_market_from_model(
        {"HOME_WIN": 45, "DRAW": 30, "AWAY_WIN": 25},
        home_odd_hint=2.2,
        overround_pct=5,
    )

    assert market["raw_odds"]["HOME_WIN"] == 2.2
    assert market["overround_pct"] == pytest.approx(5.0, abs=0.15)
    assert market["overround_pct"] > 0


@pytest.mark.parametrize("hint", [1.0, float("nan"), float("inf")])
def test_default_market_rejects_invalid_home_hint(hint: float) -> None:
    with pytest.raises(ValueError):
        ValueCalc.default_market_from_model(
            {"HOME_WIN": 40, "DRAW": 30, "AWAY_WIN": 30}, home_odd_hint=hint
        )


@pytest.mark.parametrize(
    ("probability", "odd", "expected_cap"),
    [(80.0, 1.8, 5.0), (60.0, 2.5, 3.0), (40.0, 4.0, 1.5)],
)
def test_kelly_respects_odds_based_caps(
    probability: float, odd: float, expected_cap: float
) -> None:
    assert ValueCalc._kelly_stake(probability, odd) == expected_cap


@pytest.mark.parametrize(
    ("probability", "odd"),
    [(101.0, 2.0), (-1.0, 2.0), (float("nan"), 2.0), (50.0, float("inf"))],
)
def test_kelly_rejects_invalid_inputs(probability: float, odd: float) -> None:
    with pytest.raises(ValueError):
        ValueCalc._kelly_stake(probability, odd)


def test_value_threshold_boundary_is_deterministic() -> None:
    assert ValueCalc._is_value_bet(edge_pct=2.5, implied_probability=50.0) is True
    assert ValueCalc._is_value_bet(edge_pct=2.49, implied_probability=50.0) is False
    assert ValueCalc._is_value_bet(float("nan"), 50.0) is False


def test_professional_evaluation_selects_best_value_and_single_odd_fallback() -> None:
    analysis = {"all_probabilities": {"HOME_WIN": 60.0, "DRAW": 20.0, "AWAY_WIN": 20.0}}
    market = ValueCalc.devig_1x2(2.0, 3.5, 4.0)

    evaluated = ValueCalc.calculate_professional(analysis, market)
    fallback = ValueCalc.calculate_professional(analysis, None, fallback_odd=2.0)

    assert evaluated["value_bet"] is True
    assert evaluated["best_pick"]["outcome"] == "HOME_WIN"
    assert evaluated["best_pick"]["edge"] == 20.0
    assert evaluated["best_pick"]["kelly_stake_pct"] == 3.0
    assert evaluated["value_options"] == sorted(
        evaluated["value_options"], key=lambda option: option["edge"], reverse=True
    )
    assert fallback["market"] == "SINGLE_ODD"


def test_professional_evaluation_returns_neutral_result_without_market() -> None:
    result = ValueCalc.calculate_professional(
        {"all_probabilities": {"HOME_WIN": 40, "DRAW": 30, "AWAY_WIN": 30}},
        None,
    )

    assert result == {
        "value_bet": False,
        "edge": 0.0,
        "implied_probability": 0.0,
        "fair_odd": 0.0,
        "best_pick": None,
        "value_options": [],
        "market": None,
    }
