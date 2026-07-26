from datetime import UTC, datetime, timedelta

import pytest

from app.db.models import MatchPrediction
from app.prediction.backtest import BacktestEngine

NOW = datetime.now(UTC).replace(tzinfo=None)


def prediction(
    *,
    predicted: str = "HOME_WIN",
    actual: str | None = "HOME_WIN",
    odd: float | None = 2.0,
    edge: float | None = 5.0,
    kelly: float | None = 4.0,
    probability: float = 60.0,
    minute: int = 0,
    closing_odds: float | None = None,
) -> MatchPrediction:
    return MatchPrediction(
        prediction=predicted,
        actual_result=actual,
        odd=odd,
        edge=edge,
        kelly_stake=kelly,
        probability=probability,
        created_at=NOW + timedelta(minutes=minute),
        closing_odds=closing_odds,
    )


def test_empty_backtest_preserves_bankroll() -> None:
    result = BacktestEngine.run_simulation(
        [prediction(actual=None)], initial_bankroll=500
    )

    assert result["final_bankroll"] == 500
    assert result["total_bets"] == 0
    assert result["bankroll_history"] == [500]


def test_flat_strategy_runs_chronologically_and_tracks_drawdown() -> None:
    result = BacktestEngine.run_simulation(
        [
            prediction(predicted="HOME_WIN", actual="AWAY_WIN", minute=2),
            prediction(predicted="DRAW", actual="DRAW", minute=1),
        ],
        initial_bankroll=100,
        strategy="flat",
        flat_stake_amount=10,
        min_edge_pct=3,
    )

    assert result["bankroll_history"] == [100, 110, 100]
    assert result["final_bankroll"] == 100
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["max_drawdown_pct"] == pytest.approx(9.09, abs=0.01)


def test_edge_unresolved_and_invalid_odds_are_skipped() -> None:
    result = BacktestEngine.run_simulation(
        [
            prediction(edge=2.99),
            prediction(actual=None),
            prediction(odd=1.0),
            prediction(odd=None),
        ],
        strategy="flat",
        min_edge_pct=3,
    )

    assert result["total_bets"] == 0
    assert result["final_bankroll"] == 1000


def test_kelly_strategies_apply_fraction_and_five_percent_cap() -> None:
    full_kelly = BacktestEngine.run_simulation(
        [prediction(kelly=80)], initial_bankroll=1000, strategy="kelly"
    )
    fractional = BacktestEngine.run_simulation(
        [prediction(kelly=80)],
        initial_bankroll=1000,
        strategy="fractional_kelly",
        kelly_fraction=0.25,
    )

    assert full_kelly["final_bankroll"] == 1050
    assert fractional["final_bankroll"] == 1012.5


def test_flat_stake_is_capped_and_bankruptcy_stops_simulation() -> None:
    result = BacktestEngine.run_simulation(
        [
            prediction(predicted="HOME_WIN", actual="AWAY_WIN", minute=1),
            prediction(predicted="DRAW", actual="DRAW", minute=2),
        ],
        initial_bankroll=100,
        strategy="flat",
        flat_stake_amount=1000,
    )

    assert result["final_bankroll"] == 0
    assert result["total_bets"] == 1
    assert result["max_drawdown_pct"] == 100


def test_calibration_error_matches_known_example() -> None:
    error = BacktestEngine._compute_calibration_error([(0.8, 1), (0.6, 0)])

    assert error == pytest.approx(0.4)


def test_commission_and_portfolio_limits_are_applied() -> None:
    result = BacktestEngine.run_simulation(
        [prediction(), prediction(minute=1)],
        initial_bankroll=100,
        strategy="flat",
        flat_stake_amount=20,
        commission_pct=10,
        max_stake_pct=10,
        max_daily_exposure_pct=15,
    )

    assert result["bankroll_history"] == [100, 109, 113.5]
    assert result["total_staked"] == 15
    assert result["commission_pct"] == 10


def test_closing_odds_requirement_reports_skipped_records() -> None:
    result = BacktestEngine.run_simulation(
        [prediction(), prediction(closing_odds=1.95, minute=1)],
        strategy="flat",
        require_closing_odds=True,
    )

    assert result["total_bets"] == 1
    assert result["closing_odds_coverage_pct"] == 50
    assert result["skipped_reasons"] == {"missing_closing_odds": 1}


def test_post_kickoff_analysis_is_excluded_as_leakage() -> None:
    row = prediction()
    row.analyzed_at = datetime(2026, 7, 23, 20, tzinfo=UTC)
    row.kickoff = datetime(2026, 7, 23, 19, tzinfo=UTC)

    result = BacktestEngine.run_simulation([row], strategy="flat")

    assert result["total_bets"] == 0
    assert result["skipped_reasons"] == {"post_kickoff_analysis": 1}


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_bankroll": 0},
        {"strategy": "martingale"},
        {"flat_stake_amount": 0},
        {"kelly_fraction": 0},
        {"kelly_fraction": 1.1},
        {"min_edge_pct": -1},
        {"commission_pct": 21},
        {"max_stake_pct": 0},
        {"max_daily_exposure_pct": 101},
    ],
)
def test_engine_rejects_invalid_direct_inputs(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BacktestEngine.run_simulation([prediction()], **overrides)
