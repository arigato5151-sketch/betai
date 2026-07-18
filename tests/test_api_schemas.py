import pytest
from pydantic import ValidationError

from app.api.endpoints import (
    ActualResultUpdate,
    AnalysisRequest,
    BacktestRequest,
    TeamStatsInput,
)


def valid_analysis_payload() -> dict[str, object]:
    return {
        "home_team": "  Home FC  ",
        "away_team": "Away FC",
        "home_stats": {"form": 72.126, "attack": 61.235, "defense": 58, "xg": 1.4567},
        "away_stats": {"form": 68, "attack": 59, "defense": 62, "xg": 1.2345},
        "odd": 2.12345,
    }


def test_analysis_schema_normalizes_names_and_precision() -> None:
    request = AnalysisRequest.model_validate(valid_analysis_payload())

    assert request.home_team == "Home FC"
    assert request.home_stats.form == 72.13
    assert request.home_stats.xg == 1.457
    assert request.odd == 2.123


@pytest.mark.parametrize(
    "payload",
    [
        {"form": -1, "attack": 50, "defense": 50, "xg": 1.2},
        {"form": 50, "attack": 101, "defense": 50, "xg": 1.2},
        {"form": 50, "attack": 50, "defense": 50, "xg": 5.1},
    ],
)
def test_team_stats_schema_rejects_out_of_range_values(
    payload: dict[str, float]
) -> None:
    with pytest.raises(ValidationError):
        TeamStatsInput.model_validate(payload)


def test_analysis_schema_rejects_blank_team_and_invalid_odd() -> None:
    payload = valid_analysis_payload()
    payload.update(home_team="   ", odd=1.0)

    with pytest.raises(ValidationError) as error:
        AnalysisRequest.model_validate(payload)

    assert error.value.error_count() == 2


def test_backtest_schema_enforces_bankroll_and_strategy() -> None:
    with pytest.raises(ValidationError):
        BacktestRequest(initial_bankroll=9.99, strategy="kelly")
    with pytest.raises(ValidationError):
        BacktestRequest(initial_bankroll=1000, strategy="martingale")


def test_actual_result_schema_accepts_only_supported_results() -> None:
    assert ActualResultUpdate(actual_result="DRAW").actual_result == "DRAW"
    with pytest.raises(ValidationError):
        ActualResultUpdate(actual_result="CANCELLED")
