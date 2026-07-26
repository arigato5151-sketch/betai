from dataclasses import FrozenInstanceError

import pytest

from app.core.config import settings
from app.prediction.player_impact import PlayerImpactCalculator


@pytest.fixture(autouse=True)
def configured_player_impact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "PLAYER_IMPACT_MIN_RATED_STARTERS", 7)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_REPLACEMENT_FACTOR", 0.5)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_MIN_STRENGTH_RATIO", 0.5)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_MAX_STRENGTH_RATIO", 1.05)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_XG_ELASTICITY", 1.0)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_MIN_XG_MULTIPLIER", 0.5)
    monkeypatch.setattr(settings, "PLAYER_CRITICAL_ABSENCE_WEIGHT", 0.25)
    monkeypatch.setattr(settings, "PLAYER_QUESTIONABLE_ABSENCE_WEIGHT", 0.4)


def _lineup() -> list[int]:
    return list(range(1, 12))


def test_insufficient_reference_coverage_returns_neutral_impact() -> None:
    ratings = {player_id: 7.0 for player_id in range(1, 7)}

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        missing_player_ids=[1],
    )

    assert result.strength_ratio == 1.0
    assert result.team_strength_ratio == 1.0
    assert result.xg_multiplier == 1.0
    assert result.data_available is False
    assert result.rated_reference_starters == 6


@pytest.mark.parametrize(
    "reference_lineup",
    [None, [], list(range(1, 11)), [1] * 11],
)
def test_incomplete_or_duplicate_reference_lineup_is_neutral(
    reference_lineup: list[int] | None,
) -> None:
    result = PlayerImpactCalculator.assess(
        {player_id: 7.0 for player_id in _lineup()},
        reference_lineup=reference_lineup,
    )

    assert result.data_available is False
    assert result.strength_ratio == 1.0


def test_critical_missing_player_costs_more_than_below_average_player() -> None:
    ratings = {player_id: 6.0 for player_id in _lineup()}
    ratings[11] = 9.0

    ordinary = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        missing_player_ids=[1],
    )
    critical = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        missing_player_ids=[11],
    )

    assert ordinary.data_available is True
    assert ordinary.critical_missing_player_ids == ()
    assert critical.critical_missing_player_ids == (11,)
    assert critical.critical_missing_count == 1
    assert critical.strength_ratio < ordinary.strength_ratio < 1.0
    assert critical.xg_multiplier < ordinary.xg_multiplier


def test_partial_rating_coverage_is_imputed_without_false_strength_change() -> None:
    ratings = {player_id: 7.0 for player_id in range(1, 8)}

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        current_lineup=_lineup(),
    )

    assert result.data_available is True
    assert result.rated_reference_starters == 7
    assert result.rated_current_starters == 7
    assert result.reference_total_impact == pytest.approx(77.0)
    assert result.adjusted_total_impact == pytest.approx(77.0)
    assert result.strength_ratio == 1.0


def test_confirmed_starter_overrides_stale_missing_status() -> None:
    ratings = {player_id: 7.0 for player_id in _lineup()}

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        current_lineup=_lineup(),
        missing_player_ids=[1],
        questionable_player_ids=[2],
    )

    assert result.missing_player_ids == ()
    assert result.questionable_player_ids == ()
    assert result.strength_ratio == 1.0


def test_confirmed_lineup_penalizes_questionable_reference_player_who_is_absent() -> (
    None
):
    ratings = {player_id: 10.0 for player_id in _lineup()}
    ratings[12] = 10.0
    current = list(range(1, 11)) + [12]

    baseline = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        current_lineup=current,
    )
    questionable = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        current_lineup=current,
        questionable_player_ids=[11],
    )

    # The observed replacement is already discounted; availability uncertainty
    # applies an additional probability-weighted loss to the omitted starter.
    assert baseline.adjusted_total_impact == pytest.approx(105.0)
    assert questionable.questionable_player_ids == (11,)
    assert questionable.adjusted_total_impact == pytest.approx(103.0)
    assert questionable.strength_ratio < baseline.strength_ratio


def test_questionable_player_uses_probability_weight_and_missing_takes_priority() -> (
    None
):
    ratings = {player_id: 10.0 for player_id in _lineup()}

    questionable = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        questionable_player_ids=[1],
    )
    both = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        missing_player_ids=[1],
        questionable_player_ids=[1, 2],
    )

    assert questionable.strength_ratio == pytest.approx(108.0 / 110.0, abs=1e-6)
    assert both.missing_player_ids == (1,)
    assert both.questionable_player_ids == (2,)


def test_confirmed_lineup_uses_rated_replacement_quality() -> None:
    ratings = {player_id: 10.0 for player_id in _lineup()}
    ratings[12] = 20.0
    current = list(range(1, 11)) + [12]

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        current_lineup=current,
        missing_player_ids=[11],
    )

    # Ten retained starters plus a 50%-credited high-quality replacement.
    assert result.adjusted_total_impact == pytest.approx(110.0)
    assert result.strength_ratio == 1.0
    assert result.rated_current_starters == 11


def test_strong_replacement_and_xg_boost_are_capped() -> None:
    ratings = {player_id: 10.0 for player_id in _lineup()}
    ratings[12] = 50.0

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        current_lineup=list(range(1, 11)) + [12],
    )

    assert result.strength_ratio == 1.05
    assert result.xg_multiplier == 1.05


def test_xg_elasticity_respects_configured_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "PLAYER_IMPACT_REPLACEMENT_FACTOR", 0.0)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_XG_ELASTICITY", 3.0)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_MIN_XG_MULTIPLIER", 0.8)
    ratings = {player_id: 10.0 for player_id in _lineup()}

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
        missing_player_ids=list(range(1, 8)),
    )

    assert result.strength_ratio == 0.5
    assert result.xg_multiplier == 0.8


def test_dict_ratings_and_json_string_player_ids_are_supported() -> None:
    ratings = {
        str(player_id): {
            "rating": "7.0",
            "minutes": 900,
            "appearances": 10,
            "goals": 1,
            "assists": 1,
        }
        for player_id in _lineup()
    }

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=[str(player_id) for player_id in _lineup()],
    )

    assert result.data_available is True
    assert result.reference_average_impact == 7.0
    assert result.strength_ratio == 1.0


def test_contribution_only_dict_is_normalized_by_exposure() -> None:
    ratings = {
        player_id: {
            "minutes": 900,
            "appearances": 10,
            "goals": 10,
            "assists": 0,
        }
        for player_id in _lineup()
    }

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
    )

    assert result.data_available is True
    assert result.reference_average_impact == 1.0


def test_nan_infinite_boolean_and_malformed_values_are_ignored() -> None:
    ratings: dict[object, object] = {
        1: float("nan"),
        2: float("inf"),
        3: True,
        4: {"rating": "-1"},
        5: {"rating": "not-a-number"},
        6: {"minutes": "invalid", "goals": 3},
        7: None,
        "bad-id": 8.0,
        **{player_id: 7.0 for player_id in range(8, 12)},
    }

    result = PlayerImpactCalculator.assess(
        ratings,
        reference_lineup=_lineup(),
    )

    assert result.data_available is False
    assert result.strength_ratio == 1.0
    assert result.xg_multiplier == 1.0


def test_result_is_immutable() -> None:
    result = PlayerImpactCalculator.assess(
        {player_id: 7.0 for player_id in _lineup()},
        reference_lineup=_lineup(),
    )

    with pytest.raises(FrozenInstanceError):
        result.strength_ratio = 0.5  # type: ignore[misc]
