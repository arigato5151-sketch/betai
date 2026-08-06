from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.prediction.ml.ml_pipeline import (
    AWAY_WIN,
    DRAW,
    HOME_WIN,
    MultiTierDatasetBuilder,
    Tier1Model,
    Tier2Model,
)


def _fixture(index: int, *, rich: bool, result: str) -> dict[str, object]:
    fixture: dict[str, object] = {
        "kickoff": datetime(2025, 8, 1, tzinfo=UTC) + timedelta(days=index),
        "league_id": 39 if rich else 2,
        "home_team": f"Home {index % 3}",
        "away_team": f"Away {index % 3}",
        "home_goals": (index + 1) % 3,
        "away_goals": index % 2,
        "actual_result": result,
    }
    if rich:
        fixture.update(
            {
                "home_shots": 12,
                "away_shots": 8,
                "home_shots_on_target": 5,
                "away_shots_on_target": 3,
                "home_corners": 6,
                "away_corners": 4,
                "home_fouls": 10,
                "away_fouls": 12,
                "opening_home_odd": 1.9,
                "opening_draw_odd": 3.4,
                "opening_away_odd": 4.2,
                "closing_home_odd": 1.8,
                "closing_draw_odd": 3.5,
                "closing_away_odd": 4.4,
            }
        )
    return fixture


def test_builder_splits_rich_and_result_only_fixtures_without_current_stat_leakage() -> (
    None
):
    fixtures = [
        _fixture(0, rich=True, result="HOME_WIN"),
        _fixture(1, rich=False, result="DRAW"),
    ]

    datasets = MultiTierDatasetBuilder().build(fixtures)

    assert len(datasets.tier1_features) == 1
    assert len(datasets.tier2_features) == 1
    assert datasets.tier1_features.iloc[0]["home_avg_shots"] == 0.0
    assert datasets.tier1_target.tolist() == [HOME_WIN]
    assert datasets.tier2_target.tolist() == [DRAW]


@pytest.mark.parametrize("model_type", [Tier1Model, Tier2Model])
def test_tier_models_train_predict_and_evaluate_three_outcomes(
    model_type: type[Tier1Model] | type[Tier2Model],
) -> None:
    model = model_type(backend="sklearn")
    rows = []
    for index in range(12):
        row: dict[str, object] = {}
        for feature in model.FEATURES:
            if feature in model.CATEGORICAL_FEATURES:
                row[feature] = f"{feature}-{index % 3}"
            else:
                row[feature] = float(index + 1)
        rows.append(row)
    features = pd.DataFrame(rows)
    target = np.array([AWAY_WIN, DRAW, HOME_WIN] * 4)

    model.train(features, target)
    probabilities = model.predict_proba(features.iloc[:2])
    metrics = model.evaluate(features.iloc[:6], target[:6])

    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(metrics) == {"samples", "accuracy", "f1_macro", "log_loss"}


def test_training_rejects_missing_outcome_class() -> None:
    model = Tier2Model(backend="sklearn")
    features = pd.DataFrame(
        [
            {
                name: "team" if name in model.CATEGORICAL_FEATURES else 1.0
                for name in model.FEATURES
            }
        ]
        * 2
    )

    with pytest.raises(ValueError, match="all three outcome classes"):
        model.train(features, [AWAY_WIN, DRAW])
