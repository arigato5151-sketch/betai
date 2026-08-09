"""Model leakage gate (P7).

Consolidated quantitative gate for temporal hygiene of the training
pipeline.  CI runs this file as the dedicated "model leakage test" step:
if a future result can change the features of a past fixture, or if a
temporal holdout no longer respects chronological order, the gate fails.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from app.prediction.ml.ml_pipeline import MultiTierDatasetBuilder
from app.prediction.ml.train_tiered_models import _temporal_split


def _fixture(
    index: int,
    *,
    home: str,
    away: str,
    result: str,
    goals_home: int,
    goals_away: int,
) -> dict[str, object]:
    return {
        "kickoff": datetime(2025, 8, 1, tzinfo=UTC) + timedelta(days=index),
        "league_id": 39,
        "home_team": home,
        "away_team": away,
        "home_goals": goals_home,
        "away_goals": goals_away,
        "actual_result": result,
    }


def _season() -> list[dict[str, object]]:
    """Repeated three-team round robin, one match per day.

    Leakage-free pipelines must only reflect pre-match information, so a
    win by the strong side on day N must not change the features seen on
    day N's own fixture nor earlier ones.
    """
    fixtures: list[dict[str, object]] = []
    cycle = [
        ("Strong", "Weak", 3, 0, "HOME_WIN"),
        ("Strong", "Mid", 2, 1, "HOME_WIN"),
        ("Mid", "Weak", 1, 0, "HOME_WIN"),
        ("Weak", "Mid", 0, 2, "AWAY_WIN"),
        ("Mid", "Strong", 1, 2, "AWAY_WIN"),
        ("Weak", "Strong", 0, 3, "AWAY_WIN"),
    ]
    for index in range(18):
        home, away, gh, ga, result = cycle[index % len(cycle)]
        fixtures.append(
            _fixture(
                index,
                home=home,
                away=away,
                result=result,
                goals_home=gh,
                goals_away=ga,
            )
        )
    return fixtures


def _numeric(frame: pd.DataFrame) -> np.ndarray:
    return frame.select_dtypes(include=["number"]).to_numpy(dtype=float), [
        str(name) for name in frame.select_dtypes(include=["number"]).columns
    ]


def test_gate_past_rows_do_not_change_when_future_results_are_removed() -> None:
    full = MultiTierDatasetBuilder().build(_season())
    past = MultiTierDatasetBuilder().build(_season()[:12])
    assert len(past.tier2_features) == 12
    joined, full_columns = _numeric(
        full.tier2_features.iloc[:12].reset_index(drop=True)
    )
    truncated, past_columns = _numeric(past.tier2_features.reset_index(drop=True))
    assert full_columns == past_columns
    np.testing.assert_allclose(
        joined,
        truncated,
        rtol=1e-9,
        atol=1e-9,
        err_msg="Removing later fixtures altered features of earlier ones: "
        "the pipeline leaks future information backwards",
    )


def test_gate_walk_forward_split_preserves_temporal_order() -> None:
    datasets = MultiTierDatasetBuilder().build(_season())
    train_x, test_x, train_y, test_y = _temporal_split(
        datasets.tier2_features, datasets.tier2_target
    )
    assert len(train_x) + len(test_x) == len(datasets.tier2_features)
    assert len(test_x) == 3
    assert len(train_x) >= 12
    test_numeric, _ = _numeric(test_x)
    tail_numeric, _ = _numeric(
        datasets.tier2_features.iloc[len(train_x) :].reset_index(drop=True)
    )
    np.testing.assert_allclose(
        test_numeric,
        tail_numeric,
        rtol=1e-9,
        atol=1e-9,
        err_msg="Temporal holdout does not sample the chronologically last rows",
    )


def test_gate_no_closing_odds_target_leakage_and_no_current_stat_leakage() -> None:
    rich = dict(_season()[0])
    rich.update(
        {
            "home_shots": 12,
            "away_shots": 9,
            "home_shots_on_target": 5,
            "away_shots_on_target": 2,
            "home_corners": 6,
            "away_corners": 3,
            "home_fouls": 10,
            "away_fouls": 12,
            "opening_home_odd": 1.9,
            "opening_draw_odd": 3.4,
            "opening_away_odd": 4.2,
            "closing_home_odd": 1.05,
            "closing_draw_odd": 1.2,
            "closing_away_odd": 21.0,
        }
    )
    datasets = MultiTierDatasetBuilder().build([rich])
    assert len(datasets.tier1_features) == 1
    for column in ("closing_home_odd", "closing_draw_odd", "closing_away_odd"):
        assert column not in datasets.tier1_features.columns
    assert datasets.tier1_features.iloc[0]["home_avg_goals"] == 0.0
    assert datasets.tier1_features.iloc[0]["away_avg_goals"] == 0.0
