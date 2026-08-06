from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.prediction.ml.model_router import TieredModelArtifactStore
from app.prediction.ml.train_tiered_models import train_tiered_models


def _fixture(index: int, *, rich: bool) -> dict[str, object]:
    outcomes = ("AWAY_WIN", "DRAW", "HOME_WIN")
    fixture: dict[str, object] = {
        "kickoff": datetime(2024, 8, 1, tzinfo=UTC) + timedelta(days=index),
        "league_id": 39 if rich else 2,
        "home_team": f"Home {index % 4}",
        "away_team": f"Away {index % 4}",
        "home_goals": index % 3,
        "away_goals": (index + 1) % 3,
        "actual_result": outcomes[index % 3],
    }
    if rich:
        fixture.update(
            {
                "home_shots": 10 + index % 5,
                "away_shots": 8,
                "home_shots_on_target": 4,
                "away_shots_on_target": 3,
                "home_corners": 5,
                "away_corners": 4,
                "home_fouls": 9,
                "away_fouls": 11,
                "opening_home_odd": 1.9,
                "opening_draw_odd": 3.4,
                "opening_away_odd": 4.3,
                "closing_home_odd": 1.8,
                "closing_draw_odd": 3.5,
                "closing_away_odd": 4.5,
            }
        )
    return fixture


def test_training_smoke_exports_signed_tiered_artifact(tmp_path) -> None:
    fixtures = [
        *(_fixture(index, rich=True) for index in range(18)),
        *(_fixture(index + 40, rich=False) for index in range(18)),
    ]
    store = TieredModelArtifactStore(artifacts_dir=tmp_path)

    result = train_tiered_models(fixtures, artifact_store=store, backend="sklearn")

    assert store.active_path.is_file()
    assert store.verify(store.active_path) is True
    assert store.load_active() is not None
    assert result["tier1_metrics"]["samples"] >= 3
    assert result["tier2_metrics"]["samples"] >= 3


def test_training_rejects_insufficient_tier_data(tmp_path) -> None:
    fixtures = [_fixture(index, rich=True) for index in range(3)]

    with pytest.raises(ValueError, match="required per tier"):
        train_tiered_models(
            fixtures,
            artifact_store=TieredModelArtifactStore(artifacts_dir=tmp_path),
            backend="sklearn",
        )
