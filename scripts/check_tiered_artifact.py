"""Verify the signed tiered-artifact store end to end without a live API.

Exercises export (HMAC signing), reload, and rollback on a throwaway artifact
directory so CI can prove the signed model pipeline is not broken.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("API_FOOTBALL_KEY", "DEMO_KEY")
sys.path.insert(0, str(BACKEND_DIR))

from app.prediction.ml.ml_pipeline import Tier1Model, Tier2Model  # noqa: E402
from app.prediction.ml.model_router import TieredModelArtifactStore  # noqa: E402


def _synthetic_frame(features: tuple[str, ...], n_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: dict[str, list[object]] = {}
    for column in features:
        if column == "league_id":
            rows[column] = [39 if index % 2 == 0 else 140 for index in range(n_samples)]
        elif column in ("home_team", "away_team"):
            rows[column] = [
                ("Manchester City" if index % 2 == 0 else "Arsenal")
                for index in range(n_samples)
            ]
        else:
            rows[column] = rng.uniform(0.0, 3.0, n_samples).tolist()
    return pd.DataFrame(rows, columns=features)


def main() -> None:
    n_samples = 36
    tier1_model = Tier1Model(backend="sklearn").train(
        _synthetic_frame(Tier1Model.FEATURES, n_samples),
        [index % 3 for index in range(n_samples)],
    )
    tier2_model = Tier2Model(backend="sklearn").train(
        _synthetic_frame(Tier2Model.FEATURES, n_samples),
        [index % 3 for index in range(n_samples)],
    )

    with tempfile.TemporaryDirectory() as directory:
        store = TieredModelArtifactStore(artifacts_dir=Path(directory) / "artifacts")
        first = store.export(
            tier1_model,
            tier2_model,
            tier1_metrics={"accuracy": 1.0},
            tier2_metrics={"accuracy": 1.0},
            metadata={"hint": "ci"},
        )
        assert first.artifact_version

        assert store.verify(store.active_path), "active artifact HMAC must validate"
        active = store.load_active()
        assert active is not None and active.artifact_version == first.artifact_version

        second = store.export(
            tier1_model,
            tier2_model,
            tier1_metrics={"accuracy": 1.0},
            tier2_metrics={"accuracy": 1.0},
        )
        assert active.artifact_version != second.artifact_version
        assert store.verify(store.previous_path) is True
        assert store.rollback() is True
        rolled_back = store.load_active()
        assert rolled_back is not None
        assert rolled_back.artifact_version == active.artifact_version
        print(
            f"tiered artifact OK: {active.artifact_version} -> {second.artifact_version} -> rollback"
        )


if __name__ == "__main__":
    main()
