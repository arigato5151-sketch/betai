"""Train and sign the data-rich and result-only football outcome models.

Run from the repository root with:
``python -m backend.app.prediction.ml.train_tiered_models``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ and __package__.startswith("backend."):
    backend_dir = Path(__file__).resolve().parents[3]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.session import SessionLocal
from app.prediction.ml.ml_pipeline import (
    EstimatorBackend,
    MultiTierDatasetBuilder,
    Tier1Model,
    Tier2Model,
)
from app.prediction.ml.model_router import TieredModelArtifactStore

MINIMUM_SAMPLES_PER_TIER = 12


def load_historical_fixtures() -> list[object]:
    """Read all normalized source data populated by the fixture data pipeline."""
    with SessionLocal() as db:
        return list(HistoricalFixtureRepository(db).get_all())


def _temporal_split(
    features, target, *, minimum_samples: int = MINIMUM_SAMPLES_PER_TIER
):
    if len(features) < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} completed fixtures are required per tier"
        )
    test_size = max(3, int(len(features) * 0.2))
    if len(features) - test_size < 3:
        raise ValueError("Not enough fixtures remain for tier model training")
    return (
        features.iloc[:-test_size].reset_index(drop=True),
        features.iloc[-test_size:].reset_index(drop=True),
        target.iloc[:-test_size].reset_index(drop=True),
        target.iloc[-test_size:].reset_index(drop=True),
    )


def train_tiered_models(
    fixtures: Sequence[object] | None = None,
    *,
    artifact_store: TieredModelArtifactStore | None = None,
    backend: EstimatorBackend = "lightgbm",
) -> dict[str, object]:
    """Build, evaluate, sign, and promote a Tier 1/Tier 2 model bundle."""
    source_fixtures = (
        list(fixtures) if fixtures is not None else load_historical_fixtures()
    )
    datasets = MultiTierDatasetBuilder().build(source_fixtures)
    tier1_train_x, tier1_test_x, tier1_train_y, tier1_test_y = _temporal_split(
        datasets.tier1_features, datasets.tier1_target
    )
    tier2_train_x, tier2_test_x, tier2_train_y, tier2_test_y = _temporal_split(
        datasets.tier2_features, datasets.tier2_target
    )

    tier1 = Tier1Model(backend=backend).train(tier1_train_x, tier1_train_y)
    tier2 = Tier2Model(backend=backend).train(tier2_train_x, tier2_train_y)
    tier1_metrics = tier1.evaluate(tier1_test_x, tier1_test_y)
    tier2_metrics = tier2.evaluate(tier2_test_x, tier2_test_y)
    store = artifact_store or TieredModelArtifactStore()
    bundle = store.export(
        tier1,
        tier2,
        tier1_metrics=tier1_metrics,
        tier2_metrics=tier2_metrics,
        metadata={
            "training_source": "historical_fixtures",
            "training_backend": backend,
            "tier1_training_samples": len(tier1_train_x),
            "tier2_training_samples": len(tier2_train_x),
            "tier1_test_samples": len(tier1_test_x),
            "tier2_test_samples": len(tier2_test_x),
        },
    )
    return {
        "artifact_version": bundle.artifact_version,
        "trained_at": bundle.trained_at,
        "tier1_metrics": tier1_metrics,
        "tier2_metrics": tier2_metrics,
        "metadata": bundle.metadata,
    }


def main() -> None:
    print(json.dumps(train_tiered_models(), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
