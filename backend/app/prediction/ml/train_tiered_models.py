"""Train and sign the data-rich and result-only football outcome models.

The training task can source its completed-match data from two places:

* ``database`` -- completed fixtures already persisted by the fixture data
  pipeline (see ``HistoricalFixtureRepository``); used by default.
* ``pipeline`` -- raw league-season CSVs fetched live via
  ``data_pipeline.FootballDataFetcher``.  Optional odds enrichment merges
  opening/closing 1X2 odds from the historical database so fixtures with
  bookmaker odds are promoted to the data-rich Tier 1 set.

Run from the repository root with:
``python -m backend.app.prediction.ml.train_tiered_models``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

if __package__ and __package__.startswith("backend."):
    backend_dir = Path(__file__).resolve().parents[3]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd

from app.db.historical_repository import HistoricalFixtureRepository
from app.db.session import SessionLocal
from app.prediction.ml.ml_pipeline import (
    EstimatorBackend,
    MultiTierDatasetBuilder,
    Tier1Model,
    Tier2Model,
)
from app.prediction.ml.model_router import TieredModelArtifactStore

try:
    from data_pipeline import FootballDataFetcher
except ImportError:  # pragma: no cover - exercised only when runner cwd is on path
    FootballDataFetcher = None  # type: ignore[assignment,misc]

MINIMUM_SAMPLES_PER_TIER = 12

# map ``data_pipeline`` league keys to the stable football API league ids used
# across the platform (also the ids used for Tier 1 routing).
PIPELINE_LEAGUE_IDS: dict[str, int] = {
    "Premier_League": 39,
    "La_Liga": 140,
    "Serie_A": 135,
    "Bundesliga": 78,
    "Ligue_1": 61,
    "Super_Lig": 203,
    "Eredivisie": 88,
    "Liga_Portugal": 94,
}

_FTR_TO_RESULT = {"H": "HOME_WIN", "D": "DRAW", "A": "AWAY_WIN"}


def load_historical_fixtures() -> list[object]:
    """Read all normalized source data populated by the fixture data pipeline."""
    with SessionLocal() as db:
        return list(HistoricalFixtureRepository(db).get_all())


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def normalize_pipeline_row(row: object, league_id: int) -> dict[str, object]:
    """Convert one ``FootballDataFetcher`` term: a row into a builder fixture."""

    def get(name: str) -> object:
        if isinstance(row, dict):
            return row.get(name)
        return getattr(row, name, None)

    ftr = str(get("FTR") or "").strip().upper()
    try:
        kickoff = pd.Timestamp(get("Date")).to_pydatetime()
    except (TypeError, ValueError):
        kickoff = None
    home_score = _optional_int(get("FTHG"))
    away_score = _optional_int(get("FTAG"))

    return {
        "kickoff": kickoff,
        "league_id": league_id,
        "home_team": str(get("HomeTeam") or "").strip(),
        "away_team": str(get("AwayTeam") or "").strip(),
        "home_goals": home_score,
        "away_goals": away_score,
        "actual_result": _FTR_TO_RESULT.get(ftr),
        "home_shots": _optional_int(get("HS")),
        "away_shots": _optional_int(get("AS")),
        "home_shots_on_target": _optional_int(get("HST")),
        "away_shots_on_target": _optional_int(get("AST")),
        "home_corners": _optional_int(get("HC")),
        "away_corners": _optional_int(get("AC")),
        "home_fouls": _optional_int(get("HF")),
        "away_fouls": _optional_int(get("AF")),
    }


def build_odds_provider_from_database() -> object:
    """Return a best-effort odds provider keyed by (league, home, away).

    The provider reads a pipeline fixture dict and returns opening/closing 1X2
    odds previously persisted for the matching teams, or an empty mapping when
    no data-rich counterpart exists.
    """
    with SessionLocal() as db:
        supplements = list(HistoricalFixtureRepository(db).get_all())

    lookup: dict[tuple[int, str, str], dict[str, float]] = {}
    for fixture in supplements:
        try:
            home = str(fixture.home_team).strip().lower()
            away = str(fixture.away_team).strip().lower()
            league_id = int(fixture.league_id)
        except (AttributeError, TypeError, ValueError):
            continue
        odds: dict[str, float] = {}
        for name in (
            "opening_home_odd",
            "opening_draw_odd",
            "opening_away_odd",
            "closing_home_odd",
            "closing_draw_odd",
            "closing_away_odd",
        ):
            value = _optional_float(getattr(fixture, name, None))
            if value is None:
                odds = {}
                break
            odds[name] = value
        if not odds:
            continue
        lookup.setdefault((league_id, home, away), odds)

    def provider(fixture: dict[str, object]) -> dict[str, float]:
        key = (
            int(str(fixture["league_id"])),
            str(fixture["home_team"]).strip().lower(),
            str(fixture["away_team"]).strip().lower(),
        )
        return lookup.get(key, {})

    return provider


def load_fixtures_from_pipeline(
    seasons: Sequence[str],
    leagues: Sequence[str] | None = None,
    *,
    fetcher: Any = None,
    enrich_odds: Any = None,
) -> list[dict[str, object]]:
    """Fetch and normalize all match data pulled via ``data_pipeline``.

    ``enrich_odds`` is an optional callable that receives a normalized fixture
    dict and returns a mapping of odds fields to merge in. When omitted, fetched
    rows keep the statistics columns provided by ``FootballDataFetcher`` and are
    therefore classified as Tier 2 (result + standings only).
    """
    if FootballDataFetcher is None:  # pragma: no cover - path guard
        raise RuntimeError(
            "data_pipeline is not importable; run from the repository root "
            "where 'data_pipeline.py' is on sys.path."
        )
    source = fetcher if fetcher is not None else FootballDataFetcher()
    league_keys = list(leagues or PIPELINE_LEAGUE_IDS)
    fixtures: list[dict[str, object]] = []
    for league in league_keys:
        try:
            league_id = PIPELINE_LEAGUE_IDS[league]
        except KeyError as exc:
            supported = ", ".join(PIPELINE_LEAGUE_IDS)
            raise ValueError(
                f"Unsupported league '{league}'. Supported leagues: {supported}"
            ) from exc
        for season in seasons:
            frame = source.get_league_data(season, league)
            for _, row in frame.iterrows():
                fixture = normalize_pipeline_row(row, league_id)
                if enrich_odds is not None:
                    fixture.update(enrich_odds(fixture))
                fixtures.append(fixture)
    return fixtures


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
    seasons: Sequence[str] | None = None,
    leagues: Sequence[str] | None = None,
    pipeline_fetcher: object | None = None,
    enrich_odds: object | None = None,
) -> dict[str, object]:
    """Build, evaluate, sign, and promote a Tier 1/Tier 2 model bundle."""
    if fixtures is None:
        if seasons:
            fixtures = load_fixtures_from_pipeline(
                seasons,
                leagues,
                fetcher=pipeline_fetcher,
                enrich_odds=enrich_odds or build_odds_provider_from_database(),
            )
        else:
            fixtures = load_historical_fixtures()
    datasets = MultiTierDatasetBuilder().build(fixtures)
    tier1_train_x, tier1_test_x, tier1_train_y, tier1_test_y = _temporal_split(
        datasets.tier1_features, datasets.tier1_target
    )
    tier2_train_x, tier2_test_x, tier2_train_y, tier2_test_y = _temporal_split(
        datasets.tier2_features, datasets.tier2_target
    )

    tier1 = cast(
        Tier1Model, Tier1Model(backend=backend).train(tier1_train_x, tier1_train_y)
    )
    tier2 = cast(
        Tier2Model, Tier2Model(backend=backend).train(tier2_train_x, tier2_train_y)
    )
    tier1_metrics = tier1.evaluate(tier1_test_x, tier1_test_y)
    tier2_metrics = tier2.evaluate(tier2_test_x, tier2_test_y)
    store = artifact_store or TieredModelArtifactStore()
    source = "pipeline" if seasons else "historical_fixtures"
    bundle = store.export(
        tier1,
        tier2,
        tier1_metrics=tier1_metrics,
        tier2_metrics=tier2_metrics,
        metadata={
            "training_source": source,
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and sign tiered models.")
    parser.add_argument(
        "--source",
        choices=("pipeline", "db"),
        default="pipeline",
        help="Where to read the completed match data from.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2425"],
        help="football-data season codes, e.g. 2425.",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(PIPELINE_LEAGUE_IDS),
        default=sorted(PIPELINE_LEAGUE_IDS),
        help="Which leagues to fetch when source=pipeline.",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Skip odds enrichment so stats-only rows stay Tier 2.",
    )
    parser.add_argument(
        "--backend",
        choices=("lightgbm", "sklearn"),
        default="lightgbm",
        help="Estimator backend for both tiers.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Override the tiered artifact storage directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    store = (
        TieredModelArtifactStore(artifacts_dir=args.artifacts_dir)
        if args.artifacts_dir
        else TieredModelArtifactStore()
    )
    if args.source == "pipeline":
        result = train_tiered_models(
            artifact_store=store,
            backend=args.backend,
            seasons=args.seasons,
            leagues=args.leagues,
            enrich_odds=None if args.no_odds else build_odds_provider_from_database(),
        )
    else:
        result = train_tiered_models(artifact_store=store, backend=args.backend)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
