"""Train and sign the data-rich and result-only football outcome models.

The training task can source its completed-match data from two places:

* ``database`` -- completed fixtures already persisted by the fixture data
  pipeline (see ``HistoricalFixtureRepository``); used by default.
* ``pipeline`` -- raw league-season CSVs fetched live via
  ``data_pipeline.FootballDataFetcher``. Optional odds enrichment merges
  opening 1X2 odds from the historical database so fixtures with information
  available before kickoff are promoted to the market-aware Tier 1 set.

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
from sklearn.metrics import accuracy_score, log_loss

from app.core.config import settings
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


class ModelPromotionRejected(RuntimeError):
    """Raised when a candidate fails the production promotion gate."""

    def __init__(self, tier1_metrics: dict[str, object]) -> None:
        super().__init__("Tier 1 candidate failed the production promotion gate")
        self.tier1_metrics = dict(tier1_metrics)


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
    """Return a best-effort opening-odds provider keyed by team identity.

    The provider reads a pipeline fixture dict and returns opening/closing 1X2
    odds previously persisted for the matching teams, or an empty mapping when
    no pre-kickoff market counterpart exists.
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


def _market_benchmark_metrics(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    model_probabilities: np.ndarray,
) -> dict[str, object]:
    """Compare Tier 1 with the market using paired holdout loss differences.

    The closing line is the only honest benchmark. When closing 1X2 odds are
    present and valid they are used; otherwise the benchmark falls back to the
    opening price and the caller can treat that as weaker evidence.
    """
    closing_columns = ("closing_away_odd", "closing_draw_odd", "closing_home_odd")
    opening_columns = ("opening_away_odd", "opening_draw_odd", "opening_home_odd")

    def valid_odds(columns: tuple[str, ...]) -> np.ndarray | None:
        try:
            odds = (
                features.loc[:, columns].apply(pd.to_numeric, errors="coerce")
            ).to_numpy()
        except KeyError:
            return None
        if len(odds) == 0 or not np.isfinite(odds).all() or np.any(odds <= 1.0):
            return None
        return odds

    market_line = "closing"
    odds = valid_odds(closing_columns)
    if odds is None:
        market_line = "opening"
        odds = valid_odds(opening_columns)
    if odds is None:
        raise ValueError("Tier 1 holdout requires valid closing or opening 1X2 odds")

    inverse = 1.0 / odds
    market_probabilities = inverse / inverse.sum(axis=1, keepdims=True)
    actual = target.to_numpy(dtype=int)
    candidate_probabilities = np.asarray(model_probabilities, dtype=float)
    if (
        candidate_probabilities.shape != market_probabilities.shape
        or not np.isfinite(candidate_probabilities).all()
        or np.any(candidate_probabilities < 0)
        or np.any(candidate_probabilities.sum(axis=1) <= 0)
    ):
        raise ValueError("Tier 1 candidate returned invalid holdout probabilities")
    candidate_probabilities = candidate_probabilities / candidate_probabilities.sum(
        axis=1, keepdims=True
    )

    market_log_loss = float(log_loss(actual, market_probabilities, labels=[0, 1, 2]))
    market_brier = float(
        np.mean(
            np.sum(
                (market_probabilities - np.eye(3, dtype=float)[actual]) ** 2,
                axis=1,
            )
        )
    )
    model_log_loss = float(log_loss(actual, candidate_probabilities, labels=[0, 1, 2]))
    one_hot = np.eye(3, dtype=float)[actual]
    model_brier = float(
        np.mean(np.sum((candidate_probabilities - one_hot) ** 2, axis=1))
    )
    log_loss_improvement = market_log_loss - model_log_loss
    brier_improvement = market_brier - model_brier

    indices = np.arange(len(actual))
    clipped_market = np.clip(market_probabilities[indices, actual], 1e-15, 1.0)
    clipped_candidate = np.clip(candidate_probabilities[indices, actual], 1e-15, 1.0)
    per_sample_log_improvement = np.log(clipped_candidate) - np.log(clipped_market)
    per_sample_brier_improvement = np.sum(
        (market_probabilities - one_hot) ** 2
        - (candidate_probabilities - one_hot) ** 2,
        axis=1,
    )
    rng = np.random.default_rng(42)
    bootstrap_indices = rng.integers(
        0,
        len(actual),
        size=(settings.TIERED_PROMOTION_BOOTSTRAP_SAMPLES, len(actual)),
    )
    lower_percentile = (1.0 - settings.TIERED_PROMOTION_CONFIDENCE) * 100.0
    log_loss_lower_bound = float(
        np.percentile(
            per_sample_log_improvement[bootstrap_indices].mean(axis=1),
            lower_percentile,
        )
    )
    brier_lower_bound = float(
        np.percentile(
            per_sample_brier_improvement[bootstrap_indices].mean(axis=1),
            lower_percentile,
        )
    )
    sample_sufficient = len(actual) >= settings.TIERED_PROMOTION_MIN_HOLDOUT_SAMPLES
    beats_market = (
        sample_sufficient
        and log_loss_improvement >= settings.MIN_TIERED_MARKET_LOG_LOSS_IMPROVEMENT
        and brier_improvement >= settings.MIN_TIERED_MARKET_BRIER_IMPROVEMENT
        and log_loss_lower_bound > 0.0
        and brier_lower_bound > 0.0
    )
    return {
        "market_line": market_line,
        "market_accuracy": float(
            accuracy_score(actual, market_probabilities.argmax(axis=1))
        ),
        "market_log_loss": market_log_loss,
        "market_brier_score": market_brier,
        "log_loss_improvement_vs_market": log_loss_improvement,
        "brier_improvement_vs_market": brier_improvement,
        "market_log_loss_improvement_lower_bound": log_loss_lower_bound,
        "market_brier_improvement_lower_bound": brier_lower_bound,
        "promotion_holdout_samples": len(actual),
        "promotion_sample_sufficient": sample_sufficient,
        "beats_opening_market": beats_market,
    }


def _champion_benchmark_metrics(
    candidate_metrics: dict[str, object], champion_metrics: dict[str, object]
) -> dict[str, object]:
    """Require a material gain over the active model with bounded trade-offs."""

    def metric(source: dict[str, object], name: str) -> float:
        value = source.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Champion comparison metrics are incomplete")
        return float(value)

    candidate_log_loss = metric(candidate_metrics, "log_loss")
    candidate_brier = metric(candidate_metrics, "brier_score")
    champion_log_loss = metric(champion_metrics, "log_loss")
    champion_brier = metric(champion_metrics, "brier_score")
    values = (
        candidate_log_loss,
        candidate_brier,
        champion_log_loss,
        champion_brier,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Champion comparison metrics must be finite")

    log_loss_improvement = champion_log_loss - candidate_log_loss
    brier_improvement = champion_brier - candidate_brier
    log_loss_path = (
        log_loss_improvement >= settings.MIN_TIERED_CHAMPION_LOG_LOSS_IMPROVEMENT
        and brier_improvement >= -settings.MAX_TIERED_CHAMPION_BRIER_REGRESSION
    )
    brier_path = (
        brier_improvement >= settings.MIN_TIERED_CHAMPION_BRIER_IMPROVEMENT
        and log_loss_improvement >= -settings.MAX_TIERED_CHAMPION_LOG_LOSS_REGRESSION
    )
    return {
        "champion_log_loss": champion_log_loss,
        "champion_brier_score": champion_brier,
        "log_loss_improvement_vs_champion": log_loss_improvement,
        "brier_improvement_vs_champion": brier_improvement,
        "beats_active_champion": log_loss_path or brier_path,
    }


def _tier2_gate(target: pd.Series) -> dict[str, object]:
    """Judge whether a Tier 2 model has enough evidence to leave research mode.

    A handful of matches cannot estimate a three-class model with confidence;
    below the configured sample and per-class floors the forecasts stay marked
    as research-only downstream instead of being presented as decisions.
    """
    training_samples = int(len(target))
    per_class_counts = {
        str(label): int(count) for label, count in target.value_counts().items()
    }
    min_per_class = int(min(per_class_counts.values(), default=0))
    minimum_samples = settings.TIERED_MIN_TIER2_SAMPLES
    minimum_per_class = settings.TIERED_MIN_TIER2_SAMPLES_PER_CLASS
    reasons: list[str] = []
    if training_samples < minimum_samples:
        reasons.append("insufficient_tier2_samples")
    if min_per_class < minimum_per_class:
        reasons.append("tier2_class_imbalance")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "training_samples": training_samples,
        "minimum_samples": minimum_samples,
        "min_per_class": min_per_class,
        "minimum_per_class": minimum_per_class,
        "class_distribution": per_class_counts,
    }


def train_tiered_models(
    fixtures: Sequence[object] | None = None,
    *,
    artifact_store: TieredModelArtifactStore | None = None,
    backend: EstimatorBackend = "lightgbm",
    calibration_method: str | None = None,
    seasons: Sequence[str] | None = None,
    leagues: Sequence[str] | None = None,
    pipeline_fetcher: object | None = None,
    enrich_odds: object | None = None,
    require_market_superiority: bool = True,
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

    tier1 = Tier1Model(backend=backend)
    tier2 = Tier2Model(backend=backend)
    if calibration_method in ("isotonic", "platt", "auto", "none"):
        tier1.calibration_method = calibration_method
        tier2.calibration_method = calibration_method
    tier1 = cast(Tier1Model, tier1.train(tier1_train_x, tier1_train_y))
    tier2 = cast(Tier2Model, tier2.train(tier2_train_x, tier2_train_y))
    tier1_metrics = tier1.evaluate(tier1_test_x, tier1_test_y)
    tier1_metrics.update(
        _market_benchmark_metrics(
            tier1_test_x,
            tier1_test_y,
            model_probabilities=tier1.predict_proba(tier1_test_x),
        )
    )
    tier2_metrics = tier2.evaluate(tier2_test_x, tier2_test_y)
    tier2_metrics["tier2_gate"] = _tier2_gate(tier2_train_y)
    store = artifact_store or TieredModelArtifactStore()
    active_bundle = store.load_active()
    if active_bundle is not None:
        champion_metrics = active_bundle.tier1_model.evaluate(
            tier1_test_x, tier1_test_y
        )
        tier1_metrics.update(
            _champion_benchmark_metrics(tier1_metrics, champion_metrics)
        )
    else:
        tier1_metrics["beats_active_champion"] = None
    promotion_failures = []
    if tier1_metrics["beats_opening_market"] is not True:
        promotion_failures.append("opening_market")
    if tier1_metrics["beats_active_champion"] is False:
        promotion_failures.append("active_champion")
    # Beating the stale opening line proves nothing about the sharps. Claiming
    # market superiority therefore requires the closing line as the benchmark.
    if tier1_metrics.get("market_line") == "opening":
        promotion_failures.append("opening_only_benchmark")
    tier1_metrics["promotion_failures"] = promotion_failures
    if require_market_superiority and promotion_failures:
        raise ModelPromotionRejected(tier1_metrics)
    source = "pipeline" if seasons else "historical_fixtures"
    bundle = store.export(
        tier1,
        tier2,
        tier1_metrics=tier1_metrics,
        tier2_metrics=tier2_metrics,
        metadata={
            "training_source": source,
            "training_backend": backend,
            "calibration_method": tier1.calibration_method,
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
        "--calibration-method",
        choices=("isotonic", "platt", "auto", "none"),
        default=None,
        help="Probability calibration method; defaults to the configured setting.",
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
            calibration_method=args.calibration_method,
            seasons=args.seasons,
            leagues=args.leagues,
            enrich_odds=None if args.no_odds else build_odds_provider_from_database(),
        )
    else:
        result = train_tiered_models(
            artifact_store=store,
            backend=args.backend,
            calibration_method=args.calibration_method,
        )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
