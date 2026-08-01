from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import sys
from collections import defaultdict, deque
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import httpx
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
if not BACKEND_DIR.is_dir():
    # The backend Docker image places the app package directly under /app.
    BACKEND_DIR = ROOT_DIR
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.core.team_identity import stable_team_name_key  # noqa: E402
from app.db.models import HistoricalFixture, MatchPrediction  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.prediction.backtest import BacktestEngine  # noqa: E402
from app.prediction.stats_engine import StatsEngine, build_team_profile  # noqa: E402
from app.services.football_data_csv import (  # noqa: E402
    FOOTBALL_DATA_LEAGUES,
    FootballDataCSVClient,
    _stable_negative_id,
)

GRID_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
OUTCOMES = BacktestEngine.OUTCOMES
RICH_PARAMETERS = (
    "GOAL_TIME_DECAY_FACTOR",
    "LEAGUE_BASELINE_GOALS",
    "FORM_DECAY_WEIGHTS",
    "AWAY_ATTACK_PENALTY",
    "PROFILE_FORM_FACTOR_BASE",
    "PROFILE_FORM_FACTOR_WEIGHT",
    "HOME_ADVANTAGE_MIN_MULTIPLIER",
    "HOME_ADVANTAGE_MAX_MULTIPLIER",
    "HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR",
    "DEFAULT_DIXON_COLES_RHO",
    "LEAGUE_DIXON_COLES_RHO",
)
LEGACY_PARAMETERS = (
    "HOME_ATTACK_BOOST",
    "XG_OBSERVED_GOALS_WEIGHT",
    "XG_ATTACK_BASELINE_WEIGHT",
    "XG_CONSISTENCY_MAX_PENALTY",
    "XG_CONSISTENCY_PENALTY_WEIGHT",
    "LEGACY_ATTACK_FACTOR_BASE",
    "LEGACY_ATTACK_FACTOR_WEIGHT",
    "LEGACY_DEFENSE_FACTOR_BASE",
    "LEGACY_DEFENSE_FACTOR_WEIGHT",
    "LEGACY_FORM_FACTOR_BASE",
    "LEGACY_FORM_FACTOR_WEIGHT",
    "LEGACY_XG_OBSERVED_WEIGHT",
    "LEGACY_XG_BASELINE_WEIGHT",
)
SECONDARY_PARAMETERS = (
    "DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT",
    "DOUBLE_CHANCE_AWAY_DIFFERENCE_WEIGHT",
)
ELO_PARAMETERS = ("ELO_K_FACTOR", "ELO_HOME_ADVANTAGE_POINTS")
UNVALIDATED_PARAMETERS = {
    "FORM_DECAY_FALLBACK_WEIGHT": (
        "The form scorer truncates input to five matches, so its fallback branch "
        "is unreachable with the current implementation."
    ),
    "STRENGTH_ATTACK_WEIGHT": (
        "Only strength_rating changes; that display field is not consumed by the "
        "statistical or ML prediction path."
    ),
    "STRENGTH_DEFENSE_WEIGHT": (
        "Only strength_rating changes; that display field is not consumed by the "
        "statistical or ML prediction path."
    ),
    "STRENGTH_FORM_WEIGHT": (
        "Only strength_rating changes; that display field is not consumed by the "
        "statistical or ML prediction path."
    ),
    "HOME_FORM_BASE_MULTIPLIER": (
        "Every historical profile has goals-for/goals-against averages, so the "
        "missing-profile home-advantage fallback is not represented."
    ),
    "HOME_FORM_BOOST_DIVISOR": (
        "Every historical profile has goals-for/goals-against averages, so the "
        "missing-profile home-advantage fallback is not represented."
    ),
    "ENSEMBLE_STATS_WEIGHT": (
        "No resolved prediction contains all stats, ML and market probability "
        "components; at least 100 are required."
    ),
    "ENSEMBLE_ML_WEIGHT": (
        "No resolved prediction contains all stats, ML and market probability "
        "components; at least 100 are required."
    ),
    "ENSEMBLE_MARKET_WEIGHT": (
        "No resolved prediction contains all stats, ML and market probability "
        "components; at least 100 are required."
    ),
    "ELO_SEASON_REGRESSION": (
        "The two seasons use disjoint provider-specific team IDs, so no rating "
        "crosses a season boundary and the parameter has zero observable effect."
    ),
}


@dataclass(frozen=True)
class FixtureRecord:
    fixture_id: int
    league_id: int
    season: int
    kickoff: datetime
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    actual_result: str
    data_source: str


@dataclass(frozen=True)
class TeamWindow:
    form: str
    goals_for: float
    goals_against: float
    clean_sheets: int
    failed_to_score: int
    played: int

    def api_payload(self, venue: str) -> dict[str, object]:
        return {
            "form": self.form,
            "goals": {
                "for": {"average": {venue: self.goals_for}},
                "against": {"average": {venue: self.goals_against}},
            },
            "clean_sheet": {venue: self.clean_sheets},
            "failed_to_score": {venue: self.failed_to_score},
            "fixtures": {"played": {venue: self.played}},
        }


@dataclass(frozen=True)
class CalibrationSample:
    fixture: FixtureRecord
    home_window: TeamWindow
    away_window: TeamWindow
    home_history: tuple[FixtureRecord, ...]
    away_history: tuple[FixtureRecord, ...]


@dataclass(frozen=True)
class CandidateMetrics:
    brier_score: float
    roi_pct: float | None
    backtest_bets: int
    samples: int
    odds_samples: int


def _load_fixtures() -> list[FixtureRecord]:
    with SessionLocal() as session:
        rows = (
            session.query(HistoricalFixture)
            .order_by(HistoricalFixture.kickoff, HistoricalFixture.fixture_id)
            .all()
        )
        return [
            FixtureRecord(
                fixture_id=row.fixture_id,
                league_id=row.league_id,
                season=row.season,
                kickoff=_as_utc(row.kickoff),
                home_team_id=row.home_team_id,
                away_team_id=row.away_team_id,
                home_team=row.home_team,
                away_team=row.away_team,
                home_goals=row.home_goals,
                away_goals=row.away_goals,
                actual_result=row.actual_result,
                data_source=row.data_source,
            )
            for row in rows
        ]


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _team_window(
    fixtures: Iterable[FixtureRecord],
    team_id: int,
) -> TeamWindow:
    form: list[str] = []
    goals_for: list[int] = []
    goals_against: list[int] = []
    for fixture in fixtures:
        is_home = fixture.home_team_id == team_id
        scored = fixture.home_goals if is_home else fixture.away_goals
        conceded = fixture.away_goals if is_home else fixture.home_goals
        goals_for.append(scored)
        goals_against.append(conceded)
        form.append("W" if scored > conceded else "D" if scored == conceded else "L")
    played = len(goals_for)
    return TeamWindow(
        form="".join(form),
        goals_for=sum(goals_for) / played,
        goals_against=sum(goals_against) / played,
        clean_sheets=sum(value == 0 for value in goals_against),
        failed_to_score=sum(value == 0 for value in goals_for),
        played=played,
    )


def _build_samples(
    fixtures: list[FixtureRecord],
    *,
    recent_match_count: int,
    minimum_team_history: int,
    goal_history_match_count: int = 20,
) -> list[CalibrationSample]:
    history: dict[tuple[int, int], deque[FixtureRecord]] = defaultdict(
        lambda: deque(maxlen=max(recent_match_count, goal_history_match_count))
    )
    samples: list[CalibrationSample] = []
    pending: list[FixtureRecord] = []
    current_kickoff: datetime | None = None

    for fixture in fixtures:
        if current_kickoff is not None and fixture.kickoff != current_kickoff:
            for completed in pending:
                history[(completed.league_id, completed.home_team_id)].append(completed)
                history[(completed.league_id, completed.away_team_id)].append(completed)
            pending.clear()
        current_kickoff = fixture.kickoff
        home_history = list(history[(fixture.league_id, fixture.home_team_id)])
        away_history = list(history[(fixture.league_id, fixture.away_team_id)])
        if (
            len(home_history) >= minimum_team_history
            and len(away_history) >= minimum_team_history
        ):
            samples.append(
                CalibrationSample(
                    fixture=fixture,
                    home_window=_team_window(
                        home_history[-recent_match_count:], fixture.home_team_id
                    ),
                    away_window=_team_window(
                        away_history[-recent_match_count:], fixture.away_team_id
                    ),
                    home_history=tuple(home_history),
                    away_history=tuple(away_history),
                )
            )
        pending.append(fixture)
    return samples


@contextmanager
def _override_setting(name: str, value: object):
    original = getattr(settings, name)
    setattr(settings, name, value)
    try:
        yield
    finally:
        setattr(settings, name, original)


def _candidate_value(current: object, factor: float) -> object:
    if isinstance(current, tuple):
        if not current:
            return current
        return (current[0], *(float(value) * factor for value in current[1:]))
    if isinstance(current, dict):
        return {key: float(value) * factor for key, value in current.items()}
    return float(current) * factor


def _serializable_value(value: object) -> object:
    if isinstance(value, tuple):
        return [round(float(item), 6) for item in value]
    if isinstance(value, dict):
        return {str(key): round(float(item), 6) for key, item in value.items()}
    return round(float(value), 6)


def _profiles(
    sample: CalibrationSample,
    *,
    legacy: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    def history_frame(
        fixtures: tuple[FixtureRecord, ...], team_id: int
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "match_date": fixture.kickoff,
                    "goals_for": (
                        fixture.home_goals
                        if fixture.home_team_id == team_id
                        else fixture.away_goals
                    ),
                    "goals_against": (
                        fixture.away_goals
                        if fixture.home_team_id == team_id
                        else fixture.home_goals
                    ),
                }
                for fixture in fixtures
            ]
        )

    home = build_team_profile(
        sample.home_window.api_payload("home"),
        "home",
        match_history=history_frame(sample.home_history, sample.fixture.home_team_id),
        as_of=sample.fixture.kickoff,
    )
    away = build_team_profile(
        sample.away_window.api_payload("away"),
        "away",
        match_history=history_frame(sample.away_history, sample.fixture.away_team_id),
        as_of=sample.fixture.kickoff,
    )
    if legacy:
        home.pop("attack_strength", None)
        home.pop("defense_strength", None)
        away.pop("attack_strength", None)
        away.pop("defense_strength", None)
    return home, away


def _prediction_row(
    fixture: FixtureRecord,
    probabilities: dict[str, float],
    odds: dict[str, float],
) -> MatchPrediction:
    edges = {
        outcome: probabilities[outcome] * odds[outcome] - 1.0 for outcome in OUTCOMES
    }
    prediction = max(edges, key=edges.__getitem__)
    return MatchPrediction(
        fixture_id=fixture.fixture_id,
        prediction=prediction,
        actual_result=fixture.actual_result,
        probability=probabilities[prediction] * 100.0,
        odd=odds[prediction],
        edge=edges[prediction] * 100.0,
        kelly_stake=0.0,
        kickoff=fixture.kickoff,
        analyzed_at=fixture.kickoff - timedelta(hours=1),
        created_at=(fixture.kickoff - timedelta(hours=1)).replace(tzinfo=None),
    )


def _metrics(
    forecasts: list[tuple[dict[str, float], str]],
    predictions: list[MatchPrediction],
) -> CandidateMetrics:
    brier = BacktestEngine.multiclass_brier_score(forecasts)
    if not predictions:
        return CandidateMetrics(brier, None, 0, len(forecasts), 0)
    backtest = BacktestEngine.run_simulation(
        predictions,
        initial_bankroll=10_000.0,
        strategy="flat",
        flat_stake_amount=10.0,
        min_edge_pct=3.0,
        exclude_post_kickoff=True,
    )
    return CandidateMetrics(
        brier_score=brier,
        roi_pct=float(backtest["total_roi_pct"]),
        backtest_bets=int(backtest["total_bets"]),
        samples=len(forecasts),
        odds_samples=len(predictions),
    )


def _evaluate_statistical(
    samples: list[CalibrationSample],
    odds_by_fixture: dict[int, dict[str, float]],
    *,
    legacy: bool,
    use_global_rho: bool = False,
) -> CandidateMetrics:
    forecasts: list[tuple[dict[str, float], str]] = []
    predictions: list[MatchPrediction] = []
    for sample in samples:
        home, away = _profiles(sample, legacy=legacy)
        analysis = StatsEngine.analyze_match(
            home,
            away,
            league_id=None if use_global_rho else sample.fixture.league_id,
        )
        raw_probabilities = analysis["all_probabilities"]
        probabilities = {
            outcome: float(raw_probabilities[outcome]) / 100.0 for outcome in OUTCOMES
        }
        forecasts.append((probabilities, sample.fixture.actual_result))
        odds = odds_by_fixture.get(sample.fixture.fixture_id)
        if odds is not None:
            predictions.append(_prediction_row(sample.fixture, probabilities, odds))
    return _metrics(forecasts, predictions)


def _evaluate_secondary(
    samples: list[CalibrationSample],
    *,
    market: str,
) -> CandidateMetrics:
    squared_errors: list[float] = []
    for sample in samples:
        home, away = _profiles(sample, legacy=False)
        analysis = StatsEngine.analyze_match(
            home, away, league_id=sample.fixture.league_id
        )
        secondary = {
            str(item["market"]): item for item in analysis["secondary_markets"]
        }
        selected = secondary.get(market)
        if selected is None:
            continue
        probability = float(selected["probability"]) / 100.0
        success = (
            sample.fixture.actual_result != "AWAY_WIN"
            if market == "DOUBLE_CHANCE_1X"
            else sample.fixture.actual_result != "HOME_WIN"
        )
        squared_errors.append((probability - float(success)) ** 2)
    if not squared_errors:
        raise RuntimeError(f"No historical samples for secondary market {market}")
    return CandidateMetrics(
        brier_score=sum(squared_errors) / len(squared_errors),
        roi_pct=None,
        backtest_bets=0,
        samples=len(squared_errors),
        odds_samples=0,
    )


def _evaluate_elo(
    fixtures: list[FixtureRecord],
    odds_by_fixture: dict[int, dict[str, float]],
) -> CandidateMetrics:
    ratings: dict[int, dict[int, float]] = defaultdict(dict)
    active_season: dict[int, int] = {}
    draw_counts: dict[int, tuple[int, int]] = defaultdict(lambda: (0, 0))
    forecasts: list[tuple[dict[str, float], str]] = []
    predictions: list[MatchPrediction] = []

    for fixture in fixtures:
        league_ratings = ratings[fixture.league_id]
        prior_season = active_season.get(fixture.league_id)
        if prior_season is not None and fixture.season != prior_season:
            retention = 1.0 - settings.ELO_SEASON_REGRESSION
            ratings[fixture.league_id] = league_ratings = {
                team_id: 1500.0 + (rating - 1500.0) * retention
                for team_id, rating in league_ratings.items()
            }
        active_season[fixture.league_id] = fixture.season
        home_rating = league_ratings.setdefault(fixture.home_team_id, 1500.0)
        away_rating = league_ratings.setdefault(fixture.away_team_id, 1500.0)
        expected_home_score = 1.0 / (
            1.0
            + 10.0
            ** (
                (away_rating - (home_rating + settings.ELO_HOME_ADVANTAGE_POINTS))
                / 400.0
            )
        )
        draws, played = draw_counts[fixture.league_id]
        draw_probability = min(0.35, max(0.15, (draws + 2.6) / (played + 10)))
        probabilities = {
            "HOME_WIN": max(0.01, expected_home_score - draw_probability / 2.0),
            "DRAW": draw_probability,
            "AWAY_WIN": max(0.01, 1.0 - expected_home_score - draw_probability / 2.0),
        }
        total = sum(probabilities.values())
        probabilities = {
            outcome: value / total for outcome, value in probabilities.items()
        }
        forecasts.append((probabilities, fixture.actual_result))
        odds = odds_by_fixture.get(fixture.fixture_id)
        if odds is not None:
            predictions.append(_prediction_row(fixture, probabilities, odds))

        actual_home = (
            1.0
            if fixture.actual_result == "HOME_WIN"
            else 0.5 if fixture.actual_result == "DRAW" else 0.0
        )
        delta = settings.ELO_K_FACTOR * (actual_home - expected_home_score)
        league_ratings[fixture.home_team_id] = home_rating + delta
        league_ratings[fixture.away_team_id] = away_rating - delta
        draw_counts[fixture.league_id] = (
            draws + int(fixture.actual_result == "DRAW"),
            played + 1,
        )
    return _metrics(forecasts, predictions)


def _parse_odds(row: dict[str, str]) -> dict[str, float] | None:
    column_sets = (
        ("AvgCH", "AvgCD", "AvgCA"),
        ("B365CH", "B365CD", "B365CA"),
        ("PSCH", "PSCD", "PSCA"),
        ("AvgH", "AvgD", "AvgA"),
        ("B365H", "B365D", "B365A"),
    )
    for columns in column_sets:
        try:
            values = [float(str(row.get(column) or "").strip()) for column in columns]
        except ValueError:
            continue
        if all(math.isfinite(value) and value > 1.0 for value in values):
            return dict(zip(OUTCOMES, values, strict=True))
    return None


def _download_football_data_odds(
    *,
    season: int,
    fixture_ids: set[int],
) -> dict[int, dict[str, float]]:
    odds_by_fixture: dict[int, dict[str, float]] = {}
    headers = {"User-Agent": "BetAIPlatform/1.0 calibration-audit"}
    with httpx.Client(
        timeout=settings.FOOTBALL_DATA_TIMEOUT_SECONDS, headers=headers
    ) as client:
        for league in FOOTBALL_DATA_LEAGUES.values():
            if league.rolling_feed:
                continue
            path = FootballDataCSVClient._feed_path(league, season)
            response = client.get(
                f"{settings.FOOTBALL_DATA_BASE_URL.rstrip('/')}{path}"
            )
            if response.status_code == 404:
                continue
            response.raise_for_status()
            content = FootballDataCSVClient._decode(response.content)
            for row in csv.DictReader(io.StringIO(content)):
                if not FootballDataCSVClient._belongs_to_feed(
                    row, league=league, season=season
                ):
                    continue
                home_team = str(row.get("HomeTeam") or "").strip()
                away_team = str(row.get("AwayTeam") or "").strip()
                date_value = str(row.get("Date") or "").strip()
                if not home_team or not away_team or not date_value:
                    continue
                try:
                    kickoff = FootballDataCSVClient._parse_kickoff(
                        date_value,
                        str(row.get("Time") or "").strip(),
                        timezone_name=league.timezone,
                    )
                except ValueError:
                    continue
                home_key = f"{league.country}:{stable_team_name_key(home_team)}"
                away_key = f"{league.country}:{stable_team_name_key(away_team)}"
                natural_key = (
                    f"{league.division}:{season}:{kickoff.isoformat()}:"
                    f"{home_key}:{away_key}"
                )
                fixture_id = _stable_negative_id("football-data-fixture", natural_key)
                if fixture_id not in fixture_ids:
                    continue
                odds = _parse_odds(row)
                if odds is not None:
                    odds_by_fixture[fixture_id] = odds
    return odds_by_fixture


def _candidate_result(
    *,
    name: str,
    factor: float,
    value: object,
    metrics: CandidateMetrics,
) -> dict[str, object]:
    return {
        "name": name,
        "factor": factor,
        "value": _serializable_value(value),
        **asdict(metrics),
    }


def _recommended_factor_range(candidates: list[dict[str, object]]) -> list[float]:
    best_brier = min(float(candidate["brier_score"]) for candidate in candidates)
    tolerance = 0.001
    accepted = [
        float(candidate["factor"])
        for candidate in candidates
        if float(candidate["brier_score"]) <= best_brier + tolerance
    ]
    return [min(accepted), max(accepted)]


def _calibrate_parameter(
    name: str,
    *,
    fixtures: list[FixtureRecord],
    samples: list[CalibrationSample],
    odds_by_fixture: dict[int, dict[str, float]],
) -> dict[str, object]:
    current = getattr(settings, name)
    candidates: list[dict[str, object]] = []
    for factor in GRID_FACTORS:
        value = _candidate_value(current, factor)
        with _override_setting(name, value):
            if name in RICH_PARAMETERS:
                metrics = _evaluate_statistical(
                    samples,
                    odds_by_fixture,
                    legacy=False,
                    use_global_rho=name == "DEFAULT_DIXON_COLES_RHO",
                )
                route = (
                    "global_dixon_coles"
                    if name == "DEFAULT_DIXON_COLES_RHO"
                    else "rich_profile_1x2"
                )
            elif name in LEGACY_PARAMETERS:
                metrics = _evaluate_statistical(samples, odds_by_fixture, legacy=True)
                route = "legacy_profile_1x2"
            elif name in SECONDARY_PARAMETERS:
                market = (
                    "DOUBLE_CHANCE_1X"
                    if name == "DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT"
                    else "DOUBLE_CHANCE_X2"
                )
                metrics = _evaluate_secondary(samples, market=market)
                route = market.lower()
            elif name in ELO_PARAMETERS:
                metrics = _evaluate_elo(fixtures, odds_by_fixture)
                route = "prequential_elo_1x2"
            else:
                raise ValueError(f"Unsupported calibration parameter: {name}")
        candidates.append(
            _candidate_result(
                name=name,
                factor=factor,
                value=value,
                metrics=metrics,
            )
        )
    best = min(candidates, key=lambda candidate: float(candidate["brier_score"]))
    current_result = next(
        candidate for candidate in candidates if candidate["factor"] == 1.0
    )
    return {
        "name": name,
        "status": "validated",
        "route": route,
        "current": _serializable_value(current),
        "recommended_factor_range": _recommended_factor_range(candidates),
        "current_metrics": current_result,
        "best_metrics": best,
        "brier_improvement": round(
            float(current_result["brier_score"]) - float(best["brier_score"]),
            8,
        ),
        "grid": candidates,
    }


def run(
    *,
    with_odds: bool,
    odds_season: int,
    selected_parameters: tuple[str, ...] | None = None,
) -> dict[str, object]:
    fixtures = _load_fixtures()
    if not fixtures:
        raise RuntimeError("No historical fixtures are available")
    samples = _build_samples(
        fixtures,
        recent_match_count=settings.RECENT_FORM_MATCH_COUNT,
        minimum_team_history=settings.HISTORICAL_TRAINING_MIN_TEAM_MATCHES,
    )
    fixture_ids = {sample.fixture.fixture_id for sample in samples}
    odds_by_fixture = (
        _download_football_data_odds(season=odds_season, fixture_ids=fixture_ids)
        if with_odds
        else {}
    )
    inventory = (
        *RICH_PARAMETERS,
        *LEGACY_PARAMETERS,
        *SECONDARY_PARAMETERS,
        *ELO_PARAMETERS,
    )
    unknown = set(selected_parameters or ()) - set(inventory)
    if unknown:
        raise ValueError(f"Unknown calibration parameters: {sorted(unknown)}")
    parameters = selected_parameters or inventory
    results = [
        _calibrate_parameter(
            name,
            fixtures=fixtures,
            samples=samples,
            odds_by_fixture=odds_by_fixture,
        )
        for name in parameters
    ]
    if "GOAL_TIME_DECAY_FACTOR" in parameters and len(samples) >= 100:
        split_index = int(len(samples) * 0.8)
        selection = _calibrate_parameter(
            "GOAL_TIME_DECAY_FACTOR",
            fixtures=fixtures,
            samples=samples[:split_index],
            odds_by_fixture=odds_by_fixture,
        )
        validation = _calibrate_parameter(
            "GOAL_TIME_DECAY_FACTOR",
            fixtures=fixtures,
            samples=samples[split_index:],
            odds_by_fixture=odds_by_fixture,
        )
        selected_factor = selection["best_metrics"]["factor"]
        validation_selected = next(
            candidate
            for candidate in validation["grid"]
            if candidate["factor"] == selected_factor
        )
        validation_current = next(
            candidate for candidate in validation["grid"] if candidate["factor"] == 1.0
        )
        goal_result = next(
            result for result in results if result["name"] == "GOAL_TIME_DECAY_FACTOR"
        )
        goal_result["out_of_time"] = {
            "selection_samples": split_index,
            "validation_samples": len(samples) - split_index,
            "selected_factor": selected_factor,
            "selected_value": selection["best_metrics"]["value"],
            "validation_current": validation_current,
            "validation_selected": validation_selected,
            "validation_brier_improvement": round(
                float(validation_current["brier_score"])
                - float(validation_selected["brier_score"]),
                8,
            ),
        }
    if selected_parameters is None:
        results.extend(
            {
                "name": name,
                "status": "unvalidated",
                "current": _serializable_value(getattr(settings, name)),
                "reason": reason,
            }
            for name, reason in UNVALIDATED_PARAMETERS.items()
        )
    result_names = {str(result["name"]) for result in results}
    expected_names = (
        set(parameters)
        if selected_parameters is not None
        else set(inventory) | set(UNVALIDATED_PARAMETERS)
    )
    expected_count = len(expected_names)
    if result_names != expected_names or len(results) != expected_count:
        raise RuntimeError(
            "Calibration inventory drift: "
            f"expected {expected_count} unique fields, got {len(result_names)}"
        )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "fixtures": len(fixtures),
            "samples": len(samples),
            "league_count": len({fixture.league_id for fixture in fixtures}),
            "seasons": sorted({fixture.season for fixture in fixtures}),
            "start": min(fixture.kickoff for fixture in fixtures).isoformat(),
            "end": max(fixture.kickoff for fixture in fixtures).isoformat(),
            "odds_season": odds_season if with_odds else None,
            "odds_samples": len(odds_by_fixture),
        },
        "method": {
            "grid_factors": list(GRID_FACTORS),
            "brier": "mean multiclass Brier; binary Brier for double chance",
            "roi": (
                "BacktestEngine flat stake, 3% minimum edge, closing-odds proxy"
                if with_odds
                else "not measured"
            ),
            "leakage_control": (
                "point-in-time rolling team windows; simultaneous fixtures share "
                "the same pre-kickoff state"
            ),
        },
        "results": sorted(results, key=lambda result: str(result["name"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ±20% sensitivity analysis for prediction constants."
    )
    parser.add_argument(
        "--with-football-data-odds",
        action="store_true",
        help="Download immutable season CSVs and run the existing ROI backtest.",
    )
    parser.add_argument("--odds-season", type=int, default=2025)
    parser.add_argument(
        "--parameter",
        action="append",
        choices=sorted(
            (
                *RICH_PARAMETERS,
                *LEGACY_PARAMETERS,
                *SECONDARY_PARAMETERS,
                *ELO_PARAMETERS,
            )
        ),
        help="Calibrate only the selected field; may be repeated.",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the JSON report to this path."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print compact per-parameter results without every grid point.",
    )
    args = parser.parse_args()
    if args.summary_only:
        logging.getLogger("bet-ai-pro.backtest").setLevel(logging.WARNING)
    report = run(
        with_odds=args.with_football_data_odds,
        odds_season=args.odds_season,
        selected_parameters=tuple(args.parameter) if args.parameter else None,
    )
    if args.summary_only:
        compact_results: list[dict[str, object]] = []
        for result in report["results"]:
            if result["status"] != "validated":
                compact_results.append(result)
                continue
            current = result["current_metrics"]
            best = result["best_metrics"]
            compact_results.append(
                {
                    "name": result["name"],
                    "status": result["status"],
                    "route": result["route"],
                    "current": result["current"],
                    "recommended_factor_range": result["recommended_factor_range"],
                    "current_brier": current["brier_score"],
                    "current_roi_pct": current["roi_pct"],
                    "best_factor": best["factor"],
                    "best_value": best["value"],
                    "best_brier": best["brier_score"],
                    "best_roi_pct": best["roi_pct"],
                    "samples": current["samples"],
                    "odds_samples": current["odds_samples"],
                    "brier_improvement": result["brier_improvement"],
                    "out_of_time": result.get("out_of_time"),
                }
            )
        report["results"] = compact_results
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
