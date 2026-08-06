"""Data-quality-aware training pipelines for football outcome models.

The dataset builder is deliberately chronological: a fixture is transformed before
its own result and statistics update team state.  This prevents target leakage in
backtests and makes training features available at prediction time.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

Outcome: TypeAlias = Literal[0, 1, 2]
EstimatorBackend: TypeAlias = Literal["lightgbm", "sklearn"]


AWAY_WIN: Outcome = 0
DRAW: Outcome = 1
HOME_WIN: Outcome = 2
TARGET_BY_RESULT: dict[str, Outcome] = {
    "AWAY_WIN": AWAY_WIN,
    "DRAW": DRAW,
    "HOME_WIN": HOME_WIN,
}


@dataclass(frozen=True)
class TierDatasets:
    """Chronologically built feature frames and their 1X2 targets."""

    tier1_features: pd.DataFrame
    tier1_target: pd.Series
    tier2_features: pd.DataFrame
    tier2_target: pd.Series


@dataclass
class _TeamState:
    matches: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    elo: float = 1500.0
    form: deque[int] = field(default_factory=lambda: deque(maxlen=5))
    totals: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def average(self, name: str) -> float:
        count = self.counts[name]
        return self.totals[name] / count if count else 0.0

    @property
    def goals_for_average(self) -> float:
        return self.goals_for / self.matches if self.matches else 0.0

    @property
    def form_average(self) -> float:
        return sum(self.form) / len(self.form) if self.form else 0.0


class MultiTierDatasetBuilder:
    """Build leak-free Tier 1 and Tier 2 samples from historical fixtures."""

    _RICH_SOURCE_COLUMNS = (
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_corners",
        "away_corners",
        "home_fouls",
        "away_fouls",
        "opening_home_odd",
        "opening_draw_odd",
        "opening_away_odd",
        "closing_home_odd",
        "closing_draw_odd",
        "closing_away_odd",
    )

    def build(self, fixtures: Sequence[object]) -> TierDatasets:
        """Split completed fixtures by data richness and build pre-match features."""
        states: dict[tuple[int, str], _TeamState] = defaultdict(_TeamState)
        tier1_rows: list[dict[str, object]] = []
        tier1_targets: list[Outcome] = []
        tier2_rows: list[dict[str, object]] = []
        tier2_targets: list[Outcome] = []

        for fixture in sorted(fixtures, key=self._kickoff_sort_key):
            result = str(self._value(fixture, "actual_result") or "").upper()
            target = TARGET_BY_RESULT.get(result)
            league_id = self._as_int(self._value(fixture, "league_id"))
            home_team = self._team_name(fixture, "home_team")
            away_team = self._team_name(fixture, "away_team")
            if target is None or league_id is None or not home_team or not away_team:
                continue

            home = states[(league_id, home_team)]
            away = states[(league_id, away_team)]
            common = self._common_features(league_id, home_team, away_team, home, away)
            if self._has_rich_data(fixture):
                tier1_rows.append(
                    {**common, **self._tier1_features(fixture, home, away)}
                )
                tier1_targets.append(target)
            else:
                tier2_rows.append(
                    {**common, **self._tier2_features(league_id, home, away, states)}
                )
                tier2_targets.append(target)

            self._update_states(fixture, home, away)

        return TierDatasets(
            tier1_features=pd.DataFrame(tier1_rows, columns=Tier1Model.FEATURES),
            tier1_target=pd.Series(tier1_targets, dtype="int64", name="target"),
            tier2_features=pd.DataFrame(tier2_rows, columns=Tier2Model.FEATURES),
            tier2_target=pd.Series(tier2_targets, dtype="int64", name="target"),
        )

    @staticmethod
    def _value(fixture: object, name: str) -> object:
        if isinstance(fixture, Mapping):
            return fixture.get(name)
        return getattr(fixture, name, None)

    @classmethod
    def _kickoff_sort_key(cls, fixture: object) -> int:
        value = cls._value(fixture, "kickoff")
        try:
            return int(pd.Timestamp(value).value)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _has_rich_data(cls, fixture: object) -> bool:
        return all(
            cls._valid_number(cls._value(fixture, name))
            for name in cls._RICH_SOURCE_COLUMNS
        )

    @staticmethod
    def _valid_number(value: object) -> bool:
        return MultiTierDatasetBuilder._as_float(value) is not None

    @staticmethod
    def _as_float(value: object) -> float | None:
        try:
            parsed = float(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    @staticmethod
    def _as_int(value: object) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _team_name(cls, fixture: object, field_name: str) -> str:
        return str(cls._value(fixture, field_name) or "").strip()

    @staticmethod
    def _common_features(
        league_id: int,
        home_team: str,
        away_team: str,
        home: _TeamState,
        away: _TeamState,
    ) -> dict[str, object]:
        return {
            "league_id": str(league_id),
            "home_team": home_team,
            "away_team": away_team,
            "home_form_last5": home.form_average,
            "away_form_last5": away.form_average,
        }

    @classmethod
    def _tier1_features(
        cls, fixture: object, home: _TeamState, away: _TeamState
    ) -> dict[str, object]:
        odds: dict[str, float] = {}
        for name in (
            "opening_home_odd",
            "opening_draw_odd",
            "opening_away_odd",
            "closing_home_odd",
            "closing_draw_odd",
            "closing_away_odd",
        ):
            value = cls._as_float(cls._value(fixture, name))
            if value is None:  # Protected by _has_rich_data; preserve the invariant.
                raise ValueError(f"Tier 1 fixture is missing {name}")
            odds[name] = value
        return {
            "home_avg_shots": home.average("shots"),
            "away_avg_shots": away.average("shots"),
            "home_avg_shots_on_target": home.average("shots_on_target"),
            "away_avg_shots_on_target": away.average("shots_on_target"),
            "home_avg_corners": home.average("corners"),
            "away_avg_corners": away.average("corners"),
            "home_avg_fouls": home.average("fouls"),
            "away_avg_fouls": away.average("fouls"),
            **odds,
        }

    @staticmethod
    def _tier2_features(
        league_id: int,
        home: _TeamState,
        away: _TeamState,
        states: Mapping[tuple[int, str], _TeamState],
    ) -> dict[str, object]:
        ranks = MultiTierDatasetBuilder._league_ranks(league_id, states)
        return {
            "home_avg_goals": home.goals_for_average,
            "away_avg_goals": away.goals_for_average,
            "home_league_points": float(home.points),
            "away_league_points": float(away.points),
            "home_league_position": float(ranks.get(id(home), len(ranks) + 1)),
            "away_league_position": float(ranks.get(id(away), len(ranks) + 1)),
            "home_elo": home.elo,
            "away_elo": away.elo,
        }

    @staticmethod
    def _league_ranks(
        league_id: int, states: Mapping[tuple[int, str], _TeamState]
    ) -> dict[int, int]:
        league_states = [
            state
            for (current_league, _), state in states.items()
            if current_league == league_id
        ]
        ordered = sorted(
            league_states,
            key=lambda state: (
                state.points,
                state.goals_for - state.goals_against,
                state.goals_for,
            ),
            reverse=True,
        )
        return {id(state): position for position, state in enumerate(ordered, start=1)}

    @classmethod
    def _update_states(
        cls, fixture: object, home: _TeamState, away: _TeamState
    ) -> None:
        home_goals = cls._as_int(cls._value(fixture, "home_goals"))
        away_goals = cls._as_int(cls._value(fixture, "away_goals"))
        if home_goals is None or away_goals is None:
            return

        home_points, away_points, score = (
            (3, 0, 1.0) if home_goals > away_goals else (0, 3, 0.0)
        )
        if home_goals == away_goals:
            home_points, away_points, score = 1, 1, 0.5
        cls._update_team(
            home, goals_for=home_goals, goals_against=away_goals, points=home_points
        )
        cls._update_team(
            away, goals_for=away_goals, goals_against=home_goals, points=away_points
        )

        expected_home = 1.0 / (1.0 + 10.0 ** ((away.elo - home.elo) / 400.0))
        adjustment = 24.0 * (score - expected_home)
        home.elo += adjustment
        away.elo -= adjustment
        cls._update_optional_stats(fixture, home, away)

    @staticmethod
    def _update_team(
        state: _TeamState, *, goals_for: int, goals_against: int, points: int
    ) -> None:
        state.matches += 1
        state.goals_for += goals_for
        state.goals_against += goals_against
        state.points += points
        state.form.append(points)

    @classmethod
    def _update_optional_stats(
        cls, fixture: object, home: _TeamState, away: _TeamState
    ) -> None:
        columns = {
            "shots": ("home_shots", "away_shots"),
            "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
            "corners": ("home_corners", "away_corners"),
            "fouls": ("home_fouls", "away_fouls"),
        }
        for statistic, (home_column, away_column) in columns.items():
            for state, column in ((home, home_column), (away, away_column)):
                value = cls._value(fixture, column)
                parsed = cls._as_float(value)
                if parsed is not None:
                    state.totals[statistic] += parsed
                    state.counts[statistic] += 1


class _BaseTierModel:
    """Shared fit, probability-alignment and evaluation mechanics."""

    FEATURES: tuple[str, ...] = ()
    CATEGORICAL_FEATURES: tuple[str, ...] = ("league_id", "home_team", "away_team")

    def __init__(
        self, *, backend: EstimatorBackend = "lightgbm", random_state: int = 42
    ) -> None:
        self.backend = backend
        self.random_state = random_state
        self.pipeline: Pipeline | None = None

    def train(self, X: pd.DataFrame, y: Sequence[int] | pd.Series) -> "_BaseTierModel":
        frame, target = self._validate_training_data(X, y)
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self._preprocessor()),
                ("classifier", self._classifier()),
            ]
        )
        self.pipeline.fit(frame, target)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model must be trained before predict_proba is called")
        frame = self._validate_features(X)
        probabilities = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
        classes = np.asarray(
            self.pipeline.named_steps["classifier"].classes_, dtype=int
        )
        aligned = np.zeros((len(frame), 3), dtype=float)
        aligned[:, classes] = probabilities
        return aligned

    def evaluate(
        self, X_test: pd.DataFrame, y_test: Sequence[int] | pd.Series
    ) -> dict[str, float | int]:
        target = self._validate_target(y_test, expected_length=len(X_test))
        probabilities = self.predict_proba(X_test)
        predictions = probabilities.argmax(axis=1)
        return {
            "samples": len(target),
            "accuracy": float(accuracy_score(target, predictions)),
            "f1_macro": float(
                f1_score(
                    target,
                    predictions,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
            "log_loss": float(log_loss(target, probabilities, labels=[0, 1, 2])),
        }

    def _preprocessor(self) -> ColumnTransformer:
        numeric = [
            name for name in self.FEATURES if name not in self.CATEGORICAL_FEATURES
        ]
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", RobustScaler()),
                        ]
                    ),
                    numeric,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    list(self.CATEGORICAL_FEATURES),
                ),
            ],
            remainder="drop",
        )

    def _classifier(self) -> LGBMClassifier | HistGradientBoostingClassifier:
        if self.backend == "lightgbm":
            return LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=250,
                learning_rate=0.04,
                max_depth=-1,
                num_leaves=24,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=2,
                verbosity=-1,
            )
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=200,
            max_leaf_nodes=20,
            random_state=self.random_state,
        )

    def _validate_training_data(
        self, X: pd.DataFrame, y: Sequence[int] | pd.Series
    ) -> tuple[pd.DataFrame, np.ndarray]:
        frame = self._validate_features(X)
        target = self._validate_target(y, expected_length=len(frame))
        if len(np.unique(target)) != 3:
            raise ValueError(
                "Training data must include all three outcome classes: 0, 1 and 2"
            )
        return frame, target

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        missing = set(self.FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        return X.loc[:, self.FEATURES].copy()

    @staticmethod
    def _validate_target(
        y: Sequence[int] | pd.Series, *, expected_length: int
    ) -> np.ndarray:
        target = np.asarray(y, dtype=int)
        if target.ndim != 1 or len(target) != expected_length:
            raise ValueError("y must be one-dimensional and have the same length as X")
        if not np.isin(target, [AWAY_WIN, DRAW, HOME_WIN]).all():
            raise ValueError("Target values must be 0 (away), 1 (draw), or 2 (home)")
        return target


class Tier1Model(_BaseTierModel):
    """LightGBM model for fixtures with statistics and pre-match odds."""

    FEATURES = (
        "league_id",
        "home_team",
        "away_team",
        "home_avg_shots",
        "away_avg_shots",
        "home_avg_shots_on_target",
        "away_avg_shots_on_target",
        "home_avg_corners",
        "away_avg_corners",
        "home_avg_fouls",
        "away_avg_fouls",
        "home_form_last5",
        "away_form_last5",
        "opening_home_odd",
        "opening_draw_odd",
        "opening_away_odd",
        "closing_home_odd",
        "closing_draw_odd",
        "closing_away_odd",
    )


class Tier2Model(_BaseTierModel):
    """LightGBM model for result-only fixtures with standings and Elo context."""

    FEATURES = (
        "league_id",
        "home_team",
        "away_team",
        "home_avg_goals",
        "away_avg_goals",
        "home_form_last5",
        "away_form_last5",
        "home_league_points",
        "away_league_points",
        "home_league_position",
        "away_league_position",
        "home_elo",
        "away_elo",
    )
