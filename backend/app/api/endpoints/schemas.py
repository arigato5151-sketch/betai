from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.prediction.input_catalog import AnalysisInputCatalog
from app.prediction.ml.ml_pipeline import Tier1Model, Tier2Model


class PlatformStatusResponse(BaseModel):
    api_mode: str
    registration_enabled: bool


class MLRuntimeStatus(BaseModel):
    inference_success: int = Field(ge=0)
    inference_failure: int = Field(ge=0)


class MLTrainingDataStatus(BaseModel):
    labeled_predictions: int = Field(ge=0)
    historical_fixtures: int = Field(ge=0)
    minimum_samples: int = Field(ge=1)
    historical_minimum_team_matches: int = Field(ge=1)


class MLMonitoringStatus(BaseModel):
    status: Literal["model_unavailable", "insufficient_data", "stable", "drift"]
    drift_detected: bool
    samples: int = Field(ge=0)
    required_samples: int = Field(ge=1)
    window_size: int = Field(ge=0)
    recent_brier: float | None
    baseline_brier: float | None
    brier_delta: float | None
    brier_delta_lower_bound: float | None
    threshold: float = Field(gt=0)
    confidence: float = Field(ge=0, lt=1)
    artifact_version: str | None
    metric_source: Literal["ml_component"]
    ordering: Literal["kickoff_desc"]
    evaluated_at: datetime


class MLLiveEvaluationStatus(BaseModel):
    status: Literal["model_unavailable", "insufficient_data", "available"]
    verified_samples: int = Field(ge=0)
    required_samples: int = Field(ge=1)
    claims_enabled: bool
    artifact_version: str | None


class MLStatusResponse(BaseModel):
    ready: bool
    model_name: str | None
    artifact_version: str | None
    metrics: dict[str, Any]
    runtime: MLRuntimeStatus
    rollback_available: bool
    training_data: MLTrainingDataStatus
    monitoring: MLMonitoringStatus
    live_evaluation: MLLiveEvaluationStatus


class HistoricalLeagueCoverageStatus(BaseModel):
    league_id: int
    league_name: str
    fixtures: int = Field(ge=0)
    latest_kickoff: datetime | None
    expected_minimum_fixtures: int = Field(ge=0)
    available: bool
    fresh: bool
    covered: bool


class HistoricalDataQualityStatus(BaseModel):
    fixtures: int = Field(ge=0)
    leagues: int = Field(ge=0)
    seasons: int = Field(ge=0)
    oldest_kickoff: datetime | None
    newest_kickoff: datetime | None
    last_updated: datetime | None
    freshness_hours: float | None = Field(default=None, ge=0)
    lineup_coverage_pct: float = Field(ge=0, le=100)
    source_counts: dict[str, int]
    current_season: int
    current_season_coverage: list[HistoricalLeagueCoverageStatus]
    current_season_covered_leagues: int = Field(ge=0)
    current_season_missing_league_ids: list[int]


class PredictionDataQualityStatus(BaseModel):
    total: int = Field(ge=0)
    excluded_from_training: int = Field(ge=0)
    quarantined_results: int = Field(ge=0)
    labeled: int = Field(ge=0)
    labeled_coverage_pct: float = Field(ge=0, le=100)
    closing_odds_coverage_pct: float = Field(ge=0, le=100)
    provenance_coverage_pct: float = Field(ge=0, le=100)


class SyncRunStatus(BaseModel):
    job_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    fixtures_processed: int = Field(ge=0)
    failures: list[dict[str, Any]]
    error_type: str | None


class ProviderHealthStatus(BaseModel):
    status: str
    enabled: bool | None = None
    provider: str | None = None
    circuit_open_until: datetime | None = None
    consecutive_failures: int | None = Field(default=None, ge=0)
    daily_limit: int | None = Field(default=None, ge=0)
    daily_remaining: int | None = Field(default=None, ge=0)
    minute_limit: int | None = Field(default=None, ge=0)
    minute_remaining: int | None = Field(default=None, ge=0)
    last_status_code: int | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error: str | None = None
    updated_at: datetime | None = None


class DataQualityProvidersStatus(BaseModel):
    api_football: ProviderHealthStatus
    sportmonks: ProviderHealthStatus


class DataQualityResponse(BaseModel):
    status: Literal["healthy", "warning", "critical"]
    score: float = Field(ge=0, le=100)
    historical: HistoricalDataQualityStatus
    predictions: PredictionDataQualityStatus
    latest_sync: SyncRunStatus | None
    providers: DataQualityProvidersStatus


class RegistrationRequest(BaseModel):
    username: str = Field(
        ..., min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=12, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        local, separator, domain = normalized.partition("@")
        if not separator or not local or "." not in domain:
            raise ValueError("Geçerli bir e-posta adresi girin.")
        return normalized


class TeamStatsInput(BaseModel):
    form: float = Field(..., ge=0, le=100, description="Team form (0-100)")
    attack: float = Field(..., ge=0, le=100, description="Attack strength (0-100)")
    defense: float = Field(..., ge=0, le=100, description="Defense strength (0-100)")
    xg: float = Field(..., ge=0, le=5, description="Expected goals (0-5)")

    @field_validator("form", "attack", "defense")
    @classmethod
    def validate_stats(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("Must be numeric")
        return round(float(v), 2)

    @field_validator("xg")
    @classmethod
    def validate_xg(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("xG must be numeric")
        return round(float(v), 3)


class Odds1X2Input(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    home_win: float = Field(..., alias="HOME_WIN", gt=1.0, le=1000.0)
    draw: float = Field(..., alias="DRAW", gt=1.0, le=1000.0)
    away_win: float = Field(..., alias="AWAY_WIN", gt=1.0, le=1000.0)

    def as_outcome_dict(self) -> Dict[str, float]:
        return {
            "HOME_WIN": self.home_win,
            "DRAW": self.draw,
            "AWAY_WIN": self.away_win,
        }


class AnalysisRequest(BaseModel):
    home_team: str = Field(
        ..., min_length=1, max_length=100, description="Home team name"
    )
    away_team: str = Field(
        ..., min_length=1, max_length=100, description="Away team name"
    )
    home_stats: TeamStatsInput
    away_stats: TeamStatsInput
    odd: float = Field(..., gt=1.0, le=1000.0, description="Betting odd (>1.0)")
    market_1x2: Optional[Dict[str, Any]] = None
    opening_odds_1x2: Optional[Odds1X2Input] = None
    current_odds_1x2: Optional[Odds1X2Input] = None
    opening_odds_at: Optional[datetime] = None
    current_odds_at: Optional[datetime] = None
    fixture_id: Optional[int] = Field(None, gt=0, description="API Football fixture ID")
    fixture_source: Optional[str] = Field(None, min_length=1, max_length=50)
    provider_fixture_id: Optional[str] = Field(None, min_length=1, max_length=100)
    home_team_id: Optional[int] = Field(
        None, gt=0, description="API Football home team ID"
    )
    away_team_id: Optional[int] = Field(
        None, gt=0, description="API Football away team ID"
    )
    league_id: Optional[int] = Field(None, gt=0, description="API Football league ID")
    season: Optional[int] = Field(
        None, ge=2000, le=2100, description="API Football league season"
    )
    kickoff: Optional[datetime] = Field(
        None, description="Fixture kickoff used for point-in-time rest features"
    )
    away_travel_distance_km: Optional[float] = Field(
        None,
        ge=0,
        le=20000,
        allow_inf_nan=False,
        description=(
            "Optional away-team base-to-venue distance. Server-side team locations "
            "are used when omitted."
        ),
    )
    feature_overrides: Dict[str, float] = Field(default_factory=dict)
    kelly_fraction: float = Field(
        0.25, gt=0, le=1.0, description="Fractional Kelly stake multiplier (0-1)"
    )

    @field_validator("home_team", "away_team")
    @classmethod
    def validate_team_names(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Team name cannot be empty")
        return v.strip()[:100]

    @field_validator("odd")
    @classmethod
    def validate_odd(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 1.0:
            raise ValueError("Odd must be numeric and > 1.0")
        return round(float(v), 3)

    @field_validator("feature_overrides", mode="before")
    @classmethod
    def validate_feature_overrides(cls, value: object) -> Dict[str, float]:
        return AnalysisInputCatalog.validate_overrides(value)

    @model_validator(mode="after")
    def validate_odds_snapshot_timeline(self) -> "AnalysisRequest":
        snapshots = (
            ("opening", self.opening_odds_1x2, self.opening_odds_at),
            ("current", self.current_odds_1x2, self.current_odds_at),
        )
        for label, odds, captured_at in snapshots:
            if (odds is None) != (captured_at is None):
                raise ValueError(
                    f"{label}_odds_1x2 and {label}_odds_at must be provided together"
                )

        supplied_timestamps = [
            captured_at for _, _, captured_at in snapshots if captured_at is not None
        ]
        if not supplied_timestamps:
            return self
        if self.kickoff is None:
            raise ValueError("kickoff is required when odds snapshots are provided")
        if self.kickoff.utcoffset() is None:
            raise ValueError("kickoff must include a timezone for odds validation")
        if any(captured_at.utcoffset() is None for captured_at in supplied_timestamps):
            raise ValueError("odds snapshot timestamps must include a timezone")

        kickoff_utc = self.kickoff.astimezone(timezone.utc)
        if any(
            captured_at.astimezone(timezone.utc) >= kickoff_utc
            for captured_at in supplied_timestamps
        ):
            raise ValueError("odds snapshots must be captured before kickoff")
        if (
            self.opening_odds_at is not None
            and self.current_odds_at is not None
            and self.opening_odds_at.astimezone(timezone.utc)
            > self.current_odds_at.astimezone(timezone.utc)
        ):
            raise ValueError("opening odds timestamp cannot follow current odds")
        return self


class TieredPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    league_id: int = Field(..., gt=0)
    features: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if "league_id" in value:
            raise ValueError("league_id must be supplied only as the top-level field")
        allowed = (set(Tier1Model.FEATURES) | set(Tier2Model.FEATURES)) - {"league_id"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unsupported tier features: {sorted(unknown)}")
        tier1_required = set(Tier1Model.FEATURES) - {"league_id"}
        tier2_required = set(Tier2Model.FEATURES) - {"league_id"}
        if not tier1_required.issubset(value) and not tier2_required.issubset(value):
            raise ValueError("A complete Tier 1 or Tier 2 feature contract is required")

        for name, feature in value.items():
            if name in {"home_team", "away_team"}:
                if not isinstance(feature, str) or not feature.strip():
                    raise ValueError(f"Feature {name} must be a non-empty string")
                value[name] = feature.strip()[:100]
                continue
            if isinstance(feature, bool) or not isinstance(feature, (int, float)):
                raise ValueError(f"Feature {name} must be numeric")
            if not math.isfinite(float(feature)):
                raise ValueError(f"Feature {name} must be finite")
            if name.startswith("opening_") and not 1.0 < float(feature) <= 1000.0:
                raise ValueError(f"Feature {name} must be a valid decimal odd")
        return value


class ActualResultUpdate(BaseModel):
    actual_result: str = Field(
        ..., pattern="^(HOME_WIN|DRAW|AWAY_WIN)$", description="Match result"
    )
    actual_score_home: Optional[int] = Field(
        None, ge=0, le=50, description="Home team goals"
    )
    actual_score_away: Optional[int] = Field(
        None, ge=0, le=50, description="Away team goals"
    )

    @field_validator("actual_result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        if v not in {"HOME_WIN", "DRAW", "AWAY_WIN"}:
            raise ValueError("Result must be HOME_WIN, DRAW, or AWAY_WIN")
        return v


class BacktestRequest(BaseModel):
    initial_bankroll: float = Field(
        1000.0, gt=0, le=1000000, description="Starting bankroll"
    )
    strategy: str = Field(
        "kelly",
        pattern="^(kelly|flat|fractional_kelly)$",
        description="Betting strategy",
    )
    flat_stake_amount: float = Field(10.0, gt=0, le=100000, description="Flat bet size")
    kelly_fraction: float = Field(
        0.25, gt=0, le=1.0, description="Kelly fraction (0-1)"
    )
    min_edge_pct: float = Field(3.0, ge=0, le=100, description="Minimum edge% to bet")
    commission_pct: float = Field(
        0.0, ge=0, le=20, description="Commission deducted from winning profit"
    )
    max_stake_pct: float = Field(
        5.0, gt=0, le=100, description="Maximum bankroll percentage per bet"
    )
    max_daily_exposure_pct: float = Field(
        15.0, gt=0, le=100, description="Maximum daily bankroll exposure"
    )
    require_closing_odds: bool = Field(
        True, description="Settle at closing price and skip records without one"
    )
    exclude_post_kickoff: bool = Field(
        True, description="Exclude analyses generated at or after kickoff"
    )
    limit: int = Field(
        5000, ge=1, le=50000, description="Maximum predictions to include"
    )

    @field_validator("initial_bankroll")
    @classmethod
    def validate_bankroll(cls, v: float) -> float:
        if v < 10:
            raise ValueError("Bankroll must be at least 10")
        return round(float(v), 2)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"kelly", "flat", "fractional_kelly"}
        if v not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        return v
