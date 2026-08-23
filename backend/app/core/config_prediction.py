"""Prediction engine calibration and scoring parameters."""

from pydantic import Field


class PredictionFields:
    """Statistical model tuning: Poisson, Dixon-Coles, form, ELO, fatigue, player impact."""

    LEAGUE_BASELINE_GOALS: float = Field(default=1.32, gt=0)
    GOAL_TIME_DECAY_FACTOR: float = Field(default=0.008, ge=0, le=1, allow_inf_nan=False)
    FORM_DECAY_WEIGHTS: tuple[float, ...] = Field(default=(1.0, 0.88, 0.76, 0.64, 0.52))
    FORM_DECAY_FALLBACK_WEIGHT: float = Field(default=0.4, ge=0)
    HOME_ATTACK_BOOST: float = Field(default=1.11, gt=0)
    AWAY_ATTACK_PENALTY: float = Field(default=0.93, gt=0)
    STRENGTH_ATTACK_WEIGHT: float = Field(default=0.4, ge=0)
    STRENGTH_DEFENSE_WEIGHT: float = Field(default=0.35, ge=0)
    STRENGTH_FORM_WEIGHT: float = Field(default=0.25, ge=0)
    XG_OBSERVED_GOALS_WEIGHT: float = Field(default=0.55, ge=0)
    XG_ATTACK_BASELINE_WEIGHT: float = Field(default=0.45, ge=0)
    XG_CONSISTENCY_MAX_PENALTY: float = Field(default=0.12, ge=0)
    XG_CONSISTENCY_PENALTY_WEIGHT: float = Field(default=0.04, ge=0)
    PROFILE_FORM_FACTOR_BASE: float = Field(default=0.88, ge=0)
    PROFILE_FORM_FACTOR_WEIGHT: float = Field(default=0.24, ge=0)
    LEGACY_ATTACK_FACTOR_BASE: float = Field(default=0.62, ge=0)
    LEGACY_ATTACK_FACTOR_WEIGHT: float = Field(default=0.78, ge=0)
    LEGACY_DEFENSE_FACTOR_BASE: float = Field(default=0.72, ge=0)
    LEGACY_DEFENSE_FACTOR_WEIGHT: float = Field(default=0.55, ge=0)
    LEGACY_FORM_FACTOR_BASE: float = Field(default=0.82, ge=0)
    LEGACY_FORM_FACTOR_WEIGHT: float = Field(default=0.36, ge=0)
    LEGACY_XG_OBSERVED_WEIGHT: float = Field(default=0.58, ge=0)
    LEGACY_XG_BASELINE_WEIGHT: float = Field(default=0.42, ge=0)
    HOME_ADVANTAGE_MIN_MULTIPLIER: float = Field(default=0.88, gt=0)
    HOME_ADVANTAGE_MAX_MULTIPLIER: float = Field(default=1.22, gt=0)
    HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR: float = Field(default=0.55, gt=0)
    HOME_FORM_BASE_MULTIPLIER: float = Field(default=1.08, gt=0)
    HOME_FORM_BOOST_DIVISOR: float = Field(default=450.0, gt=0)
    DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT: float = Field(default=12.0, ge=0)
    DOUBLE_CHANCE_AWAY_DIFFERENCE_WEIGHT: float = Field(default=14.0, ge=0)
    DEFAULT_DIXON_COLES_RHO: float = Field(default=-0.12)
    LEAGUE_DIXON_COLES_RHO: dict[int, float] = Field(
        default={
            39: -0.13, 140: -0.11, 135: -0.15, 78: -0.09, 61: -0.12,
            40: -0.14, 94: -0.12, 203: -0.10, 88: -0.08, 144: -0.12,
            235: -0.11, 79: -0.10, 136: -0.14, 62: -0.13,
        }
    )
    RECENT_FORM_MATCH_COUNT: int = Field(default=5, ge=1, le=20)
    HISTORICAL_FORM_MAX_AGE_DAYS: int = Field(default=365, ge=1, le=365)
    PLAYER_IMPACT_MIN_RATED_STARTERS: int = Field(default=7, ge=1, le=11)
    PLAYER_IMPACT_LOOKBACK_MATCHES: int = Field(default=10, ge=1, le=50)
    PLAYER_IMPACT_RATING_DECAY: float = Field(default=0.85, gt=0, le=1, allow_inf_nan=False)
    PLAYER_IMPACT_REPLACEMENT_FACTOR: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    PLAYER_IMPACT_MIN_STRENGTH_RATIO: float = Field(default=0.70, gt=0, le=1, allow_inf_nan=False)
    PLAYER_IMPACT_MAX_STRENGTH_RATIO: float = Field(default=1.05, ge=1, le=1.25, allow_inf_nan=False)
    PLAYER_IMPACT_XG_ELASTICITY: float = Field(default=1.0, gt=0, le=3, allow_inf_nan=False)
    PLAYER_IMPACT_MIN_XG_MULTIPLIER: float = Field(default=0.75, gt=0, le=1, allow_inf_nan=False)
    PLAYER_CRITICAL_ABSENCE_WEIGHT: float = Field(default=0.25, ge=0, le=1, allow_inf_nan=False)
    PLAYER_QUESTIONABLE_ABSENCE_WEIGHT: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    PLAYER_CONTEXT_SYNC_MAX_FIXTURES: int = Field(default=20, ge=0, le=500)
    PLAYER_CONTEXT_SYNC_CONCURRENCY: int = Field(default=3, ge=1, le=20)
    FATIGUE_LOOKBACK_DAYS: int = Field(default=14, ge=1, le=60)
    FATIGUE_MATCH_REFERENCE_COUNT: int = Field(default=4, ge=1, le=20)
    FATIGUE_IDEAL_REST_DAYS: float = Field(default=7.0, gt=0, le=30, allow_inf_nan=False)
    FATIGUE_TRAVEL_REFERENCE_KM: float = Field(default=3000.0, gt=0, le=20000, allow_inf_nan=False)
    FATIGUE_MATCH_WEIGHT: float = Field(default=0.45, ge=0, le=1, allow_inf_nan=False)
    FATIGUE_REST_WEIGHT: float = Field(default=0.40, ge=0, le=1, allow_inf_nan=False)
    FATIGUE_TRAVEL_WEIGHT: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    ELO_K_FACTOR: float = Field(default=32.0, gt=0, le=100)
    ELO_HOME_ADVANTAGE_POINTS: float = Field(default=65.0, ge=0, le=200)
    ELO_SEASON_REGRESSION: float = Field(default=0.25, ge=0, le=1)
