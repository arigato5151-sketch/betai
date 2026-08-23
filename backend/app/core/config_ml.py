"""ML model training, calibration, and inference configuration."""

import os

from pydantic import Field


class MLFields:
    """ML pipeline, calibration, drift monitoring, ensemble, and model artifact settings."""

    MIN_TRAINING_SAMPLES: int = 200
    RETRAIN_EVERY_N_NEW: int = 25
    ML_TRAINING_MAX_HISTORICAL_FIXTURES: int = Field(default=5000, ge=200, le=100000)
    ML_TRAINING_MAX_PLAYER_PERFORMANCES: int = Field(
        default=150000, ge=1000, le=1000000
    )
    ENABLE_CATBOOST_CANDIDATE: bool = True
    ENABLE_LIGHTGBM_CANDIDATE: bool = True
    ML_BOOSTER_TREES: int = Field(default=200, ge=50, le=2000)
    ML_BOOSTER_MAX_DEPTH: int = Field(default=6, ge=2, le=12)
    ML_BOOSTER_LEARNING_RATE: float = Field(
        default=0.05, gt=0, le=1, allow_inf_nan=False
    )
    ML_BOOSTER_THREADS: int = Field(default=2, ge=1, le=32)
    HISTORICAL_TRAINING_MIN_TEAM_MATCHES: int = Field(default=3, ge=1, le=10)
    MIN_MODEL_BASELINE_BRIER_IMPROVEMENT: float = Field(default=0.005, ge=0, le=1)
    MAX_MODEL_BASELINE_LOG_LOSS_REGRESSION: float = Field(default=0.0, ge=0, le=1)
    MIN_MODEL_CHAMPION_BRIER_IMPROVEMENT: float = Field(default=0.001, ge=0, le=1)
    MIN_MODEL_CHAMPION_LOG_LOSS_IMPROVEMENT: float = Field(default=0.01, ge=0, le=1)
    MAX_MODEL_CHAMPION_BRIER_REGRESSION: float = Field(default=0.01, ge=0, le=1)
    MIN_CALIBRATION_LOG_LOSS_IMPROVEMENT: float = Field(default=0.001, ge=0, le=1)
    MIN_ISOTONIC_CALIBRATION_SAMPLES: int = Field(default=500, ge=30)
    TIERED_CALIBRATION_METHOD: str = Field(
        default="auto",
        pattern="^(isotonic|platt|auto|none)$",
        description=(
            "Probability calibration for tiered 1X2 models: isotonic, platt, "
            "auto (best of both on a temporal holdout), or none."
        ),
    )
    MIN_TIERED_CALIBRATION_SAMPLES: int = Field(default=60, ge=30)
    TIERED_CALIBRATION_HOLDOUT_FRACTION: float = Field(default=0.15, ge=0.05, le=0.4)
    TIERED_PROMOTION_MIN_HOLDOUT_SAMPLES: int = Field(default=30, ge=10, le=10000)
    TIERED_PROMOTION_BOOTSTRAP_SAMPLES: int = Field(default=2000, ge=200, le=20000)
    TIERED_PROMOTION_CONFIDENCE: float = Field(default=0.95, ge=0.8, lt=1)
    MIN_TIERED_MARKET_LOG_LOSS_IMPROVEMENT: float = Field(default=0.005, ge=0, le=1)
    MIN_TIERED_MARKET_BRIER_IMPROVEMENT: float = Field(default=0.002, ge=0, le=1)
    MIN_TIERED_CHAMPION_LOG_LOSS_IMPROVEMENT: float = Field(default=0.002, ge=0, le=1)
    MIN_TIERED_CHAMPION_BRIER_IMPROVEMENT: float = Field(default=0.001, ge=0, le=1)
    MAX_TIERED_CHAMPION_LOG_LOSS_REGRESSION: float = Field(default=0.002, ge=0, le=1)
    MAX_TIERED_CHAMPION_BRIER_REGRESSION: float = Field(default=0.001, ge=0, le=1)
    TIERED_MIN_TIER2_SAMPLES: int = Field(default=500, ge=30, le=100000)
    TIERED_MIN_TIER2_SAMPLES_PER_CLASS: int = Field(default=20, ge=5, le=10000)
    KELLY_FRACTION: float = Field(default=0.25, gt=0, le=1)
    MAX_MODEL_BRIER_SCORE: float = Field(default=0.8, gt=0, le=2)
    MAX_MODEL_LOG_LOSS: float = Field(default=1.5, gt=0)
    MAX_MODEL_CALIBRATION_ERROR: float = Field(default=0.2, ge=0, le=1)
    MODEL_ARTIFACTS_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "artifacts",
        "models",
    )
    ACTIVE_MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "artifacts",
        "models",
        "active_model.pkl",
    )
    ENSEMBLE_WEIGHTS_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "artifacts",
        "models",
        "ensemble_weights.json",
    )
    MODEL_DRIFT_WINDOW_SIZE: int = Field(default=100, ge=20, le=2000)
    MODEL_DRIFT_MIN_SAMPLES: int = Field(default=30, ge=10, le=1000)
    MODEL_DRIFT_BRIER_THRESHOLD: float = Field(default=0.04, gt=0, le=1)
    MODEL_DRIFT_TOTAL_DEVIATION_THRESHOLD: float = Field(default=0.10, gt=0, le=2)
    MODEL_DRIFT_BOOTSTRAP_SAMPLES: int = Field(default=2000, ge=200, le=20000)
    MODEL_DRIFT_CONFIDENCE: float = Field(default=0.95, ge=0.8, lt=1)
    MODEL_DRIFT_RETRAIN_COOLDOWN_SECONDS: int = Field(default=259200, ge=3600, le=2592000)
    AUDIT_MIN_RELIABLE_SAMPLES: int = Field(default=30, ge=10, le=1000)
    AUDIT_BOOTSTRAP_ITERATIONS: int = Field(default=2000, ge=200, le=10000)
    AUDIT_MIN_CLOSING_SAMPLES: int = Field(default=200, ge=30, le=10000)
    AUDIT_MIN_CLOSING_ROI_PCT: float = Field(default=2.0, ge=0.0, le=100.0)
    AUDIT_MAX_CLOSING_ODDS_AGE_HOURS: float = Field(default=8.0, gt=0.0, le=72.0)
    AUDIT_CLOSING_ROI_EROSION_DELTA_PCT: float = Field(default=5.0, ge=0.0, le=100.0)
    FINANCIAL_RECOMMENDATIONS_ENABLED: bool = False
    DECISION_MIN_TOP_PROBABILITY_PCT: float = Field(default=35.0, ge=0, le=100)
    DECISION_MIN_MARGIN_PCT: float = Field(default=3.0, ge=0, le=100)
    DECISION_MAX_NORMALIZED_ENTROPY: float = Field(default=0.99, ge=0, le=1)
    DECISION_MAX_SOURCE_JSD: float = Field(default=0.15, ge=0, le=1)
    DECISION_SOURCE_DIVERGENCE_MAX_MARGIN_PCT: float = Field(default=10.0, ge=0, le=100)
    DECISION_CONDITIONAL_TOP_PROBABILITY_PCT: float = Field(
        default=30.0, ge=0, le=100,
        description="Minimum top probability for conditional status when market confirms",
    )
    DECISION_CONDITIONAL_MARGIN_PCT: float = Field(
        default=1.5, ge=0, le=100,
        description="Minimum margin for conditional status when market confirms",
    )
    DECISION_CONDITIONAL_MIN_DATA_QUALITY: float = Field(
        default=50.0, ge=0, le=100,
        description="Minimum data quality score for contextual threshold relaxation",
    )
    DECISION_MIN_MARKET_EDGE_PCT: float = Field(
        default=3.0, ge=0, le=100,
        description="Minimum edge before a decision is market-grade",
    )
    DECISION_EDGE_MARGIN_PCT: float = Field(
        default=0.0, ge=-10, le=10,
        description="Extra model probability required over implied price",
    )
    MIN_VALUE_EV_POINTS: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    ENSEMBLE_STATS_WEIGHT: float = Field(default=0.4, gt=0, le=1)
    ENSEMBLE_ML_WEIGHT: float = Field(default=0.2, ge=0, le=1)
    ENSEMBLE_MARKET_WEIGHT: float = Field(default=0.4, ge=0, le=1)
    MIN_ENSEMBLE_CALIBRATION_SAMPLES: int = Field(default=100, ge=30)
    ENSEMBLE_HOLDOUT_FRACTION: float = Field(default=0.2, ge=0.1, le=0.4)
    ENSEMBLE_MIN_SOURCE_WEIGHT: float = Field(default=0.05, ge=0, le=0.3)
    ENSEMBLE_MIN_LOG_LOSS_IMPROVEMENT: float = Field(default=0.001, ge=0)
    ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES: int = Field(default=30, ge=6)
    ENSEMBLE_BMA_PRIOR_STRENGTH: float = Field(default=50.0, gt=0, allow_inf_nan=False)
    ENSEMBLE_BMA_HALF_LIFE_DAYS: float = Field(default=180.0, gt=0, allow_inf_nan=False)
    ENSEMBLE_BMA_MIN_DATA_QUALITY_SCORE: float = Field(
        default=0.0, ge=0, le=100, allow_inf_nan=False
    )
    ENSEMBLE_BMA_MAX_BRIER_REGRESSION: float = Field(
        default=0.005, ge=0, le=1, allow_inf_nan=False
    )
    ENSEMBLE_BMA_STATS_LOW_DATA_BOOST: float = Field(
        default=1.5, ge=1, le=5, allow_inf_nan=False
    )
    ENSEMBLE_BMA_ML_HIGH_QUALITY_BOOST: float = Field(
        default=1.5, ge=1, le=5, allow_inf_nan=False
    )
