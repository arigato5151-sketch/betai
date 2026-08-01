import json
import os
from pathlib import Path
from typing import Annotated, List, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.secrets import load_external_secrets

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
SECRET_SOURCE_STATUS = load_external_secrets(ENV_FILE)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "test", "production"] = "development"
    SECRET_PROVIDER: Literal["env", "vault", "azure_key_vault"] = "env"
    DEBUG: bool = True
    API_FOOTBALL_KEY: str = "DEMO_KEY"
    API_FOOTBALL_CIRCUIT_FAILURE_THRESHOLD: int = Field(default=4, ge=2, le=20)
    API_FOOTBALL_CIRCUIT_OPEN_SECONDS: int = Field(default=60, ge=10, le=900)
    API_FOOTBALL_MAX_RETRY_AFTER_SECONDS: int = Field(default=120, ge=10, le=900)
    API_FOOTBALL_BACKOFF_JITTER_RATIO: float = Field(default=0.2, ge=0, le=1)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/bet_ai_pro"
    ALLOW_DATABASE_FALLBACK: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    MEMCACHED_HOST: str | None = None
    MEMCACHED_PORT: int = Field(default=11211, ge=1, le=65535)
    MEMCACHED_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0, le=30)
    JWT_SECRET_KEY: str = "development-access-secret-change-me"
    JWT_REFRESH_SECRET_KEY: str = "development-refresh-secret-change-me"
    MODEL_SIGNING_KEY: str = "development-model-signing-key-change-me"
    MODEL_DRIFT_WINDOW_SIZE: int = Field(default=100, ge=20, le=2000)
    MODEL_DRIFT_MIN_SAMPLES: int = Field(default=30, ge=10, le=1000)
    MODEL_DRIFT_BRIER_THRESHOLD: float = Field(default=0.04, gt=0, le=1)
    JWT_ALGORITHM: Literal["HS256", "HS384", "HS512"] = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1, le=1440)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=90)
    ACCESS_TOKEN_COOKIE_NAME: str = "bet_ai_access"
    REFRESH_TOKEN_COOKIE_NAME: str = "bet_ai_refresh"
    COOKIE_SECURE: bool = True
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
    COOKIE_DOMAIN: str | None = None
    REQUIRE_ORIGIN_HEADER: bool = False
    LOGIN_MAX_ATTEMPTS: int = Field(default=5, ge=2, le=100)
    LOGIN_WINDOW_SECONDS: int = Field(default=300, ge=10, le=86400)
    LOGIN_LOCKOUT_SECONDS: int = Field(default=900, ge=10, le=86400)
    LOGIN_REDIS_RECOVERY_SECONDS: float = Field(default=30.0, ge=1.0, le=300.0)
    CSRF_COOKIE_NAME: str = "bet_ai_csrf"
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    ADMIN_USERNAME: str = Field(default="admin", min_length=1)
    ADMIN_PASSWORD: str = Field(default="change-this-password", min_length=8)
    ALLOW_SELF_REGISTRATION: bool = False
    SELF_REGISTRATION_ROLE: Literal["viewer", "analyst"] = "viewer"
    FRONTEND_URL: str = "http://localhost:3000"
    FRONTEND_DIST_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "frontend",
        "dist",
    )
    BACKEND_CORS_ORIGINS: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"
    FOOTBALL_DATA_BASE_URL: str = "https://www.football-data.co.uk"
    FOOTBALL_DATA_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    FIXTURE_DOWNLOAD_BASE_URL: str = "https://fixturedownload.com/feed/json"
    FIXTURE_DOWNLOAD_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    UNDERSTAT_ENABLED: bool = False
    UNDERSTAT_BASE_URL: str = "https://understat.com"
    UNDERSTAT_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    UNDERSTAT_MATCH_TOLERANCE_HOURS: int = Field(default=48, ge=1, le=48)
    UNDERSTAT_REQUEST_INTERVAL_SECONDS: float = Field(default=1.5, ge=0.5, le=10)
    DERIVED_XG_ENABLED: bool = True
    DERIVED_XG_MIN_TRAINING_MATCHES: int = Field(default=500, ge=100, le=10000)
    DERIVED_XG_MAX_HOLDOUT_MAE: float = Field(default=0.60, gt=0, le=2)
    DERIVED_XG_MIN_BASELINE_IMPROVEMENT: float = Field(default=0.10, ge=0, le=1)
    DERIVED_XG_CONFIDENCE: float = Field(default=0.65, gt=0, lt=0.95)
    CLUBELO_ENABLED: bool = False
    CLUBELO_BASE_URL: str = "http://api.clubelo.com"
    CLUBELO_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=120)
    CLUBELO_CACHE_HOURS: int = Field(default=24, ge=1, le=168)
    CLUBELO_CONFIDENCE: float = Field(default=0.80, gt=0, le=1, allow_inf_nan=False)
    SPORTMONKS_ENABLED: bool = False
    SPORTMONKS_API_TOKEN: str = ""
    SPORTMONKS_BASE_URL: str = "https://api.sportmonks.com/v3/football"
    SPORTMONKS_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=120)
    SPORTMONKS_PLAYER_LOOKBACK_DAYS: int = Field(default=120, ge=30, le=365)
    SPORTMONKS_PLAYER_LOOKBACK_MATCHES: int = Field(default=10, ge=3, le=30)
    ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS: int = Field(default=300, ge=60, le=86400)
    ODDS_SNAPSHOT_CONFIDENCE: float = Field(
        default=0.90, gt=0, le=1, allow_inf_nan=False
    )
    ODDS_COLLECTOR_ENABLED: bool = True
    ODDS_COLLECTOR_RUN_INTERVAL_SECONDS: int = Field(
        default=10800,
        ge=900,
        le=86400,
    )
    ODDS_COLLECTOR_HORIZON_DAYS: int = Field(default=7, ge=1, le=14)
    ODDS_COLLECTOR_MAX_FIXTURES: int = Field(default=20, ge=1, le=200)
    ODDS_COLLECTOR_CLOSING_WINDOW_HOURS: int = Field(
        default=24,
        ge=1,
        le=168,
    )
    ODDS_COLLECTOR_CONCURRENCY: int = Field(default=2, ge=1, le=10)
    LINEUP_COLLECTOR_ENABLED: bool = True
    LINEUP_COLLECTOR_RUN_INTERVAL_SECONDS: int = Field(
        default=3600,
        ge=900,
        le=21600,
    )
    LINEUP_COLLECTOR_HORIZON_DAYS: int = Field(default=2, ge=1, le=3)
    LINEUP_COLLECTOR_MAX_FIXTURES: int = Field(default=30, ge=1, le=100)
    LINEUP_COLLECTOR_WINDOW_MINUTES: int = Field(default=120, ge=30, le=360)
    LINEUP_COLLECTOR_CONCURRENCY: int = Field(default=2, ge=1, le=10)
    AUTO_TEAM_LOCATION_ENABLED: bool = True
    WIKIDATA_LOCATION_ENABLED: bool = True
    WIKIDATA_API_URL: str = "https://www.wikidata.org/w/api.php"
    WIKIDATA_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    WIKIDATA_LOCATION_MAX_TEAMS: int = Field(default=500, ge=1, le=2000)
    WIKIDATA_LOCATION_CONCURRENCY: int = Field(default=1, ge=1, le=4)
    WIKIDATA_REQUEST_INTERVAL_SECONDS: float = Field(default=0.25, ge=0.1, le=5)
    FREE_TEAM_LOCATION_MAX_TEAMS: int = Field(default=70, ge=1, le=90)

    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEAGUE_BASELINE_GOALS: float = Field(default=1.32, gt=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md ve calibration_time_decay.json
    GOAL_TIME_DECAY_FACTOR: float = Field(
        default=0.008,
        ge=0,
        le=1,
        allow_inf_nan=False,
    )
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    FORM_DECAY_WEIGHTS: tuple[float, ...] = Field(default=(1.0, 0.88, 0.76, 0.64, 0.52))
    # TODO: kalibrasyon — mevcut form kodu beş maçta kestiği için dal erişilemiyor.
    FORM_DECAY_FALLBACK_WEIGHT: float = Field(default=0.4, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    HOME_ATTACK_BOOST: float = Field(default=1.11, gt=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    AWAY_ATTACK_PENALTY: float = Field(default=0.93, gt=0)
    # TODO: kalibrasyon — strength_rating tahmin yollarında tüketilmiyor.
    STRENGTH_ATTACK_WEIGHT: float = Field(default=0.4, ge=0)
    # TODO: kalibrasyon — strength_rating tahmin yollarında tüketilmiyor.
    STRENGTH_DEFENSE_WEIGHT: float = Field(default=0.35, ge=0)
    # TODO: kalibrasyon — strength_rating tahmin yollarında tüketilmiyor.
    STRENGTH_FORM_WEIGHT: float = Field(default=0.25, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    XG_OBSERVED_GOALS_WEIGHT: float = Field(default=0.55, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    XG_ATTACK_BASELINE_WEIGHT: float = Field(default=0.45, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    XG_CONSISTENCY_MAX_PENALTY: float = Field(default=0.12, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    XG_CONSISTENCY_PENALTY_WEIGHT: float = Field(default=0.04, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    PROFILE_FORM_FACTOR_BASE: float = Field(default=0.88, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    PROFILE_FORM_FACTOR_WEIGHT: float = Field(default=0.24, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_ATTACK_FACTOR_BASE: float = Field(default=0.62, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_ATTACK_FACTOR_WEIGHT: float = Field(default=0.78, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_DEFENSE_FACTOR_BASE: float = Field(default=0.72, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_DEFENSE_FACTOR_WEIGHT: float = Field(default=0.55, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_FORM_FACTOR_BASE: float = Field(default=0.82, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_FORM_FACTOR_WEIGHT: float = Field(default=0.36, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_XG_OBSERVED_WEIGHT: float = Field(default=0.58, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEGACY_XG_BASELINE_WEIGHT: float = Field(default=0.42, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    HOME_ADVANTAGE_MIN_MULTIPLIER: float = Field(default=0.88, gt=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    HOME_ADVANTAGE_MAX_MULTIPLIER: float = Field(default=1.22, gt=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR: float = Field(default=0.55, gt=0)
    # TODO: kalibrasyon — tarihsel profillerde eksik gol ortalaması örneği yok.
    HOME_FORM_BASE_MULTIPLIER: float = Field(default=1.08, gt=0)
    # TODO: kalibrasyon — tarihsel profillerde eksik gol ortalaması örneği yok.
    HOME_FORM_BOOST_DIVISOR: float = Field(default=450.0, gt=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT: float = Field(default=12.0, ge=0)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    DOUBLE_CHANCE_AWAY_DIFFERENCE_WEIGHT: float = Field(default=14.0, ge=0)
    # Kalibre edildi; verisiz UEFA kupaları global fallback kullanır: bkz. docs/CALIBRATION.md
    DEFAULT_DIXON_COLES_RHO: float = Field(default=-0.12)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    LEAGUE_DIXON_COLES_RHO: dict[int, float] = Field(
        default={
            39: -0.13,
            140: -0.11,
            135: -0.15,
            78: -0.09,
            61: -0.12,
            40: -0.14,
            94: -0.12,
            203: -0.10,
            88: -0.08,
            144: -0.12,
            235: -0.11,
            79: -0.10,
            136: -0.14,
            62: -0.13,
        }
    )
    # TODO: kalibrasyon — üç bileşenli çözülmüş tahmin örneği henüz yok.
    ENSEMBLE_STATS_WEIGHT: float = Field(default=0.4, gt=0, le=1)
    # TODO: kalibrasyon — üç bileşenli çözülmüş tahmin örneği henüz yok.
    ENSEMBLE_ML_WEIGHT: float = Field(default=0.2, ge=0, le=1)
    # TODO: kalibrasyon — üç bileşenli çözülmüş tahmin örneği henüz yok.
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

    # Son form feature'ları için beklenen tamamlanmış maç sayısı.
    RECENT_FORM_MATCH_COUNT: int = Field(default=5, ge=1, le=20)
    # Eski yerel snapshot yerine canlı API fallback'ine geçiş eşiği.
    HISTORICAL_FORM_MAX_AGE_DAYS: int = Field(default=45, ge=1, le=365)
    # Oyuncu rating kapsamı yetersizse kadro etkisi nötr kalır.
    PLAYER_IMPACT_MIN_RATED_STARTERS: int = Field(default=7, ge=1, le=11)
    PLAYER_IMPACT_LOOKBACK_MATCHES: int = Field(default=10, ge=1, le=50)
    PLAYER_IMPACT_RATING_DECAY: float = Field(
        default=0.85, gt=0, le=1, allow_inf_nan=False
    )
    PLAYER_IMPACT_REPLACEMENT_FACTOR: float = Field(
        default=0.75, ge=0, le=1, allow_inf_nan=False
    )
    PLAYER_IMPACT_MIN_STRENGTH_RATIO: float = Field(
        default=0.70, gt=0, le=1, allow_inf_nan=False
    )
    PLAYER_IMPACT_MAX_STRENGTH_RATIO: float = Field(
        default=1.05, ge=1, le=1.25, allow_inf_nan=False
    )
    PLAYER_IMPACT_XG_ELASTICITY: float = Field(
        default=1.0, gt=0, le=3, allow_inf_nan=False
    )
    PLAYER_IMPACT_MIN_XG_MULTIPLIER: float = Field(
        default=0.75, gt=0, le=1, allow_inf_nan=False
    )
    PLAYER_CRITICAL_ABSENCE_WEIGHT: float = Field(
        default=0.25, ge=0, le=1, allow_inf_nan=False
    )
    PLAYER_QUESTIONABLE_ABSENCE_WEIGHT: float = Field(
        default=0.35, ge=0, le=1, allow_inf_nan=False
    )
    PLAYER_CONTEXT_SYNC_MAX_FIXTURES: int = Field(default=20, ge=0, le=500)
    PLAYER_CONTEXT_SYNC_CONCURRENCY: int = Field(default=3, ge=1, le=20)
    FATIGUE_LOOKBACK_DAYS: int = Field(default=14, ge=1, le=60)
    FATIGUE_MATCH_REFERENCE_COUNT: int = Field(default=4, ge=1, le=20)
    FATIGUE_IDEAL_REST_DAYS: float = Field(
        default=7.0, gt=0, le=30, allow_inf_nan=False
    )
    FATIGUE_TRAVEL_REFERENCE_KM: float = Field(
        default=3000.0, gt=0, le=20000, allow_inf_nan=False
    )
    FATIGUE_MATCH_WEIGHT: float = Field(default=0.45, ge=0, le=1, allow_inf_nan=False)
    FATIGUE_REST_WEIGHT: float = Field(default=0.40, ge=0, le=1, allow_inf_nan=False)
    FATIGUE_TRAVEL_WEIGHT: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    ELO_K_FACTOR: float = Field(default=32.0, gt=0, le=100)
    # Kalibre edildi: bkz. docs/CALIBRATION.md
    ELO_HOME_ADVANTAGE_POINTS: float = Field(default=65.0, ge=0, le=200)
    # TODO: kalibrasyon — sezonlar arasında ortak takım kimliği bulunmuyor.
    ELO_SEASON_REGRESSION: float = Field(default=0.25, ge=0, le=1)

    MIN_TRAINING_SAMPLES: int = 200
    RETRAIN_EVERY_N_NEW: int = 25
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

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        if value.startswith("["):
            return json.loads(value)
        return [origin.strip() for origin in value.split(",") if origin.strip()]

    @field_validator("BACKEND_CORS_ORIGINS")
    @classmethod
    def validate_cors_origins(cls, origins: List[str]) -> List[str]:
        normalized: list[str] = []
        for origin in origins:
            origin = origin.rstrip("/")
            parsed = urlsplit(origin)
            if (
                origin == "*"
                or parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"Invalid CORS origin: {origin!r}")
            normalized.append(origin)
        if len(normalized) != len(set(normalized)):
            raise ValueError("BACKEND_CORS_ORIGINS contains duplicate origins")
        return normalized

    @field_validator("COOKIE_DOMAIN")
    @classmethod
    def validate_cookie_domain(cls, value: str | None) -> str | None:
        if value and any(character in value for character in ":/"):
            raise ValueError("COOKIE_DOMAIN must be a hostname, not a URL")
        return value

    @field_validator("CLUBELO_BASE_URL")
    @classmethod
    def validate_clubelo_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname != "api.clubelo.com"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("CLUBELO_BASE_URL must point to api.clubelo.com")
        return normalized

    @field_validator("SPORTMONKS_BASE_URL")
    @classmethod
    def validate_sportmonks_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.sportmonks.com"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path != "/v3/football"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "SPORTMONKS_BASE_URL must point to "
                "https://api.sportmonks.com/v3/football"
            )
        return normalized

    @field_validator("FIXTURE_DOWNLOAD_BASE_URL")
    @classmethod
    def validate_fixture_download_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "fixturedownload.com"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path != "/feed/json"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "FIXTURE_DOWNLOAD_BASE_URL must point to "
                "https://fixturedownload.com/feed/json"
            )
        return normalized

    @field_validator("UNDERSTAT_BASE_URL")
    @classmethod
    def validate_understat_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "understat.com"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("UNDERSTAT_BASE_URL must point to https://understat.com")
        return normalized

    @field_validator("WIKIDATA_API_URL")
    @classmethod
    def validate_wikidata_api_url(cls, value: str) -> str:
        normalized = value.strip()
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "www.wikidata.org"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.path != "/w/api.php"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "WIKIDATA_API_URL must point to https://www.wikidata.org/w/api.php"
            )
        return normalized

    @model_validator(mode="after")
    def validate_ensemble_policy(self) -> "Settings":
        if len(("stats", "ml", "market")) * self.ENSEMBLE_MIN_SOURCE_WEIGHT >= 1:
            raise ValueError(
                "ENSEMBLE_MIN_SOURCE_WEIGHT must leave posterior mass to distribute"
            )
        fatigue_weight = (
            self.FATIGUE_MATCH_WEIGHT
            + self.FATIGUE_REST_WEIGHT
            + self.FATIGUE_TRAVEL_WEIGHT
        )
        if not abs(fatigue_weight - 1.0) <= 1e-9:
            raise ValueError("fatigue feature weights must sum to 1.0")
        if (
            self.PLAYER_IMPACT_MIN_STRENGTH_RATIO
            > self.PLAYER_IMPACT_MAX_STRENGTH_RATIO
        ):
            raise ValueError(
                "PLAYER_IMPACT_MIN_STRENGTH_RATIO cannot exceed the maximum"
            )
        if self.PLAYER_IMPACT_MIN_XG_MULTIPLIER > self.PLAYER_IMPACT_MAX_STRENGTH_RATIO:
            raise ValueError(
                "PLAYER_IMPACT_MIN_XG_MULTIPLIER cannot exceed the maximum "
                "strength ratio"
            )
        return self

    @model_validator(mode="after")
    def validate_security_policy(self) -> "Settings":
        if self.COOKIE_SAMESITE == "none" and not self.COOKIE_SECURE:
            raise ValueError("SameSite=None cookies require COOKIE_SECURE=true")
        if self.ENVIRONMENT != "production":
            return self

        errors: list[str] = []
        if self.DEBUG:
            errors.append("DEBUG must be false")
        if not self.COOKIE_SECURE:
            errors.append("COOKIE_SECURE must be true")
        if not self.REQUIRE_ORIGIN_HEADER:
            errors.append("REQUIRE_ORIGIN_HEADER must be true")
        if not self.DATABASE_URL.startswith(
            ("postgresql://", "postgresql+psycopg2://")
        ):
            errors.append("DATABASE_URL must use PostgreSQL")
        if self.ALLOW_DATABASE_FALLBACK:
            errors.append("ALLOW_DATABASE_FALLBACK must be false")

        placeholder_values = {
            "change-this-secret",
            "change-this-refresh-secret",
            "development-access-secret-change-me",
            "development-refresh-secret-change-me",
            "development-model-signing-key-change-me",
        }
        secrets = (self.JWT_SECRET_KEY, self.JWT_REFRESH_SECRET_KEY)
        if any(len(secret) < 32 or secret in placeholder_values for secret in secrets):
            errors.append(
                "JWT secrets must be distinct, non-default and at least 32 characters"
            )
        elif self.JWT_SECRET_KEY == self.JWT_REFRESH_SECRET_KEY:
            errors.append("JWT access and refresh secrets must be distinct")
        if (
            len(self.MODEL_SIGNING_KEY) < 32
            or self.MODEL_SIGNING_KEY in placeholder_values
        ):
            errors.append(
                "MODEL_SIGNING_KEY must be non-default and at least 32 characters"
            )
        elif self.MODEL_SIGNING_KEY in secrets:
            errors.append("MODEL_SIGNING_KEY must be distinct from JWT secrets")

        if (
            len(self.ADMIN_PASSWORD) < 12
            or self.ADMIN_PASSWORD == "change-this-password"
        ):
            errors.append(
                "ADMIN_PASSWORD must be non-default and at least 12 characters"
            )
        if not self.API_FOOTBALL_KEY or self.API_FOOTBALL_KEY in {
            "DEMO_KEY",
            "your_api_key_here",
            "your_api_football_key",
        }:
            errors.append("API_FOOTBALL_KEY must be configured")
        if self.SPORTMONKS_ENABLED and (
            len(self.SPORTMONKS_API_TOKEN) < 24
            or self.SPORTMONKS_API_TOKEN
            in {"your_sportmonks_token", "replace_with_sportmonks_token"}
        ):
            errors.append(
                "SPORTMONKS_API_TOKEN must be configured when Sportmonks is enabled"
            )

        frontend_origin = self.FRONTEND_URL.rstrip("/")
        if not frontend_origin.startswith("https://"):
            errors.append("FRONTEND_URL must use HTTPS")
        if not self.BACKEND_CORS_ORIGINS:
            errors.append("BACKEND_CORS_ORIGINS cannot be empty")
        elif any(
            not origin.startswith("https://") for origin in self.BACKEND_CORS_ORIGINS
        ):
            errors.append("all production CORS origins must use HTTPS")
        if frontend_origin not in self.BACKEND_CORS_ORIGINS:
            errors.append("FRONTEND_URL must be included in BACKEND_CORS_ORIGINS")

        if errors:
            raise ValueError("Invalid production configuration: " + "; ".join(errors))
        return self


# Invalid configuration must fail fast; silently weakening security is unsafe.
settings = Settings()
