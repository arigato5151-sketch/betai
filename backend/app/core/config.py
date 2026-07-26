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
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/bet_ai_pro"
    ALLOW_DATABASE_FALLBACK: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    MEMCACHED_HOST: str | None = None
    MEMCACHED_PORT: int = Field(default=11211, ge=1, le=65535)
    MEMCACHED_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0, le=30)
    JWT_SECRET_KEY: str = "development-access-secret-change-me"
    JWT_REFRESH_SECRET_KEY: str = "development-refresh-secret-change-me"
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

    # TODO: kalibrasyon kaynağını doğrula — lig başına ortalama gol tabanı.
    LEAGUE_BASELINE_GOALS: float = Field(default=1.32, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — son beş maçın azalan form ağırlıkları.
    FORM_DECAY_WEIGHTS: tuple[float, ...] = Field(default=(1.0, 0.88, 0.76, 0.64, 0.52))
    # TODO: kalibrasyon kaynağını doğrula — beş maçtan eski form girdisi ağırlığı.
    FORM_DECAY_FALLBACK_WEIGHT: float = Field(default=0.4, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — ev sahibi xG hücum çarpanı.
    HOME_ATTACK_BOOST: float = Field(default=1.11, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — deplasman xG hücum çarpanı.
    AWAY_ATTACK_PENALTY: float = Field(default=0.93, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — güç puanındaki hücum payı.
    STRENGTH_ATTACK_WEIGHT: float = Field(default=0.4, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — güç puanındaki savunma payı.
    STRENGTH_DEFENSE_WEIGHT: float = Field(default=0.35, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — güç puanındaki form payı.
    STRENGTH_FORM_WEIGHT: float = Field(default=0.25, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — gözlenen gol ortalamasının xG payı.
    XG_OBSERVED_GOALS_WEIGHT: float = Field(default=0.55, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — hücum gücü tabanlı xG payı.
    XG_ATTACK_BASELINE_WEIGHT: float = Field(default=0.45, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — maksimum xG tutarlılık cezası.
    XG_CONSISTENCY_MAX_PENALTY: float = Field(default=0.12, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — gol farkı başına tutarlılık cezası.
    XG_CONSISTENCY_PENALTY_WEIGHT: float = Field(default=0.04, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — güçlü profil form çarpanı tabanı.
    PROFILE_FORM_FACTOR_BASE: float = Field(default=0.88, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — güçlü profil form etkisi.
    PROFILE_FORM_FACTOR_WEIGHT: float = Field(default=0.24, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil hücum çarpanı tabanı.
    LEGACY_ATTACK_FACTOR_BASE: float = Field(default=0.62, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil hücum etkisi.
    LEGACY_ATTACK_FACTOR_WEIGHT: float = Field(default=0.78, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil savunma çarpanı tabanı.
    LEGACY_DEFENSE_FACTOR_BASE: float = Field(default=0.72, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil savunma etkisi.
    LEGACY_DEFENSE_FACTOR_WEIGHT: float = Field(default=0.55, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil form çarpanı tabanı.
    LEGACY_FORM_FACTOR_BASE: float = Field(default=0.82, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil form etkisi.
    LEGACY_FORM_FACTOR_WEIGHT: float = Field(default=0.36, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil xG gözlem payı.
    LEGACY_XG_OBSERVED_WEIGHT: float = Field(default=0.58, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — eski profil lig tabanı payı.
    LEGACY_XG_BASELINE_WEIGHT: float = Field(default=0.42, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — ev avantajı alt çarpan sınırı.
    HOME_ADVANTAGE_MIN_MULTIPLIER: float = Field(default=0.88, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — ev avantajı üst çarpan sınırı.
    HOME_ADVANTAGE_MAX_MULTIPLIER: float = Field(default=1.22, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — rakip gol ortalaması için oran tabanı.
    HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR: float = Field(default=0.55, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — veri yokken ev avantajı tabanı.
    HOME_FORM_BASE_MULTIPLIER: float = Field(default=1.08, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — form farkını ev avantajına ölçekler.
    HOME_FORM_BOOST_DIVISOR: float = Field(default=450.0, gt=0)
    # TODO: kalibrasyon kaynağını doğrula — ev çifte şans olasılık katsayısı.
    DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT: float = Field(default=12.0, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — deplasman çifte şans olasılık katsayısı.
    DOUBLE_CHANCE_AWAY_DIFFERENCE_WEIGHT: float = Field(default=14.0, ge=0)
    # TODO: kalibrasyon kaynağını doğrula — varsayılan Dixon-Coles düzeltmesi.
    DEFAULT_DIXON_COLES_RHO: float = Field(default=-0.12)
    # TODO: kalibrasyon kaynağını doğrula — lig bazlı Dixon-Coles düzeltmeleri.
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
    # TODO: kalibrasyon kaynağını doğrula — ensemble içindeki istatistik modeli payı.
    ENSEMBLE_STATS_WEIGHT: float = Field(default=0.4, gt=0, le=1)
    # TODO: kalibrasyon kaynağını doğrula — ensemble içindeki kalibre ML payı.
    ENSEMBLE_ML_WEIGHT: float = Field(default=0.2, ge=0, le=1)
    # TODO: kalibrasyon kaynağını doğrula — ensemble içindeki de-vig market payı.
    ENSEMBLE_MARKET_WEIGHT: float = Field(default=0.4, ge=0, le=1)
    MIN_ENSEMBLE_CALIBRATION_SAMPLES: int = Field(default=100, ge=30)
    ENSEMBLE_HOLDOUT_FRACTION: float = Field(default=0.2, ge=0.1, le=0.4)
    ENSEMBLE_MIN_SOURCE_WEIGHT: float = Field(default=0.05, ge=0, le=0.3)
    ENSEMBLE_MIN_LOG_LOSS_IMPROVEMENT: float = Field(default=0.001, ge=0)

    # Son form feature'ları için beklenen tamamlanmış maç sayısı.
    RECENT_FORM_MATCH_COUNT: int = Field(default=5, ge=1, le=20)
    # Eski yerel snapshot yerine canlı API fallback'ine geçiş eşiği.
    HISTORICAL_FORM_MAX_AGE_DAYS: int = Field(default=45, ge=1, le=365)
    # TODO: kalibrasyon kaynağını doğrula — maç başına Elo güncelleme hızı.
    ELO_K_FACTOR: float = Field(default=32.0, gt=0, le=100)
    # TODO: kalibrasyon kaynağını doğrula — Elo beklenen skorundaki ev avantajı.
    ELO_HOME_ADVANTAGE_POINTS: float = Field(default=65.0, ge=0, le=200)
    # TODO: kalibrasyon kaynağını doğrula — sezon geçişinde ortalamaya dönüş oranı.
    ELO_SEASON_REGRESSION: float = Field(default=0.25, ge=0, le=1)

    MIN_TRAINING_SAMPLES: int = 200
    RETRAIN_EVERY_N_NEW: int = 25
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
        }
        secrets = (self.JWT_SECRET_KEY, self.JWT_REFRESH_SECRET_KEY)
        if any(len(secret) < 32 or secret in placeholder_values for secret in secrets):
            errors.append(
                "JWT secrets must be distinct, non-default and at least 32 characters"
            )
        elif self.JWT_SECRET_KEY == self.JWT_REFRESH_SECRET_KEY:
            errors.append("JWT access and refresh secrets must be distinct")

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
