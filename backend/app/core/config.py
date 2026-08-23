import json
import os
from pathlib import Path
from typing import Annotated, List, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.core.config_ml import MLFields
from app.core.config_prediction import PredictionFields
from app.core.config_providers import ProviderFields
from app.core.secrets import load_external_secrets

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
SECRET_SOURCE_STATUS = load_external_secrets(ENV_FILE)


class Settings(ProviderFields, MLFields, PredictionFields, BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "test", "production"] = "development"
    SECRET_PROVIDER: Literal["env", "vault", "azure_key_vault"] = "env"
    DEBUG: bool = False
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/bet_ai_pro"
    ALLOW_DATABASE_FALLBACK: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    MEMCACHED_HOST: str | None = None
    MEMCACHED_PORT: int = Field(default=11211, ge=1, le=65535)
    MEMCACHED_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0, le=30)
    JWT_SECRET_KEY: str = "development-access-secret-change-me"
    JWT_REFRESH_SECRET_KEY: str = "development-refresh-secret-change-me"
    MODEL_SIGNING_KEY: str = "development-model-signing-key-change-me"
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
    BATCH_PREDICTION_MAX_REQUESTS: int = Field(default=10, ge=1, le=100)
    BATCH_PREDICTION_WINDOW_SECONDS: int = Field(default=60, ge=1, le=3600)
    REGISTRATION_MAX_REQUESTS: int = Field(default=5, ge=1, le=100)
    REGISTRATION_WINDOW_SECONDS: int = Field(default=3600, ge=60, le=86400)
    CSRF_COOKIE_NAME: str = "bet_ai_csrf"
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    ADMIN_USERNAME: str = Field(default="admin", min_length=1)
    ADMIN_PASSWORD: str | None = None
    BOOTSTRAP_ADMIN_SECRET: str | None = None
    ALLOW_SELF_REGISTRATION: bool = False
    SELF_REGISTRATION_ROLE: Literal["viewer", "analyst"] = "viewer"
    FRONTEND_URL: str = "http://localhost:3000"
    FRONTEND_DIST_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "frontend",
        "dist",
    )
    BACKEND_CORS_ORIGINS: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"
    METRICS_TOKEN: str | None = None

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

    @field_validator("THE_ODDS_API_BASE_URL")
    @classmethod
    def validate_the_odds_api_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.the-odds-api.com"
            or parsed.path
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "THE_ODDS_API_BASE_URL must point to https://api.the-odds-api.com"
            )
        return normalized

    @field_validator("ODDS_API_IO_BASE_URL")
    @classmethod
    def validate_odds_api_io_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.odds-api.io"
            or parsed.path != "/v3"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "ODDS_API_IO_BASE_URL must point to https://api.odds-api.io/v3"
            )
        return normalized

    @field_validator("FOOTBALL_DATA_ORG_BASE_URL")
    @classmethod
    def validate_football_data_org_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.football-data.org"
            or parsed.path != "/v4"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "FOOTBALL_DATA_ORG_BASE_URL must point to "
                "https://api.football-data.org/v4"
            )
        return normalized

    @field_validator("THESPORTSDB_BASE_URL")
    @classmethod
    def validate_thesportsdb_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "www.thesportsdb.com"
            or not parsed.path.startswith("/api/v1/json/")
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("THESPORTSDB_BASE_URL must point to TheSportsDB v1 API")
        return normalized

    @field_validator("OPENLIGADB_BASE_URL")
    @classmethod
    def validate_openligadb_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "api.openligadb.de"
            or parsed.path
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "OPENLIGADB_BASE_URL must point to https://api.openligadb.de"
            )
        return normalized

    @field_validator("STATSBOMB_OPEN_DATA_BASE_URL")
    @classmethod
    def validate_statsbomb_open_data_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "raw.githubusercontent.com"
            or parsed.path != "/statsbomb/open-data/master/data"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "STATSBOMB_OPEN_DATA_BASE_URL must point to the official StatsBomb open-data repository"
            )
        return normalized

    @field_validator("OPENFOOTBALL_BASE_URL")
    @classmethod
    def validate_openfootball_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "raw.githubusercontent.com"
            or parsed.path != "/openfootball/football.json/master"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "OPENFOOTBALL_BASE_URL must point to the official OpenFootball repository"
            )
        return normalized

    @field_validator(
        "OPEN_METEO_FORECAST_URL",
        "OPEN_METEO_HISTORICAL_FORECAST_URL",
        "OPEN_METEO_ARCHIVE_URL",
    )
    @classmethod
    def validate_open_meteo_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        allowed = {
            ("api.open-meteo.com", "/v1/forecast"),
            ("historical-forecast-api.open-meteo.com", "/v1/forecast"),
            ("archive-api.open-meteo.com", "/v1/archive"),
        }
        if (
            parsed.scheme != "https"
            or (parsed.hostname, parsed.path) not in allowed
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Open-Meteo URLs must point to official HTTPS API hosts")
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

        if self.BOOTSTRAP_ADMIN_SECRET is None or len(self.BOOTSTRAP_ADMIN_SECRET) < 16:
            errors.append(
                "BOOTSTRAP_ADMIN_SECRET must be set and at least 16 characters"
            )
        if self.METRICS_TOKEN is None or len(self.METRICS_TOKEN) < 32:
            errors.append("METRICS_TOKEN must be set and at least 32 characters")
        if self.ADMIN_PASSWORD is not None and (
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
        if self.THE_ODDS_API_ENABLED and len(self.THE_ODDS_API_KEY) < 16:
            errors.append(
                "THE_ODDS_API_KEY must be configured when The Odds API is enabled"
            )
        if self.ODDS_API_IO_ENABLED and len(self.ODDS_API_IO_KEY) < 32:
            errors.append(
                "ODDS_API_IO_KEY must be configured when Odds-API.io is enabled"
            )
        if self.FOOTBALL_DATA_ORG_ENABLED and (
            len(self.FOOTBALL_DATA_ORG_API_KEY) < 16
            or self.FOOTBALL_DATA_ORG_API_KEY
            in {"your_football_data_org_key", "replace_with_football_data_org_key"}
        ):
            errors.append(
                "FOOTBALL_DATA_ORG_API_KEY must be configured when football-data.org is enabled"
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
