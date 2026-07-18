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
    JWT_SECRET_KEY: str = "change-this-secret"
    JWT_REFRESH_SECRET_KEY: str = "change-this-refresh-secret"
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

    MIN_TRAINING_SAMPLES: int = 200
    RETRAIN_EVERY_N_NEW: int = 25
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

        placeholder_values = {"change-this-secret", "change-this-refresh-secret"}
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
