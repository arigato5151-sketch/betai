import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import OriginValidationMiddleware


def production_settings(**overrides) -> Settings:
    values = {
        "ENVIRONMENT": "production",
        "DEBUG": False,
        "API_FOOTBALL_KEY": "live-api-key",
        "DATABASE_URL": "postgresql://user:password@db:5432/bet_ai",
        "ALLOW_DATABASE_FALLBACK": False,
        "JWT_SECRET_KEY": "a" * 32,
        "JWT_REFRESH_SECRET_KEY": "b" * 32,
        "MODEL_SIGNING_KEY": "c" * 32,
        "COOKIE_SECURE": True,
        "REQUIRE_ORIGIN_HEADER": True,
        "ADMIN_PASSWORD": "strong-admin-password",
        "FRONTEND_URL": "https://bets.example.com",
        "BACKEND_CORS_ORIGINS": "https://bets.example.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_valid_production_security_configuration() -> None:
    settings = production_settings()

    assert settings.BACKEND_CORS_ORIGINS == ["https://bets.example.com"]
    assert settings.REQUIRE_ORIGIN_HEADER is True


@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"DEBUG": True}, "DEBUG must be false"),
        ({"COOKIE_SECURE": False}, "COOKIE_SECURE must be true"),
        ({"JWT_SECRET_KEY": "weak"}, "JWT secrets must be distinct"),
        ({"MODEL_SIGNING_KEY": "weak"}, "MODEL_SIGNING_KEY must be non-default"),
        (
            {"BACKEND_CORS_ORIGINS": "http://bets.example.com"},
            "production CORS origins must use HTTPS",
        ),
        ({"DATABASE_URL": "sqlite:///local.db"}, "DATABASE_URL must use PostgreSQL"),
        ({"ALLOW_DATABASE_FALLBACK": True}, "ALLOW_DATABASE_FALLBACK must be false"),
    ],
)
def test_insecure_production_configuration_fails_fast(
    override: dict[str, object], expected_message: str
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        production_settings(**override)


def test_samesite_none_requires_secure_cookie_in_every_environment() -> None:
    with pytest.raises(ValidationError, match="SameSite=None cookies require"):
        Settings(_env_file=None, COOKIE_SAMESITE="none", COOKIE_SECURE=False)


def test_cors_origins_accept_comma_separated_and_json_values() -> None:
    comma = Settings(
        _env_file=None,
        BACKEND_CORS_ORIGINS="http://localhost:3000,http://localhost:5173",
    )
    json_value = Settings(
        _env_file=None,
        BACKEND_CORS_ORIGINS='["http://localhost:3000", "http://localhost:5173"]',
    )

    assert comma.BACKEND_CORS_ORIGINS == json_value.BACKEND_CORS_ORIGINS


def test_goal_time_decay_factor_is_configurable() -> None:
    settings = Settings(_env_file=None, GOAL_TIME_DECAY_FACTOR=0.025)

    assert settings.GOAL_TIME_DECAY_FACTOR == 0.025


@pytest.mark.parametrize(
    "value",
    [-0.001, 1.001, float("nan"), float("inf")],
)
def test_goal_time_decay_factor_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, GOAL_TIME_DECAY_FACTOR=value)


def test_booster_and_bma_settings_are_configurable() -> None:
    settings = Settings(
        _env_file=None,
        ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES=24,
        ENSEMBLE_BMA_HALF_LIFE_DAYS=90,
        ENSEMBLE_BMA_PRIOR_STRENGTH=25,
        ML_BOOSTER_TREES=300,
        ML_BOOSTER_THREADS=4,
    )

    assert settings.ENSEMBLE_BMA_MIN_LEAGUE_SAMPLES == 24
    assert settings.ENSEMBLE_BMA_HALF_LIFE_DAYS == 90
    assert settings.ENSEMBLE_BMA_PRIOR_STRENGTH == 25
    assert settings.ML_BOOSTER_TREES == 300
    assert settings.ML_BOOSTER_THREADS == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ENSEMBLE_BMA_HALF_LIFE_DAYS", 0),
        ("ENSEMBLE_BMA_PRIOR_STRENGTH", float("nan")),
        ("ENSEMBLE_BMA_MIN_DATA_QUALITY_SCORE", 101),
        ("ENSEMBLE_BMA_STATS_LOW_DATA_BOOST", 0.9),
        ("ML_BOOSTER_TREES", 49),
        ("ML_BOOSTER_LEARNING_RATE", float("inf")),
        ("ML_BOOSTER_THREADS", 0),
    ],
)
def test_booster_and_bma_settings_reject_invalid_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_player_impact_and_fatigue_settings_are_configurable() -> None:
    settings = Settings(
        _env_file=None,
        PLAYER_IMPACT_MIN_RATED_STARTERS=8,
        PLAYER_IMPACT_LOOKBACK_MATCHES=12,
        PLAYER_IMPACT_RATING_DECAY=0.9,
        PLAYER_IMPACT_REPLACEMENT_FACTOR=0.8,
        PLAYER_IMPACT_MIN_STRENGTH_RATIO=0.75,
        PLAYER_IMPACT_MAX_STRENGTH_RATIO=1.1,
        PLAYER_IMPACT_XG_ELASTICITY=1.2,
        PLAYER_IMPACT_MIN_XG_MULTIPLIER=0.8,
        PLAYER_CRITICAL_ABSENCE_WEIGHT=0.3,
        PLAYER_QUESTIONABLE_ABSENCE_WEIGHT=0.2,
        PLAYER_CONTEXT_SYNC_MAX_FIXTURES=25,
        PLAYER_CONTEXT_SYNC_CONCURRENCY=4,
        FATIGUE_LOOKBACK_DAYS=21,
        FATIGUE_MATCH_REFERENCE_COUNT=5,
        FATIGUE_IDEAL_REST_DAYS=6.0,
        FATIGUE_TRAVEL_REFERENCE_KM=2500.0,
        FATIGUE_MATCH_WEIGHT=0.5,
        FATIGUE_REST_WEIGHT=0.3,
        FATIGUE_TRAVEL_WEIGHT=0.2,
    )

    assert settings.PLAYER_IMPACT_MIN_RATED_STARTERS == 8
    assert settings.PLAYER_IMPACT_LOOKBACK_MATCHES == 12
    assert settings.PLAYER_IMPACT_RATING_DECAY == 0.9
    assert settings.PLAYER_IMPACT_REPLACEMENT_FACTOR == 0.8
    assert settings.PLAYER_IMPACT_MIN_STRENGTH_RATIO == 0.75
    assert settings.PLAYER_IMPACT_MAX_STRENGTH_RATIO == 1.1
    assert settings.PLAYER_IMPACT_XG_ELASTICITY == 1.2
    assert settings.PLAYER_IMPACT_MIN_XG_MULTIPLIER == 0.8
    assert settings.PLAYER_CRITICAL_ABSENCE_WEIGHT == 0.3
    assert settings.PLAYER_QUESTIONABLE_ABSENCE_WEIGHT == 0.2
    assert settings.PLAYER_CONTEXT_SYNC_MAX_FIXTURES == 25
    assert settings.PLAYER_CONTEXT_SYNC_CONCURRENCY == 4
    assert settings.FATIGUE_LOOKBACK_DAYS == 21
    assert settings.FATIGUE_MATCH_REFERENCE_COUNT == 5
    assert settings.FATIGUE_IDEAL_REST_DAYS == 6.0
    assert settings.FATIGUE_TRAVEL_REFERENCE_KM == 2500.0
    assert settings.FATIGUE_MATCH_WEIGHT == 0.5
    assert settings.FATIGUE_REST_WEIGHT == 0.3
    assert settings.FATIGUE_TRAVEL_WEIGHT == 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PLAYER_IMPACT_MIN_RATED_STARTERS", 0),
        ("PLAYER_IMPACT_MIN_RATED_STARTERS", 12),
        ("PLAYER_IMPACT_LOOKBACK_MATCHES", 0),
        ("PLAYER_IMPACT_LOOKBACK_MATCHES", 51),
        ("PLAYER_IMPACT_RATING_DECAY", 0),
        ("PLAYER_IMPACT_RATING_DECAY", float("nan")),
        ("PLAYER_IMPACT_REPLACEMENT_FACTOR", -0.01),
        ("PLAYER_IMPACT_REPLACEMENT_FACTOR", 1.01),
        ("PLAYER_IMPACT_MIN_STRENGTH_RATIO", 0),
        ("PLAYER_IMPACT_MIN_STRENGTH_RATIO", 1.01),
        ("PLAYER_IMPACT_MAX_STRENGTH_RATIO", 0.99),
        ("PLAYER_IMPACT_MAX_STRENGTH_RATIO", 1.26),
        ("PLAYER_IMPACT_XG_ELASTICITY", 0),
        ("PLAYER_IMPACT_XG_ELASTICITY", float("inf")),
        ("PLAYER_IMPACT_MIN_XG_MULTIPLIER", 0),
        ("PLAYER_IMPACT_MIN_XG_MULTIPLIER", 1.01),
        ("PLAYER_CRITICAL_ABSENCE_WEIGHT", -0.01),
        ("PLAYER_CRITICAL_ABSENCE_WEIGHT", 1.01),
        ("PLAYER_QUESTIONABLE_ABSENCE_WEIGHT", -0.01),
        ("PLAYER_QUESTIONABLE_ABSENCE_WEIGHT", float("nan")),
        ("PLAYER_CONTEXT_SYNC_MAX_FIXTURES", -1),
        ("PLAYER_CONTEXT_SYNC_MAX_FIXTURES", 501),
        ("PLAYER_CONTEXT_SYNC_CONCURRENCY", 0),
        ("PLAYER_CONTEXT_SYNC_CONCURRENCY", 21),
        ("FATIGUE_LOOKBACK_DAYS", 0),
        ("FATIGUE_LOOKBACK_DAYS", 61),
        ("FATIGUE_MATCH_REFERENCE_COUNT", 0),
        ("FATIGUE_MATCH_REFERENCE_COUNT", 21),
        ("FATIGUE_IDEAL_REST_DAYS", 0),
        ("FATIGUE_IDEAL_REST_DAYS", 31),
        ("FATIGUE_TRAVEL_REFERENCE_KM", 0),
        ("FATIGUE_TRAVEL_REFERENCE_KM", 20001),
        ("FATIGUE_MATCH_WEIGHT", -0.01),
        ("FATIGUE_MATCH_WEIGHT", 1.01),
        ("FATIGUE_REST_WEIGHT", float("inf")),
        ("FATIGUE_TRAVEL_WEIGHT", float("nan")),
    ],
)
def test_player_impact_and_fatigue_settings_reject_invalid_bounds(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_fatigue_weights_must_sum_to_one() -> None:
    with pytest.raises(
        ValidationError,
        match="fatigue feature weights must sum to 1.0",
    ):
        Settings(
            _env_file=None,
            FATIGUE_MATCH_WEIGHT=0.5,
            FATIGUE_REST_WEIGHT=0.4,
            FATIGUE_TRAVEL_WEIGHT=0.2,
        )


def make_origin_client(require_origin_header: bool = True) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        OriginValidationMiddleware,
        allowed_origins=["https://bets.example.com"],
        require_origin_header=require_origin_header,
        access_cookie_name="bet_ai_access",
        refresh_cookie_name="bet_ai_refresh",
        csrf_cookie_name="bet_ai_csrf",
        csrf_header_name="X-CSRF-Token",
    )

    @app.post("/change")
    def change_state() -> dict[str, bool]:
        return {"changed": True}

    return TestClient(app)


def test_origin_middleware_allows_trusted_origin() -> None:
    response = make_origin_client().post(
        "/change", headers={"Origin": "https://bets.example.com"}
    )

    assert response.status_code == 200


def test_origin_middleware_rejects_untrusted_and_missing_origin() -> None:
    client = make_origin_client()

    assert (
        client.post("/change", headers={"Origin": "https://evil.example"}).status_code
        == 403
    )
    assert client.post("/change").status_code == 403


def test_origin_header_is_optional_outside_production() -> None:
    response = make_origin_client(require_origin_header=False).post("/change")

    assert response.status_code == 200


def test_csrf_double_submit_is_required_for_authenticated_mutations() -> None:
    client = make_origin_client(require_origin_header=False)
    client.cookies.set("bet_ai_access", "access")
    client.cookies.set("bet_ai_csrf", "csrf-token")

    missing = client.post("/change")
    valid = client.post("/change", headers={"X-CSRF-Token": "csrf-token"})

    assert missing.status_code == 403
    assert valid.status_code == 200
