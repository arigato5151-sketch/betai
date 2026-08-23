from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.endpoints import router
from app.core.passwords import hash_password
from app.db.models import Base, Permission, Role
from app.db.session import get_db
from app.db.user_repository import UserRepository

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()
app.include_router(router, prefix="/api")
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app, base_url="https://testserver")


def _seed() -> None:
    with TestingSessionLocal() as db:
        db.query(Role).delete()
        db.query(Permission).delete()
        db.flush()
        perms = [
            Permission(id=1, code="analysis:create"),
            Permission(id=2, code="history:read"),
            Permission(id=3, code="backtest:run"),
            Permission(id=4, code="audit:read"),
            Permission(id=5, code="predictions:create"),
        ]
        db.add_all(perms)
        db.flush()
        db.add(Role(id=1, name="admin", permissions=list(perms)))
        db.commit()
        UserRepository(db).create_user(
            username="admin",
            email="admin@test.local",
            password_hash=hash_password("admin-password-123"),
            role_names=["admin"],
        )


_seed()


def _login() -> TestClient:
    resp = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin-password-123"},
    )
    assert resp.status_code == 200
    return client


def test_unauthenticated_leagues_returns_401() -> None:
    response = client.get("/api/leagues")
    assert response.status_code == 401


def test_unauthenticated_upcoming_returns_401() -> None:
    response = client.get("/api/fixtures/upcoming")
    assert response.status_code == 401


def test_unauthenticated_prefill_returns_401() -> None:
    response = client.get("/api/fixtures/999/prefill")
    assert response.status_code == 401


def test_authenticated_leagues_passes() -> None:
    _login()
    response = client.get("/api/leagues")
    assert response.status_code == 200


def test_predict_tiered_error_leaks_no_internals() -> None:
    _login()
    tier1_features = {
        "home_team": "Team A",
        "away_team": "Team B",
        "home_form_last5": 7.0,
        "away_form_last5": 6.0,
        "home_avg_goals": 1.8,
        "away_avg_goals": 1.2,
        "home_elo": 1500,
        "away_elo": 1450,
        "opening_home_odd": 1.85,
        "opening_draw_odd": 3.50,
        "opening_away_odd": 4.20,
        "home_clean_sheet_streak": 1.0,
        "away_clean_sheet_streak": 0.0,
        "home_scoring_streak": 2.0,
        "away_scoring_streak": 1.0,
        "rest_days_diff": 0.0,
        "fatigue_index": 0.5,
        "home_advantage_coeff": 1.05,
        "h2h_home_win_rate": 0.45,
    }
    from app.api.endpoints import get_tiered_predictor

    predictor = Mock()
    predictor.predict.side_effect = ValueError("model shape mismatch (3,80) vs (3,81)")
    app.dependency_overrides[get_tiered_predictor] = lambda: predictor
    try:
        response = client.post(
            "/api/predict/tiered",
            json={"league_id": 88, "features": tier1_features},
        )
    finally:
        app.dependency_overrides.pop(get_tiered_predictor, None)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "İşleme hatası."
    assert "shape" not in str(body)
    assert "80" not in str(body)


def test_preview_analysis_error_leaks_no_internals() -> None:
    _login()
    payload = {
        "home_team": "Team A",
        "away_team": "Team B",
        "league_id": 88,
        "odd": 2.10,
        "home_stats": {"form": 70.0, "attack": 65.0, "defense": 55.0, "xg": 1.5},
        "away_stats": {"form": 60.0, "attack": 50.0, "defense": 60.0, "xg": 1.2},
    }
    with patch("app.api.endpoints._compute_analysis") as mock_compute:
        mock_compute.side_effect = TypeError(
            "unsupported operand type(s) for +: 'NoneType' and 'float'"
        )
        response = client.post("/api/analyze/preview", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "İşleme hatası."
    assert "NoneType" not in str(body)


def test_batch_predict_rate_limit_returns_429(monkeypatch) -> None:
    _login()
    from app.api.endpoints import batch_prediction_rate_limiter

    monkeypatch.setattr(
        batch_prediction_rate_limiter,
        "consume",
        Mock(side_effect=[(True, 0)] * 10 + [(False, 60)]),
    )
    for _ in range(10):
        response = client.post(
            "/api/predictions/batch",
            json=[999999],
        )
    response = client.post(
        "/api/predictions/batch",
        json=[999999],
    )
    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_batch_predict_rejects_empty_and_duplicate_fixture_ids() -> None:
    _login()

    empty = client.post("/api/predictions/batch", json=[])
    duplicate = client.post("/api/predictions/batch", json=[42, 42])

    assert empty.status_code == 422
    assert duplicate.status_code == 422
    assert "birden fazla" in duplicate.json()["detail"]
