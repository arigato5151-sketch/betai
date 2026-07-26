from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.endpoints import router
from app.core.config import settings
from app.core.auth import decode_token_payload
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


def _seed_users() -> None:
    permission_codes = [
        "analysis:create",
        "history:read",
        "history:update_result",
        "backtest:run",
        "audit:read",
        "users:manage",
        "roles:manage",
    ]
    with TestingSessionLocal() as db:
        permissions = [
            Permission(id=index, code=code)
            for index, code in enumerate(permission_codes, start=1)
        ]
        admin_role = Role(id=1, name="admin", permissions=permissions)
        viewer_role = Role(
            id=2,
            name="viewer",
            permissions=[
                permission
                for permission in permissions
                if permission.code in {"history:read", "audit:read"}
            ],
        )
        db.add_all([admin_role, viewer_role])
        db.commit()

        repo = UserRepository(db)
        repo.create_user(
            username="admin",
            email="admin@example.test",
            password_hash=hash_password("admin-password-123"),
            role_names=["admin"],
        )
        repo.create_user(
            username="viewer",
            email="viewer@example.test",
            password_hash=hash_password("viewer-password-123"),
            role_names=["viewer"],
        )


_seed_users()


def test_unsigned_jwt_is_rejected() -> None:
    token = jwt.encode(
        {
            "sub": "user-id",
            "type": "access",
            "ver": 0,
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        key="",
        algorithm="none",
    )

    with pytest.raises(HTTPException) as error:
        decode_token_payload(token, "access")

    assert error.value.status_code == 401


def _login(username: str, password: str, test_client: TestClient = client):
    return test_client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )


def test_login_sets_secure_httponly_cookies_without_exposing_tokens() -> None:
    response = _login("admin", "admin-password-123")

    assert response.status_code == 200
    assert "access_token" not in response.json()
    assert "refresh_token" not in response.json()
    assert response.json()["user"]["roles"] == ["admin"]
    assert "users:manage" in response.json()["user"]["permissions"]
    set_cookie = response.headers.get_list("set-cookie")
    assert len(set_cookie) == 3
    auth_cookies = [value for value in set_cookie if "bet_ai_csrf" not in value]
    csrf_cookie = next(value for value in set_cookie if "bet_ai_csrf" in value)
    assert all("HttpOnly" in value for value in auth_cookies)
    assert "HttpOnly" not in csrf_cookie
    assert all("Secure" in value for value in set_cookie)
    assert all("SameSite=lax" in value for value in set_cookie)


def test_self_registration_is_gated_and_creates_viewer_session(monkeypatch) -> None:
    disabled = client.post(
        "/api/auth/register",
        json={
            "username": "self-user",
            "email": "self-user@example.test",
            "password": "self-password-123",
        },
    )
    assert disabled.status_code == 403

    monkeypatch.setattr(settings, "ALLOW_SELF_REGISTRATION", True)
    registered = client.post(
        "/api/auth/register",
        json={
            "username": "self-user",
            "email": "self-user@example.test",
            "password": "self-password-123",
        },
    )

    assert registered.status_code == 201
    assert registered.json()["user"]["roles"] == ["viewer"]
    assert "access_token" not in registered.json()
    assert len(registered.headers.get_list("set-cookie")) == 3


def test_user_can_list_and_revoke_own_refresh_sessions() -> None:
    test_client = TestClient(app, base_url="https://testserver")
    assert _login("viewer", "viewer-password-123", test_client).status_code == 200

    sessions = test_client.get("/api/auth/sessions")

    assert sessions.status_code == 200
    assert len(sessions.json()) == 1
    assert "token_hash" not in sessions.json()[0]
    revoked = test_client.delete(f"/api/auth/sessions/{sessions.json()[0]['id']}")
    assert revoked.status_code == 204
    assert test_client.get("/api/auth/sessions").json() == []


def test_cookie_session_refresh_rotation_and_logout_flow() -> None:
    test_client = TestClient(app, base_url="https://testserver")
    assert _login("admin", "admin-password-123", test_client).status_code == 200

    session = test_client.get("/api/auth/session")
    assert session.status_code == 200
    assert session.json()["user"]["username"] == "admin"

    old_refresh_token = test_client.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME)
    refreshed = test_client.post("/api/auth/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["authenticated"] is True
    assert (
        test_client.cookies.get(settings.REFRESH_TOKEN_COOKIE_NAME) != old_refresh_token
    )

    replay_client = TestClient(app, base_url="https://testserver")
    replay_client.cookies.set(
        settings.REFRESH_TOKEN_COOKIE_NAME,
        old_refresh_token,
        path="/api/auth",
    )
    assert replay_client.post("/api/auth/refresh").status_code == 401
    assert test_client.get("/api/auth/session").status_code == 401

    logout = test_client.post("/api/auth/logout")
    assert logout.status_code == 204
    assert test_client.get("/api/auth/session").status_code == 401


def test_viewer_is_forbidden_from_creating_analysis() -> None:
    test_client = TestClient(app, base_url="https://testserver")
    assert _login("viewer", "viewer-password-123", test_client).status_code == 200

    response = test_client.post(
        "/api/analyze",
        json={
            "home_team": "Home",
            "away_team": "Away",
            "home_stats": {"form": 50, "attack": 50, "defense": 50, "xg": 1.2},
            "away_stats": {"form": 50, "attack": 50, "defense": 50, "xg": 1.2},
            "odd": 2.0,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Eksik yetki: analysis:create"


def test_admin_can_manage_users_and_read_role_catalog() -> None:
    test_client = TestClient(app, base_url="https://testserver")
    assert _login("admin", "admin-password-123", test_client).status_code == 200

    roles = test_client.get("/api/admin/roles")
    assert roles.status_code == 200
    assert {role["name"] for role in roles.json()} == {"admin", "viewer"}

    created = test_client.post(
        "/api/admin/users",
        json={
            "username": "managed-user",
            "email": "managed-user@example.test",
            "password": "managed-password-123",
            "roles": ["viewer"],
        },
    )
    assert created.status_code == 201
    assert created.json()["roles"] == ["viewer"]
    assert "password" not in created.json()

    updated = test_client.patch(
        f"/api/admin/users/{created.json()['id']}",
        json={"roles": ["admin"], "is_active": False},
    )
    assert updated.status_code == 200
    assert updated.json()["roles"] == ["admin"]
    assert updated.json()["is_active"] is False

    users = test_client.get("/api/admin/users")
    assert users.status_code == 200
    assert any(user["username"] == "managed-user" for user in users.json())


def test_viewer_cannot_access_user_management() -> None:
    test_client = TestClient(app, base_url="https://testserver")
    assert _login("viewer", "viewer-password-123", test_client).status_code == 200

    response = test_client.get("/api/admin/users")

    assert response.status_code == 403
    assert response.json()["detail"] == "Eksik yetki: users:manage"


def test_admin_cannot_remove_own_access() -> None:
    test_client = TestClient(app, base_url="https://testserver")
    login = _login("admin", "admin-password-123", test_client)
    admin_id = login.json()["user"]["id"]

    deactivate = test_client.patch(
        f"/api/admin/users/{admin_id}", json={"is_active": False}
    )
    remove_role = test_client.patch(
        f"/api/admin/users/{admin_id}", json={"roles": ["viewer"]}
    )

    assert deactivate.status_code == 400
    assert remove_role.status_code == 400
