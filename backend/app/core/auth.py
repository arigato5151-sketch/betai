from __future__ import annotations

import hashlib
import uuid
import secrets
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import jwt
from fastapi import Depends, HTTPException, Response, status
from fastapi.security import APIKeyCookie
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.passwords import hash_password, verify_password
from app.db.models import User
from app.db.session import get_db
from app.db.user_repository import UserRepository

access_cookie = APIKeyCookie(
    name=settings.ACCESS_TOKEN_COOKIE_NAME,
    scheme_name="CookieAuth",
    auto_error=False,
)

_DUMMY_PASSWORD_HASH = hash_password("invalid-password-placeholder")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=320)
    password: str = Field(..., min_length=8, max_length=256)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    refresh_session_id: str
    refresh_expires_at: datetime
    expires_in: int


class CurrentUser(BaseModel):
    id: str
    username: str
    email: str
    roles: list[str]
    permissions: list[str]


class SessionResponse(BaseModel):
    authenticated: bool = True
    user: CurrentUser
    expires_in: int


def _unauthorized(detail: str = "Geçersiz veya süresi dolmuş oturum.") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _create_token(
    *,
    subject: str,
    token_type: Literal["access", "refresh"],
    secret_key: str,
    expires_delta: timedelta,
    token_version: int,
    token_id: str | None = None,
) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    expires_at = now + expires_delta
    payload: dict[str, Any] = {
        "sub": subject,
        "type": token_type,
        "ver": token_version,
        "iat": now,
        "exp": expires_at,
    }
    if token_id:
        payload["jti"] = token_id
    return (
        jwt.encode(payload, secret_key, algorithm=settings.JWT_ALGORITHM),
        expires_at,
    )


def create_token_pair(user: User) -> TokenPair:
    access_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    refresh_session_id = str(uuid.uuid4())
    access_token, _ = _create_token(
        subject=user.id,
        token_type="access",
        secret_key=settings.JWT_SECRET_KEY,
        expires_delta=timedelta(minutes=access_minutes),
        token_version=user.token_version,
    )
    refresh_token, refresh_expires_at = _create_token(
        subject=user.id,
        token_type="refresh",
        secret_key=settings.JWT_REFRESH_SECRET_KEY,
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        token_version=user.token_version,
        token_id=refresh_session_id,
    )
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        refresh_session_id=refresh_session_id,
        refresh_expires_at=refresh_expires_at,
        expires_in=access_minutes * 60,
    )


def decode_token_payload(
    token: str,
    expected_type: Literal["access", "refresh"],
) -> dict[str, Any]:
    secret = (
        settings.JWT_SECRET_KEY
        if expected_type == "access"
        else settings.JWT_REFRESH_SECRET_KEY
    )
    try:
        payload = jwt.decode(token, secret, algorithms=[settings.JWT_ALGORITHM])
    except InvalidTokenError as exc:
        raise _unauthorized() from exc

    if not payload.get("sub") or payload.get("type") != expected_type:
        raise _unauthorized()
    if expected_type == "refresh" and not payload.get("jti"):
        raise _unauthorized()
    return payload


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def authenticate_user(db: Session, identifier: str, password: str) -> User | None:
    user = UserRepository(db).get_by_identifier(identifier)
    password_hash = user.password_hash if user else _DUMMY_PASSWORD_HASH
    valid_password = verify_password(password, password_hash)
    if not user or not valid_password or not user.is_active:
        return None
    return user


def current_user_from_model(user: User) -> CurrentUser:
    return CurrentUser(
        id=user.id,
        username=user.username,
        email=user.email,
        roles=UserRepository.role_names(user),
        permissions=UserRepository.permission_codes(user),
    )


def issue_token_pair(
    db: Session,
    user: User,
    *,
    user_agent: str | None,
    ip_address: str | None,
) -> TokenPair:
    tokens = create_token_pair(user)
    UserRepository(db).create_refresh_session(
        session_id=tokens.refresh_session_id,
        user_id=user.id,
        token_hash=token_hash(tokens.refresh_token),
        expires_at=tokens.refresh_expires_at,
        user_agent=user_agent,
        ip_address=ip_address,
    )
    return tokens


def rotate_refresh_token(
    db: Session,
    refresh_token: str,
    *,
    user_agent: str | None,
    ip_address: str | None,
) -> tuple[User, TokenPair]:
    payload = decode_token_payload(refresh_token, "refresh")
    repo = UserRepository(db)
    session = repo.get_refresh_session(token_hash(refresh_token))
    now = datetime.now(timezone.utc)

    if not session or session.id != payload["jti"]:
        raise _unauthorized("Refresh oturumu geçersiz veya daha önce kullanılmış.")
    if session.revoked_at is not None:
        compromised_user = repo.get_by_id(str(payload["sub"]))
        if compromised_user:
            # Reuse detection invalidates the whole token family via token_version.
            repo.revoke_all_sessions(compromised_user, now)
        raise _unauthorized("Refresh token tekrar kullanımı algılandı.")

    expires_at = session.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= now:
        repo.revoke_refresh_session(session, now)
        raise _unauthorized()

    user = repo.get_by_id(str(payload["sub"]))
    if (
        not user
        or not user.is_active
        or int(payload.get("ver", -1)) != user.token_version
    ):
        raise _unauthorized()

    repo.revoke_refresh_session(session, now)
    return user, issue_token_pair(
        db,
        user,
        user_agent=user_agent,
        ip_address=ip_address,
    )


def revoke_refresh_token(db: Session, refresh_token: str | None) -> None:
    if not refresh_token:
        return
    session = UserRepository(db).get_refresh_session(token_hash(refresh_token))
    if session and session.revoked_at is None:
        UserRepository(db).revoke_refresh_session(session, datetime.now(timezone.utc))


def set_auth_cookies(response: Response, tokens: TokenPair) -> None:
    response.set_cookie(
        key=settings.ACCESS_TOKEN_COOKIE_NAME,
        value=tokens.access_token,
        max_age=tokens.expires_in,
        path="/",
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
    )
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=secrets.token_urlsafe(32),
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/",
        httponly=False,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
    )
    response.set_cookie(
        key=settings.REFRESH_TOKEN_COOKIE_NAME,
        value=tokens.refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/api/auth",
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
    )


def clear_auth_cookies(response: Response) -> None:
    for cookie_name, path in (
        (settings.ACCESS_TOKEN_COOKIE_NAME, "/"),
        (settings.REFRESH_TOKEN_COOKIE_NAME, "/api/auth"),
        (settings.CSRF_COOKIE_NAME, "/"),
    ):
        response.delete_cookie(
            cookie_name,
            path=path,
            domain=settings.COOKIE_DOMAIN,
            secure=settings.COOKIE_SECURE,
            httponly=True,
            samesite=settings.COOKIE_SAMESITE,
        )


def require_authenticated_user(
    token: str | None = Depends(access_cookie),
    db: Session = Depends(get_db),
) -> CurrentUser:
    if not token:
        raise _unauthorized("Oturum çerezi bulunamadı.")
    payload = decode_token_payload(token, "access")
    user = UserRepository(db).get_by_id(str(payload["sub"]))
    if (
        not user
        or not user.is_active
        or int(payload.get("ver", -1)) != user.token_version
    ):
        raise _unauthorized()
    return current_user_from_model(user)


def require_permission(permission: str) -> Callable[..., CurrentUser]:
    def dependency(
        user: CurrentUser = Depends(require_authenticated_user),
    ) -> CurrentUser:
        if permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Eksik yetki: {permission}",
            )
        return user

    return dependency
