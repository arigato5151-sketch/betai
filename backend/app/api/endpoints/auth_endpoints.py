from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.endpoints.schemas import RegistrationRequest
from app.core.auth import (
    LoginRequest,
    SessionResponse,
    authenticate_user,
    clear_auth_cookies,
    current_user_from_model,
    issue_token_pair,
    require_authenticated_user,
    revoke_refresh_token,
    rotate_refresh_token,
    set_auth_cookies,
)
from app.core.config import settings
from app.core.passwords import hash_password
from app.core.rate_limit import login_rate_limiter, registration_rate_limiter
from app.db.session import get_db
from app.db.user_repository import UserRepository

router = APIRouter()


@router.post(
    "/auth/register",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    body: RegistrationRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    if not settings.ALLOW_SELF_REGISTRATION:
        raise HTTPException(status_code=403, detail="Kullanıcı kaydı devre dışı.")
    ip_address = request.client.host if request.client else "unknown"
    allowed, retry_after = registration_rate_limiter.consume(ip_address)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Çok fazla kayıt denemesi.",
            headers={"Retry-After": str(retry_after)},
        )
    try:
        user = UserRepository(db).create_user(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
            role_names=[settings.SELF_REGISTRATION_ROLE],
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Kullanıcı adı veya e-posta zaten kullanımda.",
        ) from exc
    tokens = issue_token_pair(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=ip_address,
    )
    set_auth_cookies(response, tokens)
    return SessionResponse(
        user=current_user_from_model(user),
        expires_in=tokens.expires_in,
    )


@router.post("/auth/login", response_model=SessionResponse)
def login(
    body: LoginRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    ip_address = request.client.host if request.client else "unknown"
    retry_after = login_rate_limiter.retry_after(body.username, ip_address)
    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail="Çok fazla başarısız giriş denemesi.",
            headers={"Retry-After": str(retry_after)},
        )
    user = authenticate_user(db, body.username, body.password)
    if not user:
        login_rate_limiter.record_failure(body.username, ip_address)
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya parola hatalı.")
    login_rate_limiter.reset(body.username, ip_address)
    tokens = issue_token_pair(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=ip_address,
    )
    set_auth_cookies(response, tokens)
    return SessionResponse(
        user=current_user_from_model(user),
        expires_in=tokens.expires_in,
    )


@router.post("/auth/refresh", response_model=SessionResponse)
def refresh_access_token(
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
    refresh_token: str | None = Cookie(
        default=None, alias=settings.REFRESH_TOKEN_COOKIE_NAME
    ),
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh çerezi bulunamadı.")
    user, tokens = rotate_refresh_token(
        db,
        refresh_token,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    set_auth_cookies(response, tokens)
    return SessionResponse(
        user=current_user_from_model(user),
        expires_in=tokens.expires_in,
    )


@router.get("/auth/session", response_model=SessionResponse)
def get_session(user=Depends(require_authenticated_user)):
    return SessionResponse(
        user=user,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/auth/sessions")
def list_auth_sessions(
    user=Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    sessions = UserRepository(db).list_active_sessions(user.id)
    return [
        {
            "id": session.id,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "user_agent": session.user_agent,
            "ip_address": session.ip_address,
        }
        for session in sessions
    ]


@router.delete("/auth/sessions/{session_id}", status_code=204)
def revoke_auth_session(
    session_id: str,
    user=Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    revoked = UserRepository(db).revoke_session_by_id(
        user.id, session_id, datetime.now(timezone.utc)
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="Aktif oturum bulunamadı.")


@router.post("/auth/logout", status_code=204)
def logout(
    response: Response,
    db: Session = Depends(get_db),
    refresh_token: str | None = Cookie(
        default=None, alias=settings.REFRESH_TOKEN_COOKIE_NAME
    ),
):
    revoke_refresh_token(db, refresh_token)
    clear_auth_cookies(response)
