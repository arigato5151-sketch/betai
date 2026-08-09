from __future__ import annotations

import os
import secrets

from app.core.config import settings
from app.core.passwords import hash_password
from app.db.session import SessionLocal
from app.db.user_repository import UserRepository

PLACEHOLDER_PASSWORD = "change-this-password"
MIN_PASSWORD_LENGTH = 12
MIN_BOOTSTRAP_SECRET_LENGTH = 16


def _assert_bootstrap_secret() -> str:
    secret = settings.BOOTSTRAP_ADMIN_SECRET
    if not secret or len(secret) < MIN_BOOTSTRAP_SECRET_LENGTH:
        raise SystemExit(
            "BOOTSTRAP_ADMIN_SECRET must be set and at least "
            f"{MIN_BOOTSTRAP_SECRET_LENGTH} characters."
        )
    return secret


def _resolve_password() -> tuple[str, bool]:
    password = settings.ADMIN_PASSWORD
    if password is None or not password.strip():
        return secrets.token_urlsafe(MIN_PASSWORD_LENGTH), True
    if len(password) < MIN_PASSWORD_LENGTH or password == PLACEHOLDER_PASSWORD:
        raise SystemExit(
            f"ADMIN_PASSWORD must be at least {MIN_PASSWORD_LENGTH} characters "
            "and not the default placeholder."
        )
    return password, False


def main() -> None:
    username = settings.ADMIN_USERNAME.strip().lower()
    email = os.getenv("ADMIN_EMAIL", "admin@example.invalid").strip().lower()
    _assert_bootstrap_secret()
    password, generated = _resolve_password()

    with SessionLocal() as db:
        repo = UserRepository(db)
        existing = repo.get_by_identifier(username)
        if existing:
            print(f"Admin bootstrap skipped; user already exists: {username}")
            return

        repo.create_user(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role_names=["admin"],
        )
        if generated:
            print(
                f"Admin user created: {username}\n"
                f"Generated one-time password: {password}\n"
                "Change it after first sign-in."
            )
        else:
            print(f"Admin user created: {username}")


if __name__ == "__main__":
    main()
