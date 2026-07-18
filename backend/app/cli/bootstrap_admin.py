from __future__ import annotations

import os

from app.core.config import settings
from app.core.passwords import hash_password
from app.db.session import SessionLocal
from app.db.user_repository import UserRepository


def main() -> None:
    username = settings.ADMIN_USERNAME.strip().lower()
    password = settings.ADMIN_PASSWORD
    email = os.getenv("ADMIN_EMAIL", "admin@example.invalid").strip().lower()

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
        print(f"Admin user created: {username}")


if __name__ == "__main__":
    main()
