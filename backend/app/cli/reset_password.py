from __future__ import annotations

import argparse
import getpass
import os
from datetime import datetime, timezone

from app.core.passwords import hash_password
from app.db.session import SessionLocal
from app.db.user_repository import UserRepository

MIN_PASSWORD_LENGTH = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset a Bet AI Platform user's password"
    )
    parser.add_argument(
        "--identifier",
        required=True,
        help="Username or email of the user to reset",
    )
    return parser.parse_args()


def _resolve_new_password() -> str:
    password = os.getenv("NEW_PASSWORD")
    if not password:
        password = getpass.getpass("New password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise SystemExit(
            f"Password must contain at least {MIN_PASSWORD_LENGTH} characters."
        )
    return password


def main() -> None:
    args = parse_args()
    password = _resolve_new_password()

    with SessionLocal() as db:
        repo = UserRepository(db)
        user = repo.get_by_identifier(args.identifier)
        if user is None:
            raise SystemExit(f"User not found: {args.identifier}")

        user.password_hash = hash_password(password)
        now = datetime.now(timezone.utc)
        repo.revoke_all_sessions(user, now)
        db.commit()
        db.refresh(user)
        print(
            f"Password reset for user: {user.username} "
            f"({', '.join(repo.role_names(user))})"
        )
        print("Active sessions revoked; re-login required.")


if __name__ == "__main__":
    main()