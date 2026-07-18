from __future__ import annotations

import argparse
import getpass
import os

from app.core.passwords import hash_password
from app.db.session import SessionLocal
from app.db.user_repository import UserRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Bet AI Platform user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument(
        "--role",
        action="append",
        choices=["admin", "analyst", "viewer"],
        required=True,
        dest="roles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.getenv("NEW_USER_PASSWORD") or getpass.getpass("Password: ")
    if len(password) < 12:
        raise SystemExit("Password must contain at least 12 characters.")

    with SessionLocal() as db:
        repo = UserRepository(db)
        if repo.get_by_identifier(args.username) or repo.get_by_identifier(args.email):
            raise SystemExit("Username or email already exists.")
        user = repo.create_user(
            username=args.username,
            email=args.email,
            password_hash=hash_password(password),
            role_names=args.roles,
        )
        print(f"User created: {user.username} ({', '.join(repo.role_names(user))})")


if __name__ == "__main__":
    main()
