from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import RefreshSession, Role, User


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> User | None:
        return self.db.query(User).filter(User.id == user_id).first()

    def get_by_identifier(self, identifier: str) -> User | None:
        normalized = identifier.strip().lower()
        return (
            self.db.query(User)
            .filter(
                or_(
                    User.username == normalized,
                    User.email == normalized,
                )
            )
            .first()
        )

    def create_user(
        self,
        *,
        username: str,
        email: str,
        password_hash: str,
        role_names: list[str],
    ) -> User:
        normalized_roles = sorted(set(role_names))
        roles = self.db.query(Role).filter(Role.name.in_(normalized_roles)).all()
        found = {role.name for role in roles}
        missing = set(normalized_roles) - found
        if missing:
            raise ValueError(f"Bilinmeyen roller: {', '.join(sorted(missing))}")

        user = User(
            username=username.strip().lower(),
            email=email.strip().lower(),
            password_hash=password_hash,
            roles=roles,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def list_users(self) -> list[User]:
        return (
            self.db.query(User)
            .options(selectinload(User.roles).selectinload(Role.permissions))
            .order_by(User.username.asc())
            .all()
        )

    def list_roles(self) -> list[Role]:
        return (
            self.db.query(Role)
            .options(selectinload(Role.permissions))
            .order_by(Role.name.asc())
            .all()
        )

    def update_access(
        self,
        user: User,
        *,
        role_names: list[str] | None,
        is_active: bool | None,
    ) -> User:
        if role_names is not None:
            normalized_roles = sorted(set(role_names))
            roles = self.db.query(Role).filter(Role.name.in_(normalized_roles)).all()
            found = {role.name for role in roles}
            missing = set(normalized_roles) - found
            if missing:
                raise ValueError(f"Bilinmeyen roller: {', '.join(sorted(missing))}")
            user.roles = roles

        if is_active is not None:
            user.is_active = is_active

        now = datetime.now(timezone.utc)
        self.db.query(RefreshSession).filter(
            RefreshSession.user_id == user.id,
            RefreshSession.revoked_at.is_(None),
        ).update({RefreshSession.revoked_at: now})
        user.token_version += 1
        self.db.commit()
        self.db.refresh(user)
        return user

    @staticmethod
    def role_names(user: User) -> list[str]:
        return sorted(role.name for role in user.roles)

    @staticmethod
    def permission_codes(user: User) -> list[str]:
        return sorted(
            {permission.code for role in user.roles for permission in role.permissions}
        )

    def create_refresh_session(
        self,
        *,
        session_id: str,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
        user_agent: str | None,
        ip_address: str | None,
    ) -> RefreshSession:
        session = RefreshSession(
            id=session_id,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            user_agent=user_agent[:512] if user_agent else None,
            ip_address=ip_address[:64] if ip_address else None,
        )
        self.db.add(session)
        self.db.commit()
        return session

    def get_refresh_session(self, token_hash: str) -> RefreshSession | None:
        return (
            self.db.query(RefreshSession)
            .filter(RefreshSession.token_hash == token_hash)
            .first()
        )

    def list_active_sessions(self, user_id: str) -> list[RefreshSession]:
        now = datetime.now(timezone.utc)
        return (
            self.db.query(RefreshSession)
            .filter(
                RefreshSession.user_id == user_id,
                RefreshSession.revoked_at.is_(None),
                RefreshSession.expires_at > now,
            )
            .order_by(RefreshSession.created_at.desc())
            .all()
        )

    def revoke_session_by_id(
        self, user_id: str, session_id: str, revoked_at: datetime
    ) -> bool:
        session = (
            self.db.query(RefreshSession)
            .filter(
                RefreshSession.id == session_id,
                RefreshSession.user_id == user_id,
                RefreshSession.revoked_at.is_(None),
            )
            .first()
        )
        if session is None:
            return False
        session.revoked_at = revoked_at
        self.db.commit()
        return True

    def revoke_refresh_session(
        self, session: RefreshSession, revoked_at: datetime
    ) -> None:
        session.revoked_at = revoked_at
        self.db.commit()

    def revoke_all_sessions(self, user: User, revoked_at: datetime) -> None:
        self.db.query(RefreshSession).filter(
            RefreshSession.user_id == user.id,
            RefreshSession.revoked_at.is_(None),
        ).update({RefreshSession.revoked_at: revoked_at})
        user.token_version += 1
        self.db.commit()
