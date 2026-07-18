from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def utc_now_naive() -> datetime.datetime:
    """Return UTC without tzinfo for the legacy timezone-naive prediction column."""
    return utc_now().replace(tzinfo=None)


user_roles = Table(
    "user_roles",
    Base.metadata,
    Column(
        "user_id",
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    ),
)

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column(
        "role_id", Integer, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    ),
    Column(
        "permission_id",
        Integer,
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    roles: Mapped[list[Role]] = relationship(
        secondary=user_roles, lazy="selectin", back_populates="users"
    )
    refresh_sessions: Mapped[list[RefreshSession]] = relationship(
        cascade="all, delete-orphan",
        back_populates="user",
    )


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    users: Mapped[list[User]] = relationship(
        secondary=user_roles, lazy="selectin", back_populates="roles"
    )
    permissions: Mapped[list[Permission]] = relationship(
        secondary=role_permissions,
        lazy="selectin",
        back_populates="roles",
    )


class Permission(Base):
    __tablename__ = "permissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    roles: Mapped[list[Role]] = relationship(
        secondary=role_permissions,
        lazy="selectin",
        back_populates="permissions",
    )


class RefreshSession(Base):
    __tablename__ = "refresh_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    user: Mapped[User] = relationship(back_populates="refresh_sessions")


class MatchPrediction(Base):
    __tablename__ = "match_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fixture_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, unique=True, index=True
    )
    home_team: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    away_team: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    home_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    league_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    home_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_xg: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_form: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_form: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_attack: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_defense: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_attack: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_defense: Mapped[float | None] = mapped_column(Float, nullable=True)

    prediction: Mapped[str | None] = mapped_column(String, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_draw: Mapped[float | None] = mapped_column(Float, nullable=True)

    odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    edge: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_value_bet: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    kelly_stake: Mapped[float | None] = mapped_column(Float, nullable=True)

    ml_cluster: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ml_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    actual_result: Mapped[str | None] = mapped_column(String, nullable=True)
    actual_score_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_score_away: Mapped[int | None] = mapped_column(Integer, nullable=True)

    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True, default=utc_now_naive, index=True
    )
