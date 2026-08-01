from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Table,
    UniqueConstraint,
    Index,
    false,
    true,
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
    feature_snapshot: Mapped[dict[str, float] | None] = mapped_column(
        JSON, nullable=True
    )
    feature_schema_version: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    feature_snapshot_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    probability_components: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True
    )
    ensemble_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_artifact_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    data_quality: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    kickoff: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    analyzed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=utc_now
    )
    analysis_lead_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_snapshot_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    actual_result: Mapped[str | None] = mapped_column(String, nullable=True)
    actual_score_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_score_away: Mapped[int | None] = mapped_column(Integer, nullable=True)

    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clv: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True, default=utc_now_naive, index=True
    )


class HistoricalFixture(Base):
    __tablename__ = "historical_fixtures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    league_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    kickoff: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    home_team_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    away_team_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    half_time_home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    half_time_away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_yellow_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_red_cards: Mapped[int | None] = mapped_column(Integer, nullable=True)
    opening_home_odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    opening_draw_odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    opening_away_odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_home_odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_draw_odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_away_odd: Mapped[float | None] = mapped_column(Float, nullable=True)
    home_starting_xi: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    away_starting_xi: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    actual_result: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(8), nullable=False)
    data_source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="api_football",
        server_default="api_football",
        index=True,
    )
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class HistoricalPlayerPerformance(Base):
    __tablename__ = "historical_player_performances"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "player_id",
            name="uq_historical_player_performances_fixture_player",
        ),
        Index(
            "ix_historical_player_performances_team_kickoff",
            "team_id",
            "kickoff",
        ),
        Index(
            "ix_historical_player_performances_player_kickoff",
            "player_id",
            "kickoff",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("historical_fixtures.fixture_id", ondelete="CASCADE"),
        nullable=False,
    )
    league_id: Mapped[int] = mapped_column(Integer, nullable=False)
    kickoff: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    started: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    position: Mapped[str | None] = mapped_column(String(20), nullable=True)
    goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="api_football",
        server_default="api_football",
    )
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class TeamLocation(Base):
    __tablename__ = "team_locations"
    __table_args__ = (
        UniqueConstraint(
            "data_source",
            "team_id",
            name="uq_team_locations_source_team",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    data_source: Mapped[str] = mapped_column(String(50), nullable=False)
    team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    location_source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="manual",
        server_default="manual",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        server_default="1.0",
    )
    details: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class ProviderTeamMapping(Base):
    __tablename__ = "provider_team_mappings"
    __table_args__ = (
        UniqueConstraint(
            "canonical_source",
            "canonical_team_id",
            "provider",
            name="uq_provider_team_mappings_canonical_provider",
        ),
        UniqueConstraint(
            "provider",
            "provider_team_key",
            name="uq_provider_team_mappings_provider_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_source: Mapped[str] = mapped_column(String(50), nullable=False)
    canonical_team_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, index=True
    )
    canonical_team_name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    provider_team_key: Mapped[str] = mapped_column(String(150), nullable=False)
    provider_team_name: Mapped[str] = mapped_column(String(150), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(150), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class ExternalFeatureSnapshot(Base):
    __tablename__ = "external_feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "canonical_source",
            "canonical_team_id",
            "provider",
            "feature_name",
            "captured_at",
            name="uq_external_feature_snapshots_observation",
        ),
        Index(
            "ix_external_feature_snapshots_lookup",
            "canonical_source",
            "canonical_team_id",
            "feature_name",
            "captured_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_source: Mapped[str] = mapped_column(String(50), nullable=False)
    canonical_team_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    numeric_value: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_fallback: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    details: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class FixtureOddsSnapshot(Base):
    __tablename__ = "fixture_odds_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "fixture_id",
            "captured_at",
            name="uq_fixture_odds_snapshots_fixture_captured",
        ),
        Index(
            "ix_fixture_odds_snapshots_fixture_captured",
            "fixture_id",
            "captured_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fixture_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    home_odd: Mapped[float] = mapped_column(Float, nullable=False)
    draw_odd: Mapped[float] = mapped_column(Float, nullable=False)
    away_odd: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    bookmaker: Mapped[str | None] = mapped_column(String(100), nullable=True)
    captured_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    details: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class SyncRun(Base):
    __tablename__ = "sync_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, index=True
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    target_seasons: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    fixtures_processed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    failures: Mapped[list[dict[str, object]] | None] = mapped_column(
        JSON, nullable=True
    )
    error_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
