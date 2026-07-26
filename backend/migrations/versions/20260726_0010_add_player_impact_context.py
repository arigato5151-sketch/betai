"""Add historical player performance and team location context.

Revision ID: 20260726_0010
Revises: 20260723_0009
Create Date: 2026-07-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260726_0010"
down_revision: Union[str, None] = "20260723_0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "historical_player_performances",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fixture_id", sa.BigInteger(), nullable=False),
        sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("started", sa.Boolean(), nullable=False),
        sa.Column("minutes", sa.Integer(), nullable=True),
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("position", sa.String(length=20), nullable=True),
        sa.Column("goals", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column(
            "source",
            sa.String(length=50),
            nullable=False,
            server_default="api_football",
        ),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["fixture_id"],
            ["historical_fixtures.fixture_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "fixture_id",
            "player_id",
            name="uq_historical_player_performances_fixture_player",
        ),
    )
    op.create_index(
        "ix_historical_player_performances_player_kickoff",
        "historical_player_performances",
        ["player_id", "kickoff"],
    )
    op.create_index(
        "ix_historical_player_performances_team_kickoff",
        "historical_player_performances",
        ["team_id", "kickoff"],
    )

    op.create_table(
        "team_locations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("data_source", sa.String(length=50), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "data_source",
            "team_id",
            name="uq_team_locations_source_team",
        ),
    )


def downgrade() -> None:
    op.drop_table("team_locations")
    op.drop_index(
        "ix_historical_player_performances_team_kickoff",
        table_name="historical_player_performances",
    )
    op.drop_index(
        "ix_historical_player_performances_player_kickoff",
        table_name="historical_player_performances",
    )
    op.drop_table("historical_player_performances")
