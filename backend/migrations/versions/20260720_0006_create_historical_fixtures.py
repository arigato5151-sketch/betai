"""Create historical fixtures store.

Revision ID: 20260720_0006
Revises: 20260719_0005
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260720_0006"
down_revision: Union[str, None] = "20260719_0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "historical_fixtures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("league_id", sa.Integer(), nullable=False),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_team", sa.String(length=100), nullable=False),
        sa.Column("away_team", sa.String(length=100), nullable=False),
        sa.Column("home_goals", sa.Integer(), nullable=False),
        sa.Column("away_goals", sa.Integer(), nullable=False),
        sa.Column("actual_result", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fixture_id"),
    )
    for column in ("league_id", "season", "kickoff", "home_team_id", "away_team_id"):
        op.create_index(
            f"ix_historical_fixtures_{column}",
            "historical_fixtures",
            [column],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("historical_fixtures")
