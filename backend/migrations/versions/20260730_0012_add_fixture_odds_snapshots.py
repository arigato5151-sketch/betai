"""Add point-in-time fixture odds snapshots.

Revision ID: 20260730_0012
Revises: 20260730_0011
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_0012"
down_revision: Union[str, None] = "20260730_0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fixture_odds_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fixture_id", sa.BigInteger(), nullable=False),
        sa.Column("home_odd", sa.Float(), nullable=False),
        sa.Column("draw_odd", sa.Float(), nullable=False),
        sa.Column("away_odd", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("bookmaker", sa.String(length=100), nullable=True),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "fixture_id",
            "captured_at",
            name="uq_fixture_odds_snapshots_fixture_captured",
        ),
    )
    op.create_index(
        "ix_fixture_odds_snapshots_fixture_captured",
        "fixture_odds_snapshots",
        ["fixture_id", "captured_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_fixture_odds_snapshots_fixture_captured",
        table_name="fixture_odds_snapshots",
    )
    op.drop_table("fixture_odds_snapshots")
