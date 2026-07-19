"""Add point-in-time ML feature snapshots.

Revision ID: 20260719_0004
Revises: 20260712_0003
Create Date: 2026-07-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260719_0004"
down_revision: Union[str, None] = "20260712_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column("feature_snapshot", sa.JSON(), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("feature_schema_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("feature_snapshot_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("match_predictions", "feature_snapshot_at")
    op.drop_column("match_predictions", "feature_schema_version")
    op.drop_column("match_predictions", "feature_snapshot")
