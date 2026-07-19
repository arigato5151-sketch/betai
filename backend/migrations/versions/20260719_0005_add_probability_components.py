"""Add ensemble probability component snapshots.

Revision ID: 20260719_0005
Revises: 20260719_0004
Create Date: 2026-07-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260719_0005"
down_revision: Union[str, None] = "20260719_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column("probability_components", sa.JSON(), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("ensemble_version", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("match_predictions", "ensemble_version")
    op.drop_column("match_predictions", "probability_components")
