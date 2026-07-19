"""Add historical starting lineups.

Revision ID: 20260720_0007
Revises: 20260720_0006
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260720_0007"
down_revision: Union[str, None] = "20260720_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "historical_fixtures",
        sa.Column("home_starting_xi", sa.JSON(), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("away_starting_xi", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("historical_fixtures", "away_starting_xi")
    op.drop_column("historical_fixtures", "home_starting_xi")
