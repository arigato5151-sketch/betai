"""Add historical fixture expected-goals provenance.

Revision ID: 20260801_0015
Revises: 20260801_0014
Create Date: 2026-08-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260801_0015"
down_revision: Union[str, None] = "20260801_0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "historical_fixtures", sa.Column("home_xg", sa.Float(), nullable=True)
    )
    op.add_column(
        "historical_fixtures", sa.Column("away_xg", sa.Float(), nullable=True)
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("xg_source", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("xg_provider_match_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("xg_updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("historical_fixtures", "xg_updated_at")
    op.drop_column("historical_fixtures", "xg_provider_match_id")
    op.drop_column("historical_fixtures", "xg_source")
    op.drop_column("historical_fixtures", "away_xg")
    op.drop_column("historical_fixtures", "home_xg")
