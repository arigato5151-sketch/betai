"""Add expected-goals confidence provenance.

Revision ID: 20260801_0016
Revises: 20260801_0015
Create Date: 2026-08-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260801_0016"
down_revision: Union[str, None] = "20260801_0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "historical_fixtures",
        sa.Column("xg_confidence", sa.Float(), nullable=True),
    )
    op.execute(
        "UPDATE historical_fixtures SET xg_confidence = 0.95 "
        "WHERE xg_source = 'understat'"
    )


def downgrade() -> None:
    op.drop_column("historical_fixtures", "xg_confidence")
