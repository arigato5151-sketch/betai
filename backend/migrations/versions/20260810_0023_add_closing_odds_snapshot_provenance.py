"""Track the odds snapshot that produced each closing price.

Settlement now records which pre-kickoff odds snapshot (its id and capture
time) was used as the closing price, so audits can verify the closing price
was fresh and re-derive it later.

Revision ID: 20260810_0023
Revises: 20260810_0022
Create Date: 2026-08-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260810_0023"
down_revision: Union[str, None] = "20260810_0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column("closing_odds_snapshot_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("closing_odds_snapshot_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("match_predictions", "closing_odds_snapshot_id")
    op.drop_column("match_predictions", "closing_odds_snapshot_at")