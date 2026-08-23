"""Make match prediction created_at timezone-aware.

match_predictions.created_at was the only timestamp column written as a naive
UTC datetime while every neighbouring timestamp (analyzed_at, kickoff, ...) is
timezone-aware. Mixing naive and aware timestamps in comparisons and sorts
(e.g. backtest.py selecting on analyzed_at or created_at) produces subtle
timezone skew and ValueError hazards on a PostgreSQL backend. Convert the
column to timestamptz so every prediction timestamp carries UTC offset.

Revision ID: 20260816_0025
Revises: 20260814_0024
Create Date: 2026-08-16
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260816_0025"
down_revision: Union[str, None] = "20260814_0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("match_predictions") as batch_op:
        batch_op.alter_column(
            "created_at",
            existing_type=sa.DateTime(),
            type_=sa.DateTime(timezone=True),
            existing_nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("match_predictions") as batch_op:
        batch_op.alter_column(
            "created_at",
            existing_type=sa.DateTime(timezone=True),
            type_=sa.DateTime(),
            existing_nullable=True,
        )
