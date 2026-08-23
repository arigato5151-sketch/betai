"""Widen prediction fixture and team identifiers to 64-bit integers.

Offline providers use deterministic identifiers that can exceed PostgreSQL's
signed 32-bit INTEGER range. Prediction identity columns must match the
BIGINT types already used by historical fixtures.

Revision ID: 20260814_0024
Revises: 20260810_0023
Create Date: 2026-08-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260814_0024"
down_revision: Union[str, None] = "20260810_0023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("match_predictions") as batch_op:
        for column_name in ("fixture_id", "home_team_id", "away_team_id"):
            batch_op.alter_column(
                column_name,
                existing_type=sa.Integer(),
                type_=sa.BigInteger(),
                existing_nullable=True,
            )


def downgrade() -> None:
    with op.batch_alter_table("match_predictions") as batch_op:
        for column_name in ("away_team_id", "home_team_id", "fixture_id"):
            batch_op.alter_column(
                column_name,
                existing_type=sa.BigInteger(),
                type_=sa.Integer(),
                existing_nullable=True,
            )
