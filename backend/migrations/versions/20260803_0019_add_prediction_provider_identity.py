"""Add provider identity to predictions.

Revision ID: 20260803_0019
Revises: 20260803_0018
Create Date: 2026-08-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0019"
down_revision: Union[str, None] = "20260803_0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column("fixture_source", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("provider_fixture_id", sa.String(length=100), nullable=True),
    )
    op.create_index(
        "ix_match_predictions_fixture_source",
        "match_predictions",
        ["fixture_source"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_match_predictions_provider_fixture",
        "match_predictions",
        ["fixture_source", "provider_fixture_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_match_predictions_provider_fixture",
        "match_predictions",
        type_="unique",
    )
    op.drop_index(
        "ix_match_predictions_fixture_source",
        table_name="match_predictions",
    )
    op.drop_column("match_predictions", "provider_fixture_id")
    op.drop_column("match_predictions", "fixture_source")
