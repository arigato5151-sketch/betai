"""Add verified result provenance to predictions.

Revision ID: 20260803_0020
Revises: 20260803_0019
Create Date: 2026-08-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0020"
down_revision: Union[str, None] = "20260803_0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column(
            "result_verification_status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
    )
    op.add_column(
        "match_predictions",
        sa.Column("result_source", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("result_provider_fixture_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("result_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("result_verification_note", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_match_predictions_result_verification_status",
        "match_predictions",
        ["result_verification_status"],
        unique=False,
    )
    op.create_check_constraint(
        "ck_match_predictions_result_verification_status",
        "match_predictions",
        "result_verification_status IN "
        "('pending', 'verified', 'manual', 'conflict', 'rejected')",
    )

    # Existing labels have no machine-verifiable provenance. Keep them visible,
    # but do not silently admit them into training or performance reporting.
    op.execute(
        sa.text(
            "UPDATE match_predictions SET result_verification_status = 'manual', "
            "result_source = 'legacy' WHERE actual_result IS NOT NULL"
        )
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_match_predictions_result_verification_status",
        "match_predictions",
        type_="check",
    )
    op.drop_index(
        "ix_match_predictions_result_verification_status",
        table_name="match_predictions",
    )
    op.drop_column("match_predictions", "result_verification_note")
    op.drop_column("match_predictions", "result_verified_at")
    op.drop_column("match_predictions", "result_provider_fixture_id")
    op.drop_column("match_predictions", "result_source")
    op.drop_column("match_predictions", "result_verification_status")
