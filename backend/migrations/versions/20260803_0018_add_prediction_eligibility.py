"""Add prediction origin and training eligibility.

Revision ID: 20260803_0018
Revises: 20260802_0017
Create Date: 2026-08-03
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260803_0018"
down_revision: Union[str, None] = "20260802_0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column(
            "analysis_origin",
            sa.String(length=32),
            nullable=False,
            server_default="legacy",
        ),
    )
    op.add_column(
        "match_predictions",
        sa.Column(
            "eligibility_status",
            sa.String(length=16),
            nullable=False,
            server_default="unverified",
        ),
    )
    op.add_column(
        "match_predictions",
        sa.Column(
            "training_eligible",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index(
        "ix_match_predictions_training_eligible",
        "match_predictions",
        ["training_eligible"],
        unique=False,
    )
    op.create_check_constraint(
        "ck_match_predictions_analysis_origin",
        "match_predictions",
        "analysis_origin IN ('legacy', 'manual', 'fixture_user', 'automatic', 'scenario')",
    )
    op.create_check_constraint(
        "ck_match_predictions_eligibility_status",
        "match_predictions",
        "eligibility_status IN ('eligible', 'abstain', 'unverified')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_match_predictions_eligibility_status",
        "match_predictions",
        type_="check",
    )
    op.drop_constraint(
        "ck_match_predictions_analysis_origin",
        "match_predictions",
        type_="check",
    )
    op.drop_index(
        "ix_match_predictions_training_eligible",
        table_name="match_predictions",
    )
    op.drop_column("match_predictions", "training_eligible")
    op.drop_column("match_predictions", "eligibility_status")
    op.drop_column("match_predictions", "analysis_origin")
