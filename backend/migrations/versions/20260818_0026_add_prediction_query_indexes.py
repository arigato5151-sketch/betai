"""Add composite and single-column indexes for prediction query hotspots.

The three most frequent filter combinations in production:
  - search_history filters on (training_eligible, actual_result) together
  - get_all_auditable filters on (training_eligible, actual_result) together
  - search_history filters on is_value_bet
  - time-based queries filter on kickoff

Revision ID: 20260818_0026
Revises: 20260816_0025
Create Date: 2026-08-18
"""

from typing import Sequence, Union

from alembic import op

revision: str = "20260818_0026"
down_revision: Union[str, None] = "20260816_0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_match_predictions_eligible_actual",
        "match_predictions",
        ["training_eligible", "actual_result"],
    )
    op.create_index(
        "ix_match_predictions_is_value_bet",
        "match_predictions",
        ["is_value_bet"],
    )
    op.create_index(
        "ix_match_predictions_kickoff",
        "match_predictions",
        ["kickoff"],
    )


def downgrade() -> None:
    op.drop_index("ix_match_predictions_kickoff", "match_predictions")
    op.drop_index("ix_match_predictions_is_value_bet", "match_predictions")
    op.drop_index("ix_match_predictions_eligible_actual", "match_predictions")
