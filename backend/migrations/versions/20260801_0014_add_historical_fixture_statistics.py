"""Add optional historical fixture statistics and bookmaker odds.

Revision ID: 20260801_0014
Revises: 20260730_0013
Create Date: 2026-08-01
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260801_0014"
down_revision: Union[str, None] = "20260730_0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INTEGER_COLUMNS = (
    "half_time_home_goals",
    "half_time_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
)

_FLOAT_COLUMNS = (
    "opening_home_odd",
    "opening_draw_odd",
    "opening_away_odd",
    "closing_home_odd",
    "closing_draw_odd",
    "closing_away_odd",
)


def upgrade() -> None:
    for column in _INTEGER_COLUMNS:
        op.add_column(
            "historical_fixtures",
            sa.Column(column, sa.Integer(), nullable=True),
        )
    for column in _FLOAT_COLUMNS:
        op.add_column(
            "historical_fixtures",
            sa.Column(column, sa.Float(), nullable=True),
        )


def downgrade() -> None:
    for column in reversed(_FLOAT_COLUMNS):
        op.drop_column("historical_fixtures", column)
    for column in reversed(_INTEGER_COLUMNS):
        op.drop_column("historical_fixtures", column)
