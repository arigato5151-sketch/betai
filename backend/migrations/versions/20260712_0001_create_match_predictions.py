"""Create match predictions baseline.

Revision ID: 20260712_0001
Revises:
Create Date: 2026-07-12
"""

from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260712_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "match_predictions"


def build_columns() -> list[sa.Column[Any]]:
    return [
        sa.Column("fixture_id", sa.Integer(), nullable=True),
        sa.Column("home_team", sa.String(), nullable=True),
        sa.Column("away_team", sa.String(), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=True),
        sa.Column("away_team_id", sa.Integer(), nullable=True),
        sa.Column("league_id", sa.Integer(), nullable=True),
        sa.Column("home_xg", sa.Float(), nullable=True),
        sa.Column("away_xg", sa.Float(), nullable=True),
        sa.Column("home_form", sa.Float(), nullable=True),
        sa.Column("away_form", sa.Float(), nullable=True),
        sa.Column("home_attack", sa.Float(), nullable=True),
        sa.Column("home_defense", sa.Float(), nullable=True),
        sa.Column("away_attack", sa.Float(), nullable=True),
        sa.Column("away_defense", sa.Float(), nullable=True),
        sa.Column("prediction", sa.String(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("prob_home", sa.Float(), nullable=True),
        sa.Column("prob_away", sa.Float(), nullable=True),
        sa.Column("prob_draw", sa.Float(), nullable=True),
        sa.Column("odd", sa.Float(), nullable=True),
        sa.Column("edge", sa.Float(), nullable=True),
        sa.Column("is_value_bet", sa.Integer(), nullable=True),
        sa.Column("kelly_stake", sa.Float(), nullable=True),
        sa.Column("ml_cluster", sa.Integer(), nullable=True),
        sa.Column("ml_confidence", sa.Float(), nullable=True),
        sa.Column("actual_result", sa.String(), nullable=True),
        sa.Column("actual_score_home", sa.Integer(), nullable=True),
        sa.Column("actual_score_away", sa.Integer(), nullable=True),
        sa.Column("roi", sa.Float(), nullable=True),
        sa.Column("closing_odds", sa.Float(), nullable=True),
        sa.Column("clv", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    ]


INDEXES = {
    "ix_match_predictions_id": ["id"],
    "ix_match_predictions_home_team": ["home_team"],
    "ix_match_predictions_away_team": ["away_team"],
    "ix_match_predictions_league_id": ["league_id"],
    "ix_match_predictions_created_at": ["created_at"],
}


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if TABLE_NAME not in inspector.get_table_names():
        op.create_table(
            TABLE_NAME,
            sa.Column("id", sa.Integer(), nullable=False),
            *build_columns(),
            sa.PrimaryKeyConstraint("id"),
        )
    else:
        existing_columns = {
            column["name"] for column in inspector.get_columns(TABLE_NAME)
        }
        for column in build_columns():
            if column.name not in existing_columns:
                op.add_column(TABLE_NAME, column)

    inspector = sa.inspect(connection)
    existing_indexes = {
        index["name"] for index in inspector.get_indexes(TABLE_NAME) if index["name"]
    }
    for index_name, columns in INDEXES.items():
        if index_name not in existing_indexes:
            op.create_index(index_name, TABLE_NAME, columns, unique=False)

    if "ix_match_predictions_fixture_id" not in existing_indexes:
        op.create_index(
            "ix_match_predictions_fixture_id",
            TABLE_NAME,
            ["fixture_id"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_table(TABLE_NAME)
