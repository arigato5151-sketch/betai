"""Add external fixture provenance and 64-bit identifiers.

Revision ID: 20260723_0009
Revises: 20260723_0008
Create Date: 2026-07-23
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0009"
down_revision: Union[str, None] = "20260723_0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "historical_fixtures",
        "fixture_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
    )
    op.alter_column(
        "historical_fixtures",
        "home_team_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
    )
    op.alter_column(
        "historical_fixtures",
        "away_team_id",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
    )
    op.add_column(
        "historical_fixtures",
        sa.Column(
            "data_source",
            sa.String(length=50),
            nullable=False,
            server_default="api_football",
        ),
    )
    op.create_index(
        "ix_historical_fixtures_data_source",
        "historical_fixtures",
        ["data_source"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_historical_fixtures_data_source",
        table_name="historical_fixtures",
    )
    op.drop_column("historical_fixtures", "data_source")
    op.alter_column(
        "historical_fixtures",
        "away_team_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
    )
    op.alter_column(
        "historical_fixtures",
        "home_team_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
    )
    op.alter_column(
        "historical_fixtures",
        "fixture_id",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
    )
