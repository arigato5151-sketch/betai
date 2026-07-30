"""Add provenance to team locations.

Revision ID: 20260730_0013
Revises: 20260730_0012
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_0013"
down_revision: Union[str, None] = "20260730_0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "team_locations",
        sa.Column(
            "location_source",
            sa.String(length=50),
            nullable=False,
            server_default="manual",
        ),
    )
    op.add_column(
        "team_locations",
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=False,
            server_default="1.0",
        ),
    )
    op.add_column(
        "team_locations",
        sa.Column("details", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("team_locations", "details")
    op.drop_column("team_locations", "confidence")
    op.drop_column("team_locations", "location_source")
