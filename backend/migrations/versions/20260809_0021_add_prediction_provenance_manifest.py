"""Add versioned prediction provenance manifest.

Revision ID: 20260809_0021
Revises: 20260803_0020
Create Date: 2026-08-09
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260809_0021"
down_revision: Union[str, None] = "20260803_0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions",
        sa.Column("provenance_manifest", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("match_predictions", "provenance_manifest")
