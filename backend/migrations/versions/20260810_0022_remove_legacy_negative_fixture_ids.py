"""Drop legacy negative fixture ids from the historical space.

The four natural-key sources (football_data_csv, fixture_download,
openfootball_json, statsbomb_open) used to derive negative BLAKE2b ids that
never collided with the positive namespaced prediction/odds space, so closing
odds and result verification could not attach. They now emit disjoint positive
namespaced ids from app.core.namespaced_ids; old negative rows are orphaned
and are removed here (re-ingestion rebuilds them with the canonical ids).

Revision ID: 20260810_0022
Revises: 20260809_0021
Create Date: 2026-08-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260810_0022"
down_revision: Union[str, None] = "20260809_0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    result = connection.execute(
        sa.text("SELECT fixture_id FROM historical_fixtures WHERE fixture_id < 0")
    )
    stale_ids = [row[0] for row in result.fetchall()]
    if not stale_ids:
        return
    connection.execute(
        sa.text(
            "DELETE FROM historical_player_performances "
            "WHERE fixture_id IN (SELECT fixture_id FROM historical_fixtures "
            "WHERE fixture_id < 0)"
        )
    )
    connection.execute(sa.text("DELETE FROM historical_fixtures WHERE fixture_id < 0"))


def downgrade() -> None:
    pass