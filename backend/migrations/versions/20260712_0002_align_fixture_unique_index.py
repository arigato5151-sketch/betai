"""Align fixture uniqueness with SQLAlchemy metadata.

Revision ID: 20260712_0002
Revises: 20260712_0001
Create Date: 2026-07-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260712_0002"
down_revision: Union[str, None] = "20260712_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "match_predictions"
INDEX_NAME = "ix_match_predictions_fixture_id"
LEGACY_CONSTRAINT = "uq_match_predictions_fixture_id"


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    unique_constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints(TABLE_NAME)
        if constraint["name"]
    }
    indexes = {
        index["name"]: bool(index.get("unique"))
        for index in inspector.get_indexes(TABLE_NAME)
        if index["name"]
    }

    if LEGACY_CONSTRAINT in unique_constraints:
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            batch_op.drop_constraint(LEGACY_CONSTRAINT, type_="unique")

    if INDEX_NAME in indexes and not indexes[INDEX_NAME]:
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
        indexes.pop(INDEX_NAME)

    if INDEX_NAME not in indexes:
        op.create_index(INDEX_NAME, TABLE_NAME, ["fixture_id"], unique=True)


def downgrade() -> None:
    # Revision 0001 now declares the same unique index; no schema reversal is needed.
    pass
