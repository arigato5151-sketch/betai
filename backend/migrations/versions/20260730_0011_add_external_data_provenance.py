"""Add provider mappings and external feature provenance.

Revision ID: 20260730_0011
Revises: 20260726_0010
Create Date: 2026-07-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260730_0011"
down_revision: Union[str, None] = "20260726_0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "provider_team_mappings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_source", sa.String(length=50), nullable=False),
        sa.Column("canonical_team_id", sa.BigInteger(), nullable=False),
        sa.Column("canonical_team_name", sa.String(length=100), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("provider_team_key", sa.String(length=150), nullable=False),
        sa.Column("provider_team_name", sa.String(length=150), nullable=False),
        sa.Column("normalized_name", sa.String(length=150), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "canonical_source",
            "canonical_team_id",
            "provider",
            name="uq_provider_team_mappings_canonical_provider",
        ),
        sa.UniqueConstraint(
            "provider",
            "provider_team_key",
            name="uq_provider_team_mappings_provider_key",
        ),
    )
    op.create_index(
        "ix_provider_team_mappings_canonical_team_id",
        "provider_team_mappings",
        ["canonical_team_id"],
    )
    op.create_index(
        "ix_provider_team_mappings_normalized_name",
        "provider_team_mappings",
        ["normalized_name"],
    )
    op.create_index(
        "ix_provider_team_mappings_provider",
        "provider_team_mappings",
        ["provider"],
    )

    op.create_table(
        "external_feature_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("canonical_source", sa.String(length=50), nullable=False),
        sa.Column("canonical_team_id", sa.BigInteger(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("feature_name", sa.String(length=100), nullable=False),
        sa.Column("numeric_value", sa.Float(), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "is_fallback",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "canonical_source",
            "canonical_team_id",
            "provider",
            "feature_name",
            "captured_at",
            name="uq_external_feature_snapshots_observation",
        ),
    )
    op.create_index(
        "ix_external_feature_snapshots_lookup",
        "external_feature_snapshots",
        [
            "canonical_source",
            "canonical_team_id",
            "feature_name",
            "captured_at",
        ],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_external_feature_snapshots_lookup",
        table_name="external_feature_snapshots",
    )
    op.drop_table("external_feature_snapshots")
    op.drop_index(
        "ix_provider_team_mappings_provider",
        table_name="provider_team_mappings",
    )
    op.drop_index(
        "ix_provider_team_mappings_normalized_name",
        table_name="provider_team_mappings",
    )
    op.drop_index(
        "ix_provider_team_mappings_canonical_team_id",
        table_name="provider_team_mappings",
    )
    op.drop_table("provider_team_mappings")
