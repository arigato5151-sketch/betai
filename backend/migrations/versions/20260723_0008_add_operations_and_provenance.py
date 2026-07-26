"""Add operational sync tracking and prediction provenance.

Revision ID: 20260723_0008
Revises: 20260720_0007
Create Date: 2026-07-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260723_0008"
down_revision: Union[str, None] = "20260720_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_predictions", sa.Column("model_name", sa.String(100), nullable=True)
    )
    op.add_column(
        "match_predictions",
        sa.Column("model_artifact_version", sa.String(100), nullable=True),
    )
    op.add_column(
        "match_predictions", sa.Column("data_quality", sa.JSON(), nullable=True)
    )
    op.add_column(
        "match_predictions",
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("analysis_lead_minutes", sa.Float(), nullable=True),
    )
    op.add_column(
        "match_predictions",
        sa.Column("market_snapshot_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "sync_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_name", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("target_seasons", sa.JSON(), nullable=True),
        sa.Column(
            "fixtures_processed", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("failures", sa.JSON(), nullable=True),
        sa.Column("error_type", sa.String(100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sync_runs_job_name", "sync_runs", ["job_name"])
    op.create_index("ix_sync_runs_status", "sync_runs", ["status"])
    op.create_index("ix_sync_runs_started_at", "sync_runs", ["started_at"])


def downgrade() -> None:
    op.drop_index("ix_sync_runs_started_at", table_name="sync_runs")
    op.drop_index("ix_sync_runs_status", table_name="sync_runs")
    op.drop_index("ix_sync_runs_job_name", table_name="sync_runs")
    op.drop_table("sync_runs")
    op.drop_column("match_predictions", "market_snapshot_at")
    op.drop_column("match_predictions", "analysis_lead_minutes")
    op.drop_column("match_predictions", "analyzed_at")
    op.drop_column("match_predictions", "kickoff")
    op.drop_column("match_predictions", "data_quality")
    op.drop_column("match_predictions", "model_artifact_version")
    op.drop_column("match_predictions", "model_name")
