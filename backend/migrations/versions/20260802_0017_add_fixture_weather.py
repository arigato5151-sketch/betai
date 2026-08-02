"""Add match-time weather provenance to historical fixtures.

Revision ID: 20260802_0017
Revises: 20260801_0016
Create Date: 2026-08-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260802_0017"
down_revision: Union[str, None] = "20260801_0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "historical_fixtures",
        sa.Column("weather_temperature_c", sa.Float(), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("weather_precipitation_mm", sa.Float(), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("weather_wind_speed_kmh", sa.Float(), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("weather_source", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("weather_observed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "historical_fixtures",
        sa.Column("weather_updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("historical_fixtures", "weather_updated_at")
    op.drop_column("historical_fixtures", "weather_observed_at")
    op.drop_column("historical_fixtures", "weather_source")
    op.drop_column("historical_fixtures", "weather_wind_speed_kmh")
    op.drop_column("historical_fixtures", "weather_precipitation_mm")
    op.drop_column("historical_fixtures", "weather_temperature_c")
