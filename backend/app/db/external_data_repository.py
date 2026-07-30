from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.team_identity import stable_team_name_key
from app.db.models import ExternalFeatureSnapshot, ProviderTeamMapping, utc_now
from app.providers.base import ExternalDataPoint, ProviderTeamCandidate


def _positive_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("canonical_team_id must be a positive integer")
    return value


def _clean(value: str, label: str, maximum: int) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} cannot be blank")
    if len(cleaned) > maximum:
        raise ValueError(f"{label} cannot exceed {maximum} characters")
    return cleaned


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class ExternalDataRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_mapping(
        self,
        *,
        canonical_source: str,
        canonical_team_id: int,
        provider: str,
    ) -> ProviderTeamMapping | None:
        return (
            self.db.query(ProviderTeamMapping)
            .filter(
                ProviderTeamMapping.canonical_source
                == _clean(canonical_source, "canonical_source", 50),
                ProviderTeamMapping.canonical_team_id
                == _positive_id(canonical_team_id),
                ProviderTeamMapping.provider == _clean(provider, "provider", 50),
            )
            .one_or_none()
        )

    def upsert_mapping(
        self,
        *,
        canonical_source: str,
        canonical_team_id: int,
        canonical_team_name: str,
        provider: str,
        candidate: ProviderTeamCandidate,
        verified: bool = False,
    ) -> ProviderTeamMapping:
        canonical_source = _clean(canonical_source, "canonical_source", 50)
        canonical_team_id = _positive_id(canonical_team_id)
        canonical_team_name = _clean(canonical_team_name, "canonical_team_name", 100)
        provider = _clean(provider, "provider", 50)
        now = utc_now()
        values: dict[str, object] = {
            "canonical_source": canonical_source,
            "canonical_team_id": canonical_team_id,
            "canonical_team_name": canonical_team_name,
            "provider": provider,
            "provider_team_key": candidate.provider_team_key,
            "provider_team_name": candidate.provider_team_name,
            "normalized_name": stable_team_name_key(candidate.provider_team_name),
            "confidence": candidate.confidence,
            "verified": bool(verified),
            "created_at": now,
            "updated_at": now,
        }
        update_values = {
            key: value for key, value in values.items() if key != "created_at"
        }
        self._upsert(
            ProviderTeamMapping,
            values,
            conflict_columns=(
                "canonical_source",
                "canonical_team_id",
                "provider",
            ),
            update_values=update_values,
        )
        mapping = self.get_mapping(
            canonical_source=canonical_source,
            canonical_team_id=canonical_team_id,
            provider=provider,
        )
        if mapping is None:  # pragma: no cover - database contract guard
            raise RuntimeError("provider mapping was not persisted")
        return mapping

    def get_latest_snapshot(
        self,
        *,
        canonical_source: str,
        canonical_team_id: int,
        feature_name: str,
        provider: str | None = None,
        at: datetime | None = None,
    ) -> ExternalFeatureSnapshot | None:
        effective_at = _utc(at or utc_now())
        query = self.db.query(ExternalFeatureSnapshot).filter(
            ExternalFeatureSnapshot.canonical_source
            == _clean(canonical_source, "canonical_source", 50),
            ExternalFeatureSnapshot.canonical_team_id
            == _positive_id(canonical_team_id),
            ExternalFeatureSnapshot.feature_name
            == _clean(feature_name, "feature_name", 100),
            ExternalFeatureSnapshot.captured_at <= effective_at,
        )
        if provider is not None:
            query = query.filter(
                ExternalFeatureSnapshot.provider == _clean(provider, "provider", 50)
            )
        rows = query.order_by(
            ExternalFeatureSnapshot.captured_at.desc(),
            ExternalFeatureSnapshot.id.desc(),
        ).all()
        for row in rows:
            if row.expires_at is None or _utc(row.expires_at) > effective_at:
                return row
        return None

    def save_snapshot(
        self,
        *,
        canonical_source: str,
        canonical_team_id: int,
        feature_name: str,
        point: ExternalDataPoint,
    ) -> ExternalFeatureSnapshot:
        values: dict[str, object] = {
            "canonical_source": _clean(canonical_source, "canonical_source", 50),
            "canonical_team_id": _positive_id(canonical_team_id),
            "provider": _clean(point.source, "provider", 50),
            "feature_name": _clean(feature_name, "feature_name", 100),
            "numeric_value": point.value,
            "captured_at": point.captured_at,
            "expires_at": point.expires_at,
            "confidence": point.confidence,
            "is_fallback": point.is_fallback,
            "details": dict(point.details),
            "created_at": utc_now(),
        }
        self._upsert(
            ExternalFeatureSnapshot,
            values,
            conflict_columns=(
                "canonical_source",
                "canonical_team_id",
                "provider",
                "feature_name",
                "captured_at",
            ),
            update_values={
                key: value for key, value in values.items() if key != "created_at"
            },
        )
        row = (
            self.db.query(ExternalFeatureSnapshot)
            .filter(
                ExternalFeatureSnapshot.canonical_source == values["canonical_source"],
                ExternalFeatureSnapshot.canonical_team_id
                == values["canonical_team_id"],
                ExternalFeatureSnapshot.provider == values["provider"],
                ExternalFeatureSnapshot.feature_name == values["feature_name"],
                ExternalFeatureSnapshot.captured_at == values["captured_at"],
            )
            .one_or_none()
        )
        if row is None:  # pragma: no cover - database contract guard
            raise RuntimeError("external feature snapshot was not persisted")
        return row

    @staticmethod
    def to_data_point(row: ExternalFeatureSnapshot) -> ExternalDataPoint:
        if not math.isfinite(row.numeric_value):
            raise ValueError("stored external feature value is not finite")
        details = row.details if isinstance(row.details, Mapping) else {}
        return ExternalDataPoint(
            value=row.numeric_value,
            source=row.provider,
            captured_at=_utc(row.captured_at),
            expires_at=_utc(row.expires_at) if row.expires_at else None,
            confidence=row.confidence,
            is_fallback=row.is_fallback,
            details=details,
        )

    def _upsert(
        self,
        model: type[ProviderTeamMapping] | type[ExternalFeatureSnapshot],
        values: Mapping[str, Any],
        *,
        conflict_columns: tuple[str, ...],
        update_values: Mapping[str, Any],
    ) -> None:
        dialect = self.db.get_bind().dialect.name
        try:
            if dialect == "postgresql":
                postgres_statement = pg_insert(model).values(**values)
                postgres_statement = postgres_statement.on_conflict_do_update(
                    index_elements=list(conflict_columns),
                    set_=dict(update_values),
                )
                self.db.execute(postgres_statement)
            elif dialect == "sqlite":
                sqlite_statement = sqlite_insert(model).values(**values)
                sqlite_statement = sqlite_statement.on_conflict_do_update(
                    index_elements=list(conflict_columns),
                    set_=dict(update_values),
                )
                self.db.execute(sqlite_statement)
            else:
                raise RuntimeError(f"unsupported database dialect: {dialect}")
            self.db.commit()
        except (SQLAlchemyError, RuntimeError):
            self.db.rollback()
            raise
