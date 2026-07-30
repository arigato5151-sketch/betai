from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from app.core.team_identity import stable_team_name_key


def _aware_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ExternalDataPoint:
    """A numeric provider observation with auditable provenance."""

    value: float
    source: str
    captured_at: datetime
    confidence: float
    is_fallback: bool = True
    expires_at: datetime | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("external data value must be finite")
        source = self.source.strip()
        if not source:
            raise ValueError("external data source cannot be blank")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("external data confidence must be between 0 and 1")

        captured_at = _aware_utc(self.captured_at, "captured_at")
        expires_at = (
            _aware_utc(self.expires_at, "expires_at")
            if self.expires_at is not None
            else None
        )
        if expires_at is not None and expires_at <= captured_at:
            raise ValueError("expires_at must be later than captured_at")

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "captured_at", captured_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "details", dict(self.details))

    def provenance(self) -> dict[str, object]:
        return {
            "source": self.source,
            "captured_at": self.captured_at.isoformat(),
            "confidence": self.confidence,
            "is_fallback": self.is_fallback,
        }


@dataclass(frozen=True, slots=True)
class ProviderTeamCandidate:
    provider_team_key: str
    provider_team_name: str
    confidence: float

    def __post_init__(self) -> None:
        key = self.provider_team_key.strip()
        name = self.provider_team_name.strip()
        if not key or not name:
            raise ValueError("provider team key and name cannot be blank")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("provider team confidence must be between 0 and 1")
        object.__setattr__(self, "provider_team_key", key)
        object.__setattr__(self, "provider_team_name", name)

    @property
    def normalized_name(self) -> str:
        return stable_team_name_key(self.provider_team_name)
