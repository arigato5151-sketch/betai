from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.providers.base import ExternalDataPoint, ProviderTeamCandidate
from app.services.external_features import ExternalFeatureService


class FakeClubEloClient:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve_elo(
        self,
        *,
        team_name: str,
        as_of: datetime,
        provider_team_key: str | None = None,
    ) -> tuple[ProviderTeamCandidate, ExternalDataPoint]:
        self.calls += 1
        observed_at = datetime.now(UTC)
        return (
            ProviderTeamCandidate(
                provider_team_key=provider_team_key or "Fenerbahce",
                provider_team_name="Fenerbahce",
                confidence=0.8,
            ),
            ExternalDataPoint(
                value=1742.25,
                source="clubelo",
                captured_at=observed_at,
                expires_at=observed_at + timedelta(hours=24),
                confidence=0.8,
            ),
        )


@pytest.mark.asyncio
async def test_external_feature_service_persists_and_reuses_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    provider = FakeClubEloClient()
    service = ExternalFeatureService(
        session_factory=factory,
        clubelo=provider,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(settings, "CLUBELO_ENABLED", True)

    first = await service.get_team_elo(
        canonical_team_id=611,
        canonical_team_name="Fenerbahçe",
        as_of=datetime(2026, 7, 30, 18, tzinfo=UTC),
    )
    second = await service.get_team_elo(
        canonical_team_id=611,
        canonical_team_name="Fenerbahçe",
        as_of=datetime(2026, 7, 30, 18, tzinfo=UTC),
    )

    assert first is not None and second is not None
    assert first.value == second.value == 1742.25
    assert provider.calls == 1
