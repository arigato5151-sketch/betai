from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.external_data_repository import ExternalDataRepository
from app.db.models import Base
from app.providers.base import ExternalDataPoint, ProviderTeamCandidate


def test_mapping_and_snapshot_upserts_are_idempotent() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    repository = ExternalDataRepository(session)
    observed_at = datetime(2026, 7, 30, 10, tzinfo=UTC)
    try:
        candidate = ProviderTeamCandidate("Fenerbahce", "Fenerbahce", 0.8)
        first_mapping = repository.upsert_mapping(
            canonical_source="api_football",
            canonical_team_id=611,
            canonical_team_name="Fenerbahçe",
            provider="clubelo",
            candidate=candidate,
        )
        second_mapping = repository.upsert_mapping(
            canonical_source="api_football",
            canonical_team_id=611,
            canonical_team_name="Fenerbahçe",
            provider="clubelo",
            candidate=candidate,
        )
        point = ExternalDataPoint(
            value=1742.25,
            source="clubelo",
            captured_at=observed_at,
            expires_at=observed_at + timedelta(hours=24),
            confidence=0.8,
        )
        first_snapshot = repository.save_snapshot(
            canonical_source="api_football",
            canonical_team_id=611,
            feature_name="elo",
            point=point,
        )
        second_snapshot = repository.save_snapshot(
            canonical_source="api_football",
            canonical_team_id=611,
            feature_name="elo",
            point=point,
        )
        session.expire_all()

        assert first_mapping.id == second_mapping.id
        assert first_snapshot.id == second_snapshot.id
        latest = repository.get_latest_snapshot(
            canonical_source="api_football",
            canonical_team_id=611,
            feature_name="elo",
            at=observed_at + timedelta(hours=1),
        )
        assert latest is not None
        assert repository.to_data_point(latest).value == 1742.25
    finally:
        session.close()


def test_expired_snapshot_is_not_returned() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    repository = ExternalDataRepository(session)
    observed_at = datetime(2026, 7, 30, 10, tzinfo=UTC)
    try:
        repository.save_snapshot(
            canonical_source="api_football",
            canonical_team_id=611,
            feature_name="elo",
            point=ExternalDataPoint(
                value=1742.25,
                source="clubelo",
                captured_at=observed_at,
                expires_at=observed_at + timedelta(hours=1),
                confidence=0.8,
            ),
        )

        assert (
            repository.get_latest_snapshot(
                canonical_source="api_football",
                canonical_team_id=611,
                feature_name="elo",
                at=observed_at + timedelta(hours=2),
            )
            is None
        )
    finally:
        session.close()
