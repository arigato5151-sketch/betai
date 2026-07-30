from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base, FixtureOddsSnapshot
from app.services.odds_history import OddsHistoryService


def _service() -> tuple[OddsHistoryService, sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    return OddsHistoryService(session_factory=factory), factory


def _prefill(kickoff: datetime) -> dict[str, object]:
    return {
        "fixture": {
            "fixture_id": 1556549,
            "kickoff": kickoff.isoformat(),
        },
        "market_1x2": {
            "raw_odds": {
                "HOME_WIN": 2.25,
                "DRAW": 3.30,
                "AWAY_WIN": 3.10,
            },
            "bookmaker": "Test Book",
            "method": "proportional_devig",
            "overround_pct": 4.5,
        },
    }


def test_odds_history_requires_two_time_separated_observations() -> None:
    service, factory = _service()
    kickoff = datetime(2030, 7, 30, 18, tzinfo=UTC)
    opening_at = kickoff - timedelta(hours=8)

    first = service.enrich_prefill(
        _prefill(kickoff),
        captured_at=opening_at,
    )
    second = service.enrich_prefill(
        _prefill(kickoff),
        captured_at=opening_at
        + timedelta(seconds=settings.ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS),
    )

    assert first["odds_history"]["status"] == "collecting"
    assert "opening_odds_1x2" not in first
    assert second["odds_history"]["status"] == "ready"
    assert second["opening_odds_1x2"]["HOME_WIN"] == 2.25
    assert second["current_odds_1x2"]["AWAY_WIN"] == 3.10
    assert second["opening_odds_at"].endswith("+00:00")
    with factory() as db:
        assert db.query(FixtureOddsSnapshot).count() == 2


def test_odds_history_deduplicates_unchanged_rapid_observations() -> None:
    service, factory = _service()
    kickoff = datetime(2030, 7, 30, 18, tzinfo=UTC)
    captured_at = kickoff - timedelta(hours=8)
    prefill = _prefill(kickoff)

    service.enrich_prefill(prefill, captured_at=captured_at)
    service.enrich_prefill(
        prefill,
        captured_at=captured_at + timedelta(seconds=30),
    )

    with factory() as db:
        assert db.query(FixtureOddsSnapshot).count() == 1


def test_odds_history_does_not_record_after_kickoff() -> None:
    service, factory = _service()
    kickoff = datetime(2030, 7, 30, 18, tzinfo=UTC)

    result = service.enrich_prefill(
        _prefill(kickoff),
        captured_at=kickoff,
    )

    assert "odds_history" not in result
    with factory() as db:
        assert db.query(FixtureOddsSnapshot).count() == 0


def test_background_collection_takes_opening_then_waits_for_closing_window() -> None:
    service, _ = _service()
    kickoff = datetime(2030, 7, 30, 18, tzinfo=UTC)
    opening_at = kickoff - timedelta(days=5)

    assert service.should_collect(
        fixture_id=1556549,
        kickoff=kickoff,
        observed_at=opening_at,
        refresh_interval_seconds=10800,
        closing_window_hours=24,
    )
    service.enrich_prefill(_prefill(kickoff), captured_at=opening_at)

    assert not service.should_collect(
        fixture_id=1556549,
        kickoff=kickoff,
        observed_at=opening_at + timedelta(hours=3),
        refresh_interval_seconds=10800,
        closing_window_hours=24,
    )
    assert service.should_collect(
        fixture_id=1556549,
        kickoff=kickoff,
        observed_at=kickoff - timedelta(hours=23),
        refresh_interval_seconds=10800,
        closing_window_hours=24,
    )


def test_background_collection_rejects_rapid_or_post_kickoff_polling() -> None:
    service, _ = _service()
    kickoff = datetime(2030, 7, 30, 18, tzinfo=UTC)
    first_at = kickoff - timedelta(hours=12)
    service.enrich_prefill(_prefill(kickoff), captured_at=first_at)

    assert not service.should_collect(
        fixture_id=1556549,
        kickoff=kickoff,
        observed_at=first_at + timedelta(minutes=30),
        refresh_interval_seconds=10800,
        closing_window_hours=24,
    )
    assert not service.should_collect(
        fixture_id=1556549,
        kickoff=kickoff,
        observed_at=kickoff,
        refresh_interval_seconds=10800,
        closing_window_hours=24,
    )
