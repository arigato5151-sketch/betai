from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from unittest.mock import Mock

from app.db.models import Base, MatchPrediction
from app.db.repository import MatchPredictionRepository


def build_repository() -> tuple[Session, MatchPredictionRepository]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    now = datetime.now(UTC).replace(tzinfo=None)
    session.add_all(
        [
            MatchPrediction(
                home_team="Fenerbahce",
                away_team="Galatasaray",
                actual_result=None,
                is_value_bet=1,
                edge=8.2,
                odd=2.1,
                created_at=now,
            ),
            MatchPrediction(
                home_team="Besiktas",
                away_team="Trabzonspor",
                actual_result="HOME_WIN",
                is_value_bet=0,
                edge=1.2,
                odd=3.4,
                created_at=now + timedelta(minutes=1),
            ),
            MatchPrediction(
                home_team="Arsenal_100%",
                away_team="Chelsea",
                actual_result="DRAW",
                is_value_bet=None,
                edge=3.5,
                odd=2.9,
                created_at=now + timedelta(minutes=2),
            ),
        ]
    )
    session.commit()
    return session, MatchPredictionRepository(session)


def test_history_search_combines_filters_and_reports_total() -> None:
    session, repository = build_repository()
    try:
        items, total = repository.search_history(
            page=1,
            page_size=10,
            query="fener",
            result="pending",
            value="value",
        )

        assert total == 1
        assert [item.home_team for item in items] == ["Fenerbahce"]
    finally:
        session.close()


def test_history_search_paginates_and_sorts_deterministically() -> None:
    session, repository = build_repository()
    try:
        items, total = repository.search_history(page=2, page_size=1, sort="odd")

        assert total == 3
        assert [item.odd for item in items] == [2.9]
    finally:
        session.close()


def test_history_search_escapes_wildcards() -> None:
    session, repository = build_repository()
    try:
        items, total = repository.search_history(page=1, page_size=10, query="_100%")

        assert total == 1
        assert [item.home_team for item in items] == ["Arsenal_100%"]
    finally:
        session.close()


def test_sqlite_upsert_creates_then_updates_same_fixture() -> None:
    session, repository = build_repository()
    try:
        created = repository.upsert_prediction(
            {"fixture_id": 99, "home_team": "Alpha", "away_team": "Beta"}
        )
        updated = repository.upsert_prediction(
            {"fixture_id": 99, "home_team": "Updated", "away_team": "Beta"}
        )

        assert created.id == updated.id
        assert updated.home_team == "Updated"
        assert repository.get_by_fixture_id(99) is updated
    finally:
        session.close()


def test_sqlite_upsert_persists_versioned_feature_snapshot() -> None:
    session, repository = build_repository()
    snapshot = {"home_form": 72.0, "away_form": 61.0}
    try:
        repository.upsert_prediction(
            {
                "fixture_id": 100,
                "home_team": "Snapshot FC",
                "away_team": "Parity FC",
                "feature_snapshot": snapshot,
                "feature_schema_version": "ml_features_v1",
                "feature_snapshot_at": datetime(2026, 7, 19, tzinfo=UTC),
            }
        )

        session.expire_all()
        persisted = repository.get_by_fixture_id(100)

        assert persisted is not None
        assert persisted.feature_snapshot == snapshot
        assert persisted.feature_schema_version == "ml_features_v1"
        assert persisted.feature_snapshot_at is not None
    finally:
        session.close()


def test_repository_lists_labeled_unlabeled_and_counts() -> None:
    session, repository = build_repository()
    try:
        assert repository.count_labeled() == 2
        assert len(repository.get_all_labeled()) == 2
        assert [item.home_team for item in repository.get_unlabeled(limit=1)] == [
            "Fenerbahce"
        ]
        assert len(repository.get_all()) == 3
        assert repository.get_recent(limit=2)[0].home_team == "Arsenal_100%"
    finally:
        session.close()


def test_update_result_updates_optional_metrics_and_missing_record() -> None:
    session, repository = build_repository()
    try:
        record = repository.get_all()[0]
        updated = repository.update_result(
            record.id,
            "AWAY_WIN",
            actual_score_home=0,
            actual_score_away=2,
            roi=-1.0,
            clv=0.03,
            closing_odds=2.05,
        )

        assert updated is not None
        assert updated.actual_result == "AWAY_WIN"
        assert updated.actual_score_home == 0
        assert updated.actual_score_away == 2
        assert updated.roi == -1.0
        assert updated.clv == 0.03
        assert updated.closing_odds == 2.05
        assert repository.update_result(9999, "DRAW") is None
    finally:
        session.close()


def test_failed_commit_rolls_back_transaction() -> None:
    session = Mock()
    session.commit.side_effect = SQLAlchemyError("write failed")
    repository = MatchPredictionRepository(session)

    try:
        repository._commit()
    except SQLAlchemyError:
        pass
    else:
        raise AssertionError("SQLAlchemyError should be propagated")

    session.rollback.assert_called_once_with()
