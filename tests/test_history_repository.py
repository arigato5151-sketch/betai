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
                training_eligible=True,
                result_verification_status="pending",
                created_at=now,
            ),
            MatchPrediction(
                home_team="Besiktas",
                away_team="Trabzonspor",
                actual_result="HOME_WIN",
                is_value_bet=0,
                edge=1.2,
                odd=3.4,
                training_eligible=True,
                result_verification_status="verified",
                created_at=now + timedelta(minutes=1),
            ),
            MatchPrediction(
                home_team="Arsenal_100%",
                away_team="Chelsea",
                actual_result="DRAW",
                is_value_bet=None,
                edge=3.5,
                odd=2.9,
                training_eligible=True,
                result_verification_status="verified",
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


def test_pg_upsert_update_columns_only_mutates_incoming_fields() -> None:
    session, _ = build_repository()
    try:
        cols = MatchPredictionRepository._upsert_update_columns(
            {"fixture_id": 1, "home_team": "Alpha", "edge": 5.0}
        )
        assert set(cols) == {"home_team", "edge"}
        for protected in ("id", "created_at", "fixture_id", "analysis_origin"):
            assert protected not in cols
    finally:
        session.close()


def test_pg_upsert_update_columns_never_wipes_verified_labels() -> None:
    session, _ = build_repository()
    try:
        cols = MatchPredictionRepository._upsert_update_columns(
            {"fixture_id": 1, "home_team": "Updated"}
        )
        for preserved in (
            "roi",
            "closing_odds",
            "actual_result",
            "result_verification_status",
            "training_eligible",
            "result_verified_at",
        ):
            assert preserved not in cols, (
                f"{preserved} must not be reset by a partial re-analysis"
            )
    finally:
        session.close()


def test_sqlite_upsert_partial_update_preserves_existing_fields() -> None:
    session, repository = build_repository()
    try:
        repository.upsert_prediction(
            {
                "fixture_id": 101,
                "home_team": "Label FC",
                "away_team": "Value FC",
                "actual_result": "HOME_WIN",
                "result_verification_status": "verified",
                "training_eligible": True,
                "roi": 2.4,
                "closing_odds": 1.95,
            }
        )
        updated = repository.upsert_prediction(
            {"fixture_id": 101, "home_team": "Label FC", "away_team": "Renamed FC"}
        )

        assert updated.actual_result == "HOME_WIN"
        assert updated.result_verification_status == "verified"
        assert updated.training_eligible is True
        assert updated.roi == 2.4
        assert updated.closing_odds == 1.95
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


def test_sqlite_upsert_merges_equivalent_cross_provider_fixture() -> None:
    session, repository = build_repository()
    kickoff = datetime(2026, 8, 22, 11, 30, tzinfo=UTC)
    try:
        first = repository.upsert_prediction(
            {
                "fixture_id": 1_802_272_674,
                "fixture_source": "fixture_download",
                "provider_fixture_id": "1802272674",
                "league_id": 40,
                "home_team": "Millwall",
                "away_team": "Norwich City",
                "kickoff": kickoff,
                "probability": 63.49,
            }
        )
        updated = repository.upsert_prediction(
            {
                "fixture_id": 1_563_099,
                "fixture_source": "api_football",
                "provider_fixture_id": "1563099",
                "league_id": 40,
                "home_team": "Millwall",
                "away_team": "Norwich",
                "kickoff": kickoff,
                "probability": 63.02,
            }
        )

        assert updated.id == first.id
        assert updated.fixture_id == 1_802_272_674
        assert updated.fixture_source == "fixture_download"
        assert updated.away_team == "Norwich City"
        assert updated.probability == 63.02
        assert session.query(MatchPrediction).filter_by(league_id=40).count() == 1
    finally:
        session.close()


def test_upsert_never_rewrites_resolved_prediction() -> None:
    session, repository = build_repository()
    kickoff = datetime(2026, 8, 14, 18, 0, tzinfo=UTC)
    try:
        original = repository.upsert_prediction(
            {
                "fixture_id": 700,
                "league_id": 88,
                "home_team": "Telstar",
                "away_team": "Sparta Rotterdam",
                "kickoff": kickoff,
                "prediction": "HOME_WIN",
                "actual_result": "AWAY_WIN",
                "result_verification_status": "verified",
            }
        )
        unchanged = repository.upsert_prediction(
            {
                "fixture_id": 700,
                "league_id": 88,
                "home_team": "Telstar",
                "away_team": "Sparta Rotterdam",
                "kickoff": kickoff,
                "prediction": "AWAY_WIN",
            }
        )

        assert unchanged.id == original.id
        assert unchanged.prediction == "HOME_WIN"
        assert unchanged.actual_result == "AWAY_WIN"
    finally:
        session.close()


def test_sqlite_upsert_persists_versioned_feature_snapshot() -> None:
    session, repository = build_repository()
    snapshot = {"home_form": 72.0, "away_form": 61.0}
    probability_components = {
        "version": "weighted_probability_ensemble_v1",
        "weights": {"stats": 0.5, "market": 0.5},
    }
    try:
        repository.upsert_prediction(
            {
                "fixture_id": 100,
                "home_team": "Snapshot FC",
                "away_team": "Parity FC",
                "feature_snapshot": snapshot,
                "feature_schema_version": "ml_features_v1",
                "feature_snapshot_at": datetime(2026, 7, 19, tzinfo=UTC),
                "probability_components": probability_components,
                "ensemble_version": "weighted_probability_ensemble_v1",
            }
        )

        session.expire_all()
        persisted = repository.get_by_fixture_id(100)

        assert persisted is not None
        assert persisted.feature_snapshot == snapshot
        assert persisted.feature_schema_version == "ml_features_v1"
        assert persisted.feature_snapshot_at is not None
        assert persisted.probability_components == probability_components
        assert persisted.ensemble_version == "weighted_probability_ensemble_v1"
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


def test_auditable_predictions_can_be_scoped_to_model_artifact() -> None:
    session, repository = build_repository()
    try:
        rows = session.query(MatchPrediction).order_by(MatchPrediction.id).all()
        rows[0].model_artifact_version = "model-v1"
        rows[1].model_artifact_version = "model-v2"
        rows[2].model_artifact_version = "model-v2"
        session.commit()

        selected = repository.get_all_auditable(
            model_artifact_version="model-v2"
        )

        assert [row.model_artifact_version for row in selected] == [
            "model-v2",
            "model-v2",
        ]
    finally:
        session.close()


def test_repository_excludes_scenarios_from_training_and_audit() -> None:
    session, repository = build_repository()
    try:
        session.add(
            MatchPrediction(
                home_team="Scenario Home",
                away_team="Scenario Away",
                actual_result="HOME_WIN",
                analysis_origin="scenario",
                eligibility_status="abstain",
                training_eligible=False,
            )
        )
        session.commit()

        assert repository.count_labeled() == 2
        assert len(repository.get_all_labeled()) == 2
        assert len(repository.get_all_auditable()) == 3
        assert len(repository.get_all()) == 4
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
            verification_status="verified",
            result_source="api_football",
            result_provider_fixture_id="123",
        )

        assert updated is not None
        assert updated.actual_result == "AWAY_WIN"
        assert updated.actual_score_home == 0
        assert updated.actual_score_away == 2
        assert updated.roi == -1.0
        assert updated.clv == 0.03
        assert updated.closing_odds == 2.05
        assert updated.result_verification_status == "verified"
        assert updated.result_source == "api_football"
        assert updated.result_verified_at is not None
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


def test_conflicting_verified_result_is_quarantined_without_overwrite() -> None:
    session, repository = build_repository()
    try:
        record = repository.get_all()[1]
        updated = repository.update_result(
            record.id,
            "AWAY_WIN",
            actual_score_home=0,
            actual_score_away=1,
            verification_status="verified",
            result_source="api_football",
            result_provider_fixture_id="123",
        )

        assert updated is not None
        assert updated.actual_result == "HOME_WIN"
        assert updated.result_verification_status == "conflict"
        assert updated.training_eligible is False
        assert repository.count_labeled() == 1
    finally:
        session.close()


def test_prediction_created_at_default_is_timezone_aware() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    # SQLite drops tzinfo on round-trip, so expire_on_commit=False keeps the
    # in-memory default that the model guarantees (Postgres timestamptz holds it).
    session = Session(engine, expire_on_commit=False)
    try:
        prediction = MatchPrediction(
            home_team="Fenerbahce",
            away_team="Galatasaray",
            actual_result=None,
            is_value_bet=1,
            edge=8.2,
            odd=2.1,
            training_eligible=True,
            result_verification_status="pending",
        )
        session.add(prediction)
        session.commit()

        assert prediction.created_at is not None
        assert prediction.created_at.tzinfo is not None
        assert prediction.created_at.utcoffset() == timedelta(0)
    finally:
        session.close()
