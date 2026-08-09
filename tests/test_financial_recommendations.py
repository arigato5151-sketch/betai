from sqlalchemy.exc import OperationalError

from app.services import financial_recommendations as module


class _SessionContext:
    def __enter__(self):
        return object()

    def __exit__(self, *_args) -> None:
        return None


def test_financial_service_does_not_query_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(module.settings, "FINANCIAL_RECOMMENDATIONS_ENABLED", False)
    monkeypatch.setattr(
        module,
        "SessionLocal",
        lambda: (_ for _ in ()).throw(AssertionError("database must not be queried")),
    )

    decision = module.FinancialRecommendationService().evaluate()

    assert decision == {
        "eligible": False,
        "reasons": ["disabled_by_configuration"],
    }


def test_financial_service_fails_closed_when_audit_database_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(module.settings, "FINANCIAL_RECOMMENDATIONS_ENABLED", True)
    monkeypatch.setattr(
        module,
        "SessionLocal",
        lambda: (_ for _ in ()).throw(
            OperationalError("connect", {}, RuntimeError("offline"))
        ),
    )

    decision = module.FinancialRecommendationService().evaluate()

    assert decision == {"eligible": False, "reasons": ["audit_unavailable"]}


def test_financial_service_delegates_verified_rows_to_policy(monkeypatch) -> None:
    monkeypatch.setattr(module.settings, "FINANCIAL_RECOMMENDATIONS_ENABLED", True)
    monkeypatch.setattr(module, "SessionLocal", _SessionContext)
    monkeypatch.setattr(
        module,
        "MatchPredictionRepository",
        lambda _db: type("Repo", (), {"get_all_labeled": lambda self: ["row"]})(),
    )
    monkeypatch.setattr(
        module.PredictionAuditor,
        "audit_predictions",
        lambda rows: {"rows": rows},
    )
    monkeypatch.setattr(
        module.FinancialRecommendationPolicy,
        "evaluate",
        lambda audit: {"eligible": True, "audit": audit},
    )

    decision = module.FinancialRecommendationService().evaluate()

    assert decision == {"eligible": True, "audit": {"rows": ["row"]}}
