from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.repository import MatchPredictionRepository
from app.db.session import SessionLocal
from app.prediction.audit import FinancialRecommendationPolicy, PredictionAuditor

logger = logging.getLogger("bet-ai-pro.financial-recommendations")


class FinancialRecommendationService:
    """Resolve the current financial-signal gate from verified production data."""

    def evaluate(self) -> dict[str, object]:
        if not settings.FINANCIAL_RECOMMENDATIONS_ENABLED:
            return {"eligible": False, "reasons": ["disabled_by_configuration"]}
        try:
            with SessionLocal() as db:
                audit = PredictionAuditor.audit_predictions(
                    MatchPredictionRepository(db).get_all_labeled()
                )
        except SQLAlchemyError:
            logger.exception("Financial recommendation audit could not be loaded")
            return {"eligible": False, "reasons": ["audit_unavailable"]}
        return FinancialRecommendationPolicy.evaluate(audit)


financial_recommendation_service = FinancialRecommendationService()
