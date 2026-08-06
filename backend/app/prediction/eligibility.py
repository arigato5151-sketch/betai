from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from app.core.config import settings


@dataclass(frozen=True)
class PredictionEligibilityDecision:
    eligible: bool
    status: str
    reasons: tuple[str, ...]
    data_quality_score: float

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


class PredictionIneligibleError(RuntimeError):
    def __init__(self, decision: PredictionEligibilityDecision) -> None:
        super().__init__("Prediction abstained because required data is incomplete")
        self.decision = decision


class PredictionEligibilityPolicy:
    """Fail closed when an automated forecast lacks decision-grade inputs."""

    REQUIRED_CONTEXT_CHECKS = (
        "fixture_identified",
        "fixture_source_identified",
        "provider_fixture_identified",
        "league_identified",
        "kickoff_known",
    )

    @classmethod
    def evaluate(
        cls,
        data_quality: Mapping[str, object],
    ) -> PredictionEligibilityDecision:
        raw_checks = data_quality.get("checks")
        checks = raw_checks if isinstance(raw_checks, Mapping) else {}
        raw_quality_score = data_quality.get("score", 0.0)
        quality_score = (
            float(raw_quality_score)
            if isinstance(raw_quality_score, (int, float))
            and not isinstance(raw_quality_score, bool)
            else 0.0
        )

        reasons: list[str] = []
        for check in cls.REQUIRED_CONTEXT_CHECKS:
            if checks.get(check) is not True:
                reasons.append(f"missing_{check}")

        if (
            settings.AUTO_PREDICTION_REQUIRE_MARKET
            and checks.get("market_available") is not True
        ):
            reasons.append("market_unavailable")
        if settings.AUTO_PREDICTION_REQUIRE_SUFFICIENT_HISTORY:
            if checks.get("home_history_sufficient") is not True:
                reasons.append("home_history_insufficient")
            if checks.get("away_history_sufficient") is not True:
                reasons.append("away_history_insufficient")
        if quality_score < settings.AUTO_PREDICTION_MIN_DATA_QUALITY_SCORE:
            reasons.append("data_quality_below_threshold")
        if data_quality.get("manual_feature_override_count", 0) != 0:
            reasons.append("manual_override_not_automatic")

        unique_reasons = tuple(dict.fromkeys(reasons))
        eligible = not unique_reasons
        return PredictionEligibilityDecision(
            eligible=eligible,
            status="eligible" if eligible else "abstain",
            reasons=unique_reasons,
            data_quality_score=quality_score,
        )
