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

    # Interactive (user-typed) analyses have no provider metadata by design;
    # the forecast itself is still produced for the two named teams.
    INTERACTIVE_CONTEXT_CHECKS = (
        "fixture_identified",
        "league_identified",
        "kickoff_known",
    )

    @classmethod
    def evaluate(
        cls,
        data_quality: Mapping[str, object],
        *,
        interactive: bool = False,
    ) -> PredictionEligibilityDecision:
        raw_checks = data_quality.get("checks")
        checks = raw_checks if isinstance(raw_checks, Mapping) else {}
        quality_score = cls._quality_score(data_quality, interactive=interactive)

        required_checks = (
            cls.INTERACTIVE_CONTEXT_CHECKS
            if interactive
            else cls.REQUIRED_CONTEXT_CHECKS
        )
        reasons: list[str] = []
        for check in required_checks:
            if checks.get(check) is not True:
                reasons.append(f"missing_{check}")

        if (
            not interactive
            and settings.AUTO_PREDICTION_REQUIRE_MARKET
            and checks.get("market_available") is not True
        ):
            reasons.append("market_unavailable")
        market_confirms_fixture = checks.get("market_available") is True
        if settings.AUTO_PREDICTION_REQUIRE_SUFFICIENT_HISTORY:
            # The fixture outcome is determined exclusively from local
            # historical fixtures; a live market confirms the fixture is real
            # but never substitutes for the local form/head-to-head evidence,
            # so insufficient local data must block the forecast.
            local_outcome_available = (
                checks.get("h2h_available") is True
                or (
                    checks.get("home_history_sufficient") is True
                    and checks.get("away_history_sufficient") is True
                )
            )
            if not local_outcome_available:
                if checks.get("h2h_available") is not True:
                    reasons.append("local_h2h_unavailable")
                if checks.get("home_history_sufficient") is not True:
                    reasons.append("home_history_insufficient")
                if checks.get("away_history_sufficient") is not True:
                    reasons.append("away_history_insufficient")
        min_score = settings.AUTO_PREDICTION_MIN_DATA_QUALITY_SCORE
        if (
            market_confirms_fixture
            and not interactive
            and checks.get("home_history_sufficient") is not True
            and checks.get("away_history_sufficient") is not True
        ):
            # Early in a season the history checks (26 pts of the coverage
            # score) are unachievable for market-confirmed fixtures; live odds
            # are direct evidence, so lower the input-coverage floor instead of
            # blocking the forecast on evidence that cannot exist yet.
            min_score = min(
                min_score,
                settings.AUTO_PREDICTION_MIN_DATA_QUALITY_SCORE_WITH_MARKET,
            )
        if quality_score < min_score:
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

    @classmethod
    def _quality_score(
        cls,
        data_quality: Mapping[str, object],
        *,
        interactive: bool,
    ) -> float:
        if interactive:
            interactive_score = data_quality.get("interactive_score")
            if isinstance(interactive_score, (int, float)) and not isinstance(
                interactive_score, bool
            ):
                return float(interactive_score)
        raw_quality_score = data_quality.get("score", 0.0)
        return (
            float(raw_quality_score)
            if isinstance(raw_quality_score, (int, float))
            and not isinstance(raw_quality_score, bool)
            else 0.0
        )
