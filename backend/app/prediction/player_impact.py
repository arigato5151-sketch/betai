from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from app.core.config import settings


@dataclass(frozen=True, slots=True)
class TeamStrengthImpact:
    """Immutable audit trail for a pre-match squad-strength assessment."""

    strength_ratio: float
    xg_multiplier: float
    data_available: bool
    reference_average_impact: float
    reference_total_impact: float
    adjusted_total_impact: float
    rated_reference_starters: int
    rated_current_starters: int
    missing_player_ids: tuple[int, ...]
    questionable_player_ids: tuple[int, ...]
    critical_missing_player_ids: tuple[int, ...]

    @property
    def team_strength_ratio(self) -> float:
        """Expose the domain name used by the ML feature schema."""
        return self.strength_ratio

    @property
    def critical_missing_count(self) -> int:
        return len(self.critical_missing_player_ids)


class PlayerImpactCalculator:
    """Calculate lineup strength without treating every absence as equally costly."""

    _MAX_STARTERS = 11

    @classmethod
    def derive_reference_lineup(
        cls,
        player_ratings: Mapping[Any, Any] | None,
    ) -> list[int] | None:
        """Infer a typical XI from exposure, then impact, with stable tie-breaking."""
        if not isinstance(player_ratings, Mapping):
            return None

        ranked: list[tuple[float, float, float, int]] = []
        for raw_player_id, raw_value in player_ratings.items():
            player_id = cls._coerce_player_id(raw_player_id)
            impact = cls._coerce_impact(raw_value)
            if player_id is None or impact is None:
                continue
            if isinstance(raw_value, Mapping):
                minutes = cls._non_negative_finite(raw_value.get("minutes")) or 0.0
                appearances = (
                    cls._non_negative_finite(raw_value.get("appearances")) or 0.0
                )
            else:
                minutes = appearances = 0.0
            ranked.append((minutes, appearances, impact, player_id))

        if len(ranked) < cls._MAX_STARTERS:
            return None
        ranked.sort(reverse=True)
        return [player_id for *_signals, player_id in ranked[: cls._MAX_STARTERS]]

    @classmethod
    def assess(
        cls,
        player_ratings: Mapping[Any, Any] | None,
        reference_lineup: Iterable[object] | object | None,
        current_lineup: Iterable[object] | object | None = None,
        missing_player_ids: Iterable[object] | object | None = None,
        questionable_player_ids: Iterable[object] | object | None = None,
    ) -> TeamStrengthImpact:
        impacts = cls._normalize_impacts(player_ratings)
        reference_ids = cls._normalize_player_ids(
            reference_lineup,
            limit=cls._MAX_STARTERS,
        )
        rated_reference = {
            player_id: impacts[player_id]
            for player_id in reference_ids
            if player_id in impacts
        }
        minimum_coverage = int(settings.PLAYER_IMPACT_MIN_RATED_STARTERS)

        if (
            len(reference_ids) != cls._MAX_STARTERS
            or len(rated_reference) < minimum_coverage
        ):
            return cls._neutral(rated_reference_starters=len(rated_reference))

        rated_reference_total = sum(rated_reference.values())
        if not math.isfinite(rated_reference_total) or rated_reference_total <= 0.0:
            return cls._neutral(rated_reference_starters=len(rated_reference))

        reference_average = rated_reference_total / len(rated_reference)
        # The minimum-coverage setting intentionally permits partial feeds. Impute
        # uncovered starters with the known-XI average so both XI totals stay on
        # the same 11-player scale instead of rewarding whichever side has more data.
        reference = {
            player_id: rated_reference.get(player_id, reference_average)
            for player_id in reference_ids
        }
        reference_total = sum(reference.values())
        current_ids = cls._normalize_player_ids(
            current_lineup,
            limit=cls._MAX_STARTERS,
        )
        has_confirmed_lineup = len(current_ids) == cls._MAX_STARTERS
        missing = cls._relevant_absences(missing_player_ids, reference)
        questionable = tuple(
            player_id
            for player_id in cls._relevant_absences(
                questionable_player_ids,
                reference,
            )
            if player_id not in missing
        )
        if has_confirmed_lineup:
            # A confirmed starter supersedes a stale injury/availability row.
            current_id_set = set(current_ids)
            missing = tuple(
                player_id for player_id in missing if player_id not in current_id_set
            )
            questionable = tuple(
                player_id
                for player_id in questionable
                if player_id not in current_id_set
            )
        critical = tuple(
            player_id
            for player_id in missing
            if reference[player_id] > reference_average
        )

        rated_current = {
            player_id: impacts[player_id]
            for player_id in current_ids
            if player_id in impacts and player_id not in missing
        }
        current = {
            player_id: impacts.get(
                player_id,
                reference.get(player_id, reference_average),
            )
            for player_id in current_ids
            if player_id not in missing
        }
        confirmed_current = (
            has_confirmed_lineup and len(rated_current) >= minimum_coverage
        )

        if confirmed_current:
            adjusted_total = cls._confirmed_lineup_total(
                reference=reference,
                current=current,
                critical_missing=critical,
                questionable=questionable,
            )
        else:
            adjusted_total = cls._projected_lineup_total(
                reference=reference,
                missing=missing,
                questionable=questionable,
                critical_missing=critical,
            )

        min_ratio = float(settings.PLAYER_IMPACT_MIN_STRENGTH_RATIO)
        max_ratio = float(settings.PLAYER_IMPACT_MAX_STRENGTH_RATIO)
        strength_ratio = cls._clamp(
            adjusted_total / reference_total,
            minimum=min_ratio,
            maximum=max_ratio,
        )
        xg_multiplier = cls._clamp(
            1.0 + (strength_ratio - 1.0) * float(settings.PLAYER_IMPACT_XG_ELASTICITY),
            minimum=float(settings.PLAYER_IMPACT_MIN_XG_MULTIPLIER),
            maximum=max_ratio,
        )

        return TeamStrengthImpact(
            strength_ratio=round(strength_ratio, 6),
            xg_multiplier=round(xg_multiplier, 6),
            data_available=True,
            reference_average_impact=round(reference_average, 6),
            reference_total_impact=round(reference_total, 6),
            adjusted_total_impact=round(max(0.0, adjusted_total), 6),
            rated_reference_starters=len(rated_reference),
            rated_current_starters=len(rated_current),
            missing_player_ids=missing,
            questionable_player_ids=questionable,
            critical_missing_player_ids=critical,
        )

    @classmethod
    def _confirmed_lineup_total(
        cls,
        *,
        reference: Mapping[int, float],
        current: Mapping[int, float],
        critical_missing: tuple[int, ...],
        questionable: tuple[int, ...],
    ) -> float:
        reference_ids = set(reference)
        retained_total = sum(
            reference[player_id] for player_id in current if player_id in reference_ids
        )
        replacement_total = sum(
            impact
            for player_id, impact in current.items()
            if player_id not in reference_ids
        )
        replacement_factor = float(settings.PLAYER_IMPACT_REPLACEMENT_FACTOR)
        adjusted = retained_total + replacement_total * replacement_factor

        # Critical/questionable modifiers model uncertainty beyond the observed XI swap.
        net_replacement_loss = 1.0 - replacement_factor
        adjusted -= (
            sum(reference[player_id] for player_id in critical_missing)
            * float(settings.PLAYER_CRITICAL_ABSENCE_WEIGHT)
            * net_replacement_loss
        )
        adjusted -= (
            sum(reference[player_id] for player_id in questionable)
            * float(settings.PLAYER_QUESTIONABLE_ABSENCE_WEIGHT)
            * net_replacement_loss
        )
        return adjusted

    @classmethod
    def _projected_lineup_total(
        cls,
        *,
        reference: Mapping[int, float],
        missing: tuple[int, ...],
        questionable: tuple[int, ...],
        critical_missing: tuple[int, ...],
    ) -> float:
        replacement_loss = 1.0 - float(settings.PLAYER_IMPACT_REPLACEMENT_FACTOR)
        critical_ids = set(critical_missing)
        adjusted = sum(reference.values())

        for player_id in missing:
            critical_weight = (
                float(settings.PLAYER_CRITICAL_ABSENCE_WEIGHT)
                if player_id in critical_ids
                else 0.0
            )
            adjusted -= (
                reference[player_id] * replacement_loss * (1.0 + critical_weight)
            )

        questionable_weight = float(settings.PLAYER_QUESTIONABLE_ABSENCE_WEIGHT)
        adjusted -= sum(
            reference[player_id] * replacement_loss * questionable_weight
            for player_id in questionable
        )
        return adjusted

    @classmethod
    def _normalize_impacts(
        cls,
        player_ratings: Mapping[Any, Any] | None,
    ) -> dict[int, float]:
        if not isinstance(player_ratings, Mapping):
            return {}

        normalized: dict[int, float] = {}
        for raw_player_id, raw_impact in player_ratings.items():
            player_id = cls._coerce_player_id(raw_player_id)
            impact = cls._coerce_impact(raw_impact)
            if player_id is None or impact is None:
                continue
            # JSON can contain both "123" and 123 after upstream merging.
            normalized[player_id] = max(impact, normalized.get(player_id, 0.0))
        return normalized

    @classmethod
    def _coerce_impact(cls, raw_impact: object) -> float | None:
        if isinstance(raw_impact, Mapping):
            rating = cls._positive_finite(raw_impact.get("rating"))
            if rating is not None:
                return rating

            # Contribution-only feeds are normalized per appearance/90 minutes.
            minutes = cls._non_negative_finite(raw_impact.get("minutes")) or 0.0
            appearances = cls._non_negative_finite(raw_impact.get("appearances")) or 0.0
            goals = cls._non_negative_finite(raw_impact.get("goals")) or 0.0
            assists = cls._non_negative_finite(raw_impact.get("assists")) or 0.0
            exposure = max(appearances, minutes / 90.0)
            if exposure <= 0.0:
                return None
            contribution = (goals + assists * 0.7) / exposure
            return contribution if contribution > 0.0 else None

        return cls._positive_finite(raw_impact)

    @classmethod
    def _relevant_absences(
        cls,
        values: Iterable[object] | object | None,
        reference: Mapping[int, float],
    ) -> tuple[int, ...]:
        return tuple(
            player_id
            for player_id in cls._normalize_player_ids(values)
            if player_id in reference
        )

    @classmethod
    def _normalize_player_ids(
        cls,
        values: Iterable[object] | object | None,
        *,
        limit: int | None = None,
    ) -> tuple[int, ...]:
        if values is None:
            return ()
        if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
            candidates: Iterable[object] = (values,)
        else:
            candidates = values

        normalized: list[int] = []
        seen: set[int] = set()
        for value in candidates:
            player_id = cls._coerce_player_id(value)
            if player_id is None or player_id in seen:
                continue
            normalized.append(player_id)
            seen.add(player_id)
            if limit is not None and len(normalized) >= limit:
                break
        return tuple(normalized)

    @staticmethod
    def _coerce_player_id(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)  # Supports JSON object keys such as "123".
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
            return None
        return int(numeric)

    @staticmethod
    def _positive_finite(value: Any) -> float | None:
        number = PlayerImpactCalculator._non_negative_finite(value)
        return number if number is not None and number > 0.0 else None

    @staticmethod
    def _non_negative_finite(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number < 0.0:
            return None
        return number

    @staticmethod
    def _clamp(value: float, *, minimum: float, maximum: float) -> float:
        if not math.isfinite(value):
            return 1.0
        return max(minimum, min(maximum, value))

    @staticmethod
    def _neutral(*, rated_reference_starters: int = 0) -> TeamStrengthImpact:
        return TeamStrengthImpact(
            strength_ratio=1.0,
            xg_multiplier=1.0,
            data_available=False,
            reference_average_impact=0.0,
            reference_total_impact=0.0,
            adjusted_total_impact=0.0,
            rated_reference_starters=rated_reference_starters,
            rated_current_starters=0,
            missing_player_ids=(),
            questionable_player_ids=(),
            critical_missing_player_ids=(),
        )
