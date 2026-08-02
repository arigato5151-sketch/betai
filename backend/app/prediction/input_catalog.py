from __future__ import annotations

import math
from typing import Any, Mapping

from app.core.config import settings
from app.prediction.ml.explain import ExplainabilityService
from app.prediction.ml.features import FeatureEngine


class AnalysisInputCatalog:
    """Describe editable ML inputs and their point-in-time availability."""

    DIRECT_INPUTS = {
        "home_form",
        "home_attack",
        "home_defense",
        "home_xg",
        "away_form",
        "away_attack",
        "away_defense",
        "away_xg",
    }
    HOME_HISTORY_INPUTS = {
        "home_form_ema",
        "home_clean_sheet_streak",
        "home_scoring_streak",
        "home_advantage_coeff",
        "home_gf_last5",
        "home_ga_last5",
    }
    AWAY_HISTORY_INPUTS = {
        "away_form_ema",
        "away_clean_sheet_streak",
        "away_scoring_streak",
        "away_gf_last5",
        "away_ga_last5",
    }
    H2H_INPUTS = {
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_home_loss_rate",
        "h2h_avg_goals_home",
        "h2h_avg_goals_away",
    }
    AVAILABILITY_INPUTS = {
        "home_missing_players",
        "away_missing_players",
        "home_questionable_players",
        "away_questionable_players",
        "availability_report_present",
    }
    LINEUP_INPUTS = {
        "home_lineup_confirmed",
        "away_lineup_confirmed",
        "home_lineup_reference_available",
        "away_lineup_reference_available",
        "home_lineup_continuity",
        "away_lineup_continuity",
    }
    PLAYER_IMPACT_INPUTS = {
        "home_team_strength_ratio",
        "away_team_strength_ratio",
    }
    ODDS_MOVEMENT_INPUTS = {
        "odds_movement_home",
        "odds_movement_draw",
        "odds_movement_away",
    }
    WEATHER_INPUTS = {
        "weather_temperature_c",
        "weather_precipitation_mm",
        "weather_wind_speed_kmh",
        "weather_available",
    }
    IDENTITY_INPUTS = {"league_id", "home_team_id", "away_team_id"}

    @classmethod
    def _bounds(cls, name: str) -> tuple[float, float, float]:
        if name in {
            "home_form",
            "away_form",
            "home_attack",
            "away_attack",
            "home_defense",
            "away_defense",
            "home_form_ema",
            "away_form_ema",
        }:
            return (0.0, 100.0, 1.0)
        if name in {"home_xg", "away_xg"}:
            return (0.0, 5.0, 0.01)
        if name in cls.PLAYER_IMPACT_INPUTS:
            return (
                float(settings.PLAYER_IMPACT_MIN_STRENGTH_RATIO),
                float(settings.PLAYER_IMPACT_MAX_STRENGTH_RATIO),
                0.01,
            )
        if name == "fatigue_index":
            return (-1.0, 1.0, 0.01)
        if name == "rest_days_diff":
            return (-30.0, 30.0, 1.0)
        if name in cls.H2H_INPUTS and not name.startswith("h2h_avg_goals"):
            return (0.0, 1.0, 0.01)
        if name.endswith("_continuity"):
            return (0.0, 1.0, 0.01)
        if (
            name.endswith("_confirmed")
            or name.endswith("_available")
            or name == "availability_report_present"
            or name.startswith("league_")
        ):
            return (0.0, 1.0, 1.0)
        if name in cls.IDENTITY_INPUTS:
            return (0.0, 1_000_000_000.0, 1.0)
        if name in cls.AVAILABILITY_INPUTS:
            return (0.0, 30.0, 1.0)
        if name.endswith("_streak"):
            return (-20.0, 20.0, 1.0)
        if name in {"home_elo", "away_elo"}:
            return (500.0, 3500.0, 1.0)
        if name == "home_advantage_coeff":
            return (0.5, 1.5, 0.01)
        if name.endswith("_last5") or name.startswith("h2h_avg_goals"):
            return (0.0, 10.0, 0.01)
        if name in cls.ODDS_MOVEMENT_INPUTS:
            return (-100.0, 1000.0, 0.1)
        if name == "weather_temperature_c":
            return (-80.0, 65.0, 0.1)
        if name == "weather_precipitation_mm":
            return (0.0, 500.0, 0.1)
        if name == "weather_wind_speed_kmh":
            return (0.0, 300.0, 0.1)
        if name == "weather_available":
            return (0.0, 1.0, 1.0)
        return (-1_000_000.0, 1_000_000.0, 0.01)

    @classmethod
    def validate_overrides(cls, raw: object) -> dict[str, float]:
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError("feature_overrides must be an object")

        allowed = set(FeatureEngine.FEATURE_NAMES)
        result: dict[str, float] = {}
        for name, value in raw.items():
            if name not in allowed:
                raise ValueError(f"Unsupported feature override: {name}")
            if isinstance(value, bool):
                raise ValueError(f"Feature override must be numeric: {name}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature override must be numeric: {name}") from exc
            minimum, maximum, _ = cls._bounds(name)
            if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
                raise ValueError(
                    f"Feature override {name} must be between {minimum} and {maximum}"
                )
            result[name] = numeric
        return result

    @classmethod
    def _group(cls, name: str) -> str:
        if name in cls.DIRECT_INPUTS:
            return "Temel takım değerleri"
        if name in cls.HOME_HISTORY_INPUTS | cls.AWAY_HISTORY_INPUTS:
            return "Form ve son maçlar"
        if name in cls.H2H_INPUTS:
            return "İkili rekabet"
        if name in {"home_elo", "away_elo"}:
            return "Elo derecelendirmesi"
        if (
            name
            in cls.PLAYER_IMPACT_INPUTS | cls.AVAILABILITY_INPUTS | cls.LINEUP_INPUTS
        ):
            return "Kadro ve oyuncu durumu"
        if name in {"rest_days_diff", "fatigue_index"}:
            return "Dinlenme ve seyahat"
        if name in cls.ODDS_MOVEMENT_INPUTS:
            return "Oran hareketleri"
        if name in cls.WEATHER_INPUTS:
            return "Hava koşulları"
        return "Lig ve takım kimlikleri"

    @classmethod
    def _availability(
        cls,
        name: str,
        checks: Mapping[str, Any],
        calculated: Mapping[str, float],
    ) -> tuple[str, str | None]:
        if name in cls.DIRECT_INPUTS:
            return ("available", None)
        if name in cls.HOME_HISTORY_INPUTS:
            if checks.get("home_history_sufficient"):
                return ("available", None)
            if checks.get("home_history_available"):
                return ("partial", "Ev sahibi geçmişi beş maçtan az.")
            return ("missing", "Ev sahibi için yakın maç verisi yok.")
        if name in cls.AWAY_HISTORY_INPUTS:
            if checks.get("away_history_sufficient"):
                return ("available", None)
            if checks.get("away_history_available"):
                return ("partial", "Deplasman geçmişi beş maçtan az.")
            return ("missing", "Deplasman için yakın maç verisi yok.")
        if name in cls.H2H_INPUTS:
            return (
                ("available", None)
                if checks.get("h2h_available")
                else ("missing", "İkili rekabet verisi bulunamadı.")
            )
        if name in {"home_elo", "away_elo"}:
            side = "home" if name.startswith("home_") else "away"
            if checks.get(f"{side}_elo_available"):
                return ("available", None)
            if checks.get(f"{side}_history_available"):
                return ("partial", "Elo sınırlı tarihsel maçla hesaplandı.")
            return ("missing", "Elo için tarihsel maç yok.")
        if name in cls.AVAILABILITY_INPUTS:
            return (
                ("available", None)
                if checks.get("availability_available")
                else ("missing", "Sakatlık/ceza raporu bulunamadı.")
            )
        if name in cls.LINEUP_INPUTS:
            return (
                ("available", None)
                if checks.get("lineups_available")
                else ("missing", "İlk 11 verisi henüz açıklanmadı.")
            )
        if name in cls.PLAYER_IMPACT_INPUTS:
            side = "home" if name.startswith("home_") else "away"
            return (
                ("available", None)
                if checks.get(f"{side}_player_impact_available")
                else ("missing", "Yeterli oyuncu rating/kadro verisi yok.")
            )
        if name == "rest_days_diff":
            available = bool(
                checks.get("kickoff_known")
                and checks.get("home_history_sufficient")
                and checks.get("away_history_sufficient")
            )
            return (
                ("available", None)
                if available
                else ("missing", "Dinlenme hesabı için maç geçmişi eksik.")
            )
        if name == "fatigue_index":
            available = bool(
                checks.get("kickoff_known")
                and checks.get("home_history_sufficient")
                and checks.get("away_history_sufficient")
            )
            if not available:
                return ("missing", "Yorgunluk hesabı için takvim verisi eksik.")
            if not checks.get("travel_context_available"):
                return ("partial", "Seyahat mesafesi yok; kalan bileşenler kullanıldı.")
            return ("available", None)
        if name in cls.ODDS_MOVEMENT_INPUTS:
            return (
                ("available", None)
                if checks.get("odds_movement_available")
                else ("missing", "Açılış ve güncel oran çifti bulunamadı.")
            )
        if name in cls.WEATHER_INPUTS:
            return (
                ("available", None)
                if checks.get("weather_available")
                else ("missing", "Stadyum konumu veya maç saati hava verisi yok.")
            )
        if name in cls.IDENTITY_INPUTS:
            return (
                ("available", None)
                if float(calculated.get(name, 0.0)) > 0
                else ("missing", "Kimlik bilgisi bulunamadı.")
            )
        if name.startswith("league_"):
            return (
                ("available", None)
                if checks.get("league_identified")
                else ("missing", "Lig seçilmediği için gösterge nötr.")
            )
        return ("available", None)

    @classmethod
    def build(
        cls,
        calculated: Mapping[str, float],
        overrides: Mapping[str, float],
        data_quality: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        checks = data_quality.get("checks")
        safe_checks = checks if isinstance(checks, Mapping) else {}
        provenance = data_quality.get("feature_provenance")
        safe_provenance = provenance if isinstance(provenance, Mapping) else {}
        rows: list[dict[str, Any]] = []
        for name in FeatureEngine.FEATURE_NAMES:
            calculated_value = float(
                calculated.get(name, FeatureEngine.FEATURE_DEFAULTS[name])
            )
            availability, missing_reason = cls._availability(
                name, safe_checks, calculated
            )
            overridden = name in overrides
            feature_provenance = cls._provenance(
                name=name,
                availability=availability,
                overridden=overridden,
                explicit=safe_provenance.get(name),
                data_quality=data_quality,
            )
            minimum, maximum, step = cls._bounds(name)
            label = ExplainabilityService.FEATURE_LABELS.get(name)
            if label is None and name.startswith("league_"):
                label = f"Lig göstergesi ({name.removeprefix('league_')})"
            rows.append(
                {
                    "name": name,
                    "label": label or name,
                    "group": cls._group(name),
                    "value": float(overrides.get(name, calculated_value)),
                    "calculated_value": calculated_value,
                    "default_value": float(FeatureEngine.FEATURE_DEFAULTS[name]),
                    "availability": "manual" if overridden else availability,
                    "missing_reason": None if overridden else missing_reason,
                    "overridden": overridden,
                    "minimum": minimum,
                    "maximum": maximum,
                    "step": step,
                    **feature_provenance,
                }
            )
        return rows

    @classmethod
    def _provenance(
        cls,
        *,
        name: str,
        availability: str,
        overridden: bool,
        explicit: object,
        data_quality: Mapping[str, Any],
    ) -> dict[str, object]:
        if overridden:
            return {
                "source": "manual_override",
                "captured_at": None,
                "confidence": 1.0,
                "is_fallback": False,
            }
        if isinstance(explicit, Mapping):
            confidence = explicit.get("confidence", 0.0)
            return {
                "source": str(explicit.get("source") or "unknown"),
                "captured_at": explicit.get("captured_at"),
                "confidence": (
                    float(confidence) if isinstance(confidence, (int, float)) else 0.0
                ),
                "is_fallback": bool(explicit.get("is_fallback", False)),
            }

        source = cls._inferred_source(name)
        captured_at = None
        if name in cls.ODDS_MOVEMENT_INPUTS:
            odds_snapshot = data_quality.get("odds_snapshot")
            if isinstance(odds_snapshot, Mapping):
                captured_at = odds_snapshot.get("current_captured_at")
        if availability == "missing":
            source = "neutral_default"
        confidence = {
            "available": 1.0,
            "partial": 0.6,
            "missing": 0.0,
        }.get(availability, 0.0)
        return {
            "source": source,
            "captured_at": captured_at,
            "confidence": confidence,
            "is_fallback": availability in {"partial", "missing"},
        }

    @classmethod
    def _inferred_source(cls, name: str) -> str:
        if name in cls.DIRECT_INPUTS:
            return "analysis_form"
        if name in cls.IDENTITY_INPUTS or name.startswith("league_"):
            return "fixture_metadata"
        if name in cls.HOME_HISTORY_INPUTS | cls.AWAY_HISTORY_INPUTS:
            return "historical_fixtures"
        if name in cls.H2H_INPUTS:
            return "head_to_head_history"
        if name in cls.AVAILABILITY_INPUTS:
            return "api_football_availability"
        if name in cls.LINEUP_INPUTS:
            return "api_football_lineups"
        if name in cls.PLAYER_IMPACT_INPUTS:
            return "player_context"
        if name in cls.ODDS_MOVEMENT_INPUTS:
            return "market_snapshot"
        if name in {"rest_days_diff", "fatigue_index"}:
            return "schedule_context"
        return "calculated"
