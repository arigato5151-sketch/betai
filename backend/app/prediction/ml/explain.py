from typing import Any

import numpy as np

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplainabilityService:
    GENERIC_EXPLANATION = "Tahmin sebebi: yeterli özellik katkısı bulunamadı."

    FEATURE_LABELS = {
        "home_form": "Ev Sahibi Formu",
        "home_attack": "Ev Sahibi Hücum Gücü",
        "home_defense": "Ev Sahibi Savunma Gücü",
        "home_xg": "Ev Sahibi Gol Beklentisi (xG)",
        "away_form": "Deplasman Formu",
        "away_attack": "Deplasman Hücum Gücü",
        "away_defense": "Deplasman Savunma Gücü",
        "away_xg": "Deplasman Gol Beklentisi (xG)",
        "home_form_ema": "Ev Sahibi Ağırlıklı Form (EMA)",
        "away_form_ema": "Deplasman Ağırlıklı Form (EMA)",
        "rest_days_diff": "Dinlenme Günü Farkı",
        "home_clean_sheet_streak": "Ev Sahibi Gol Yememe Serisi",
        "away_clean_sheet_streak": "Deplasman Gol Yememe Serisi",
        "home_scoring_streak": "Ev Sahibi Gol Atma Serisi",
        "away_scoring_streak": "Deplasman Gol Atma Serisi",
        "h2h_home_win_rate": "H2H Geçmişi (Ev Galibiyeti)",
        "h2h_draw_rate": "H2H Geçmişi (Beraberlik)",
        "h2h_home_loss_rate": "H2H Geçmişi (Deplasman Galibiyeti)",
        "home_elo": "Ev Sahibi ELO Derecesi",
        "away_elo": "Deplasman ELO Derecesi",
        "home_team_strength_ratio": "Ev Sahibi Kadro Kalite Oranı",
        "away_team_strength_ratio": "Deplasman Kadro Kalite Oranı",
        "fatigue_index": "Yorgunluk ve Seyahat Endeksi",
        "home_advantage_coeff": "Ev Sahibi Avantaj Katsayısı",
        "home_gf_last5": "Ev Sahibinin Son 5 Maç Gol Ortalaması",
        "home_ga_last5": "Ev Sahibinin Son 5 Maç Yediği Gol Ortalaması",
        "away_gf_last5": "Deplasmanın Son 5 Maç Gol Ortalaması",
        "away_ga_last5": "Deplasmanın Son 5 Maç Yediği Gol Ortalaması",
        "h2h_avg_goals_home": "İkili Rekabette Ev Sahibi Gol Ortalaması",
        "h2h_avg_goals_away": "İkili Rekabette Deplasman Gol Ortalaması",
        "home_missing_players": "Ev Sahibi Eksik Oyuncu Sayısı",
        "away_missing_players": "Deplasman Eksik Oyuncu Sayısı",
        "home_questionable_players": "Ev Sahibi Durumu Belirsiz Oyuncu Sayısı",
        "away_questionable_players": "Deplasman Durumu Belirsiz Oyuncu Sayısı",
        "availability_report_present": "Oyuncu Uygunluk Raporu",
        "home_lineup_confirmed": "Ev Sahibi Kadro Onayı",
        "away_lineup_confirmed": "Deplasman Kadro Onayı",
        "home_lineup_reference_available": "Ev Sahibi Referans Kadrosu",
        "away_lineup_reference_available": "Deplasman Referans Kadrosu",
        "home_lineup_continuity": "Ev Sahibi Kadro Sürekliliği",
        "away_lineup_continuity": "Deplasman Kadro Sürekliliği",
        "odds_movement_home": "Ev Sahibi Oran Değişimi",
        "odds_movement_draw": "Beraberlik Oran Değişimi",
        "odds_movement_away": "Deplasman Oran Değişimi",
        "weather_temperature_c": "Maç Saati Sıcaklığı",
        "weather_precipitation_mm": "Maç Saati Yağışı",
        "weather_wind_speed_kmh": "Maç Saati Rüzgâr Hızı",
        "weather_available": "Hava Verisi Durumu",
        "league_id": "Lig Kimliği",
        "home_team_id": "Ev Sahibi Takım Kimliği",
        "away_team_id": "Deplasman Takım Kimliği",
    }

    @staticmethod
    def generate_explanation(
        model: Any, features_dict: dict[str, float], feature_names: list[str]
    ) -> list[str]:
        """Return the three strongest, safely normalized feature contributions."""
        if not feature_names:
            return [ExplainabilityService.GENERIC_EXPLANATION]

        contributions: list[tuple[str, float]] = []
        if SHAP_AVAILABLE:
            try:
                x_vec = np.array(
                    [[features_dict.get(name, 0.0) for name in feature_names]],
                    dtype=float,
                )
                shap_values = shap.TreeExplainer(model).shap_values(x_vec)
                mean_shap = ExplainabilityService._aggregate_shap_values(shap_values)
                contributions = ExplainabilityService._normalize_contributions(
                    mean_shap, feature_names, absolute=True
                )
            except Exception:
                # Optional SHAP integrations vary by model/library version.
                contributions = []

        if not contributions:
            contributions = ExplainabilityService._get_importance_fallback(
                model, feature_names
            )

        sorted_contributions = sorted(
            contributions, key=lambda item: item[1], reverse=True
        )[:3]
        if not sorted_contributions:
            return [ExplainabilityService.GENERIC_EXPLANATION]

        reasons = []
        for name, weight in sorted_contributions:
            label = ExplainabilityService.FEATURE_LABELS.get(name, name)
            reasons.append(f"{label} (+%{round(weight * 100)})")

        return [f"Tahmin sebebi: {', '.join(reasons)}"]

    @staticmethod
    def _aggregate_shap_values(shap_values: Any) -> np.ndarray:
        """Collapse supported SHAP output layouts into one feature vector."""
        if isinstance(shap_values, list):
            if not shap_values:
                return np.array([], dtype=float)
            rows = [
                np.asarray(class_values, dtype=float)[0] for class_values in shap_values
            ]
            return np.mean(np.abs(rows), axis=0)

        values = np.asarray(shap_values, dtype=float)
        if values.ndim == 3:
            # SHAP may return (samples, features, classes).
            return np.mean(np.abs(values[0]), axis=1)
        if values.ndim == 2:
            return np.abs(values[0])
        return np.array([], dtype=float)

    @staticmethod
    def _normalize_contributions(
        values: Any,
        feature_names: list[str],
        *,
        absolute: bool,
    ) -> list[tuple[str, float]]:
        if not feature_names or values is None:
            return []

        try:
            weights = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return []

        if weights.size < len(feature_names):
            return []

        weights = weights[: len(feature_names)]
        weights = np.abs(weights) if absolute else np.clip(weights, 0.0, None)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            return []

        return [
            (name, float(weights[index] / total))
            for index, name in enumerate(feature_names)
        ]

    @staticmethod
    def _get_importance_fallback(
        model: Any, feature_names: list[str]
    ) -> list[tuple[str, float]]:
        if not feature_names:
            return []

        importances = getattr(model, "feature_importances_", None)
        contributions = ExplainabilityService._normalize_contributions(
            importances, feature_names, absolute=False
        )
        if contributions:
            return contributions

        uniform_weight = 1.0 / len(feature_names)
        return [(name, uniform_weight) for name in feature_names]
