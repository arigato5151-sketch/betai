import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class FeatureEngine:
    FEATURE_NAMES = [
        "home_form",
        "home_attack",
        "home_defense",
        "home_xg",
        "away_form",
        "away_attack",
        "away_defense",
        "away_xg",
        "home_form_ema",
        "away_form_ema",
        "rest_days_diff",
        "home_clean_sheet_streak",
        "away_clean_sheet_streak",
        "home_scoring_streak",
        "away_scoring_streak",
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_home_loss_rate",
        "home_elo",
        "away_elo",
        # Enhanced features
        "home_advantage_coeff",  # Ev sahibi avantajı katsayısı (0.95-1.15)
        "home_gf_last5",  # Son 5 maçta atılan gol ortalaması
        "home_ga_last5",  # Son 5 maçta yenilen gol ortalaması
        "away_gf_last5",  # Deplasman: atılan gol ortalaması
        "away_ga_last5",  # Deplasman: yenilen gol ortalaması
        "h2h_avg_goals_home",  # H2H maçlarda ev sahibinin ort. golü
        "h2h_avg_goals_away",  # H2H maçlarda deplasmandaki ort. golü
    ]

    @staticmethod
    def calculate_elo_ratings(
        matches: List[Dict[str, Any]], k_factor: float = 32.0
    ) -> Dict[int, float]:
        """
        Chronologically calculates Elo ratings for all teams.
        Default rating is 1500.0.
        """
        elo_ratings: Dict[int, float] = {}

        # Sort matches by date
        sorted_matches = sorted(
            [m for m in matches if m.get("actual_result") is not None],
            key=lambda match: str(match.get("created_at") or ""),
        )

        for match in sorted_matches:
            h_id = match.get("home_team_id")
            a_id = match.get("away_team_id")
            result = match.get("actual_result")

            if not h_id or not a_id or result not in {"HOME_WIN", "DRAW", "AWAY_WIN"}:
                continue

            r_home = elo_ratings.setdefault(h_id, 1500.0)
            r_away = elo_ratings.setdefault(a_id, 1500.0)

            # Expected scores
            e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
            e_away = 1.0 / (1.0 + 10.0 ** ((r_home - r_away) / 400.0))

            # Actual scores
            if result == "HOME_WIN":
                s_home, s_away = 1.0, 0.0
            elif result == "AWAY_WIN":
                s_home, s_away = 0.0, 1.0
            else:
                s_home, s_away = 0.5, 0.5

            # Update ratings
            elo_ratings[h_id] = r_home + k_factor * (s_home - e_home)
            elo_ratings[a_id] = r_away + k_factor * (s_away - e_away)

        return elo_ratings

    @staticmethod
    def compute_form_ema(matches_df: pd.DataFrame, span: int = 5) -> float:
        if matches_df.empty or "points" not in matches_df.columns:
            return 50.0
        df = matches_df.sort_values("match_date", ascending=True).reset_index(drop=True)
        span = int(max(1, min(span, len(df))))
        points = pd.to_numeric(df["points"], errors="coerce").fillna(0.0).clip(0.0, 3.0)
        ema_points = float(points.ewm(span=span, adjust=False).mean().iloc[-1])
        return float(np.clip((ema_points / 3.0) * 100.0, 0.0, 100.0))

    @staticmethod
    def compute_streak(matches_df: pd.DataFrame, col: str, max_len: int = 5) -> int:
        """Belirtilen sütuna göre streak sayısını hesapla (max 5)."""
        if matches_df.empty or col not in matches_df.columns:
            return 0
        df = matches_df.sort_values("match_date", ascending=False).reset_index(
            drop=True
        )
        streak = 0
        for i in range(min(max_len, len(df))):
            value = pd.to_numeric(pd.Series([df.loc[i, col]]), errors="coerce").iloc[0]
            if pd.notna(value) and int(value) == 1:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def compute_goals_avg(
        matches_df: pd.DataFrame, for_col: str, against_col: str
    ) -> tuple:
        """
        Son 5 maçta atılan/yenilen gol ortalamasını hesapla.
        Returns: (goals_for_avg, goals_against_avg)
        """
        if matches_df.empty:
            return (0.0, 0.0)

        df = matches_df.sort_values("match_date", ascending=False).reset_index(
            drop=True
        )
        df = df.head(5)  # Son 5 maçı al

        gf_avg = (
            float(pd.to_numeric(df[for_col], errors="coerce").fillna(0.0).mean())
            if for_col in df.columns
            else 0.0
        )
        ga_avg = (
            float(pd.to_numeric(df[against_col], errors="coerce").fillna(0.0).mean())
            if against_col in df.columns
            else 0.0
        )

        return (round(gf_avg, 2), round(ga_avg, 2))

    @staticmethod
    def compute_home_advantage_coeff(home_matches_df: pd.DataFrame) -> float:
        """
        Ev sahibinin son 3 maçındaki win rate'ine dayalı avantaj katsayısı.
        Range: 0.95 (zayıf) - 1.15 (güçlü)
        """
        if home_matches_df.empty:
            return 1.0  # Varsayılan

        df = home_matches_df.sort_values("match_date", ascending=False).reset_index(
            drop=True
        )
        df = df.head(3)  # Son 3 maçı al

        if len(df) < 1:
            return 1.0

        # Win rate hesapla (home match win rate)
        wins = (df.get("result", "") == "W").sum() if "result" in df.columns else 0
        win_rate = float(wins) / len(df)

        # 1.0 + (win_rate - 0.5) * 0.3
        # win_rate 0 ise: 1.0 + (-0.5)*0.3 = 0.85
        # win_rate 1 ise: 1.0 + (0.5)*0.3 = 1.15
        coeff = 1.0 + (win_rate - 0.5) * 0.3
        return round(np.clip(coeff, 0.95, 1.15), 3)

    @staticmethod
    def compute_h2h_goals(h2h_matches: List[Dict]) -> tuple:
        """
        H2H maçlarında ev sahibi ve deplasman takımının ort. gollerini hesapla.
        Returns: (home_goals_avg, away_goals_avg)
        """
        if not h2h_matches or len(h2h_matches) < 1:
            return (1.2, 1.0)  # Varsayılan

        home_goals = pd.to_numeric(
            pd.Series([m.get("home_goals") for m in h2h_matches]), errors="coerce"
        ).dropna()
        away_goals = pd.to_numeric(
            pd.Series([m.get("away_goals") for m in h2h_matches]), errors="coerce"
        ).dropna()

        home_avg = round(float(home_goals.mean()), 2) if not home_goals.empty else 1.2
        away_avg = round(float(away_goals.mean()), 2) if not away_goals.empty else 1.0

        return (home_avg, away_avg)

    @staticmethod
    def build_inference_features(
        home_stats: Dict[str, Any],
        away_stats: Dict[str, Any],
        home_matches_df: pd.DataFrame,
        away_matches_df: pd.DataFrame,
        h2h_rates: Dict[str, float],
        h2h_matches: Optional[List[Dict]] = None,
        home_elo: float = 1500.0,
        away_elo: float = 1500.0,
        fixture_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """
        Robust feature vector inşa et - training'de kullanılan aynı formüllerle.
        Yeni enriched features: home_advantage_coeff, last5 GF/GA, H2H goals
        """
        fixture_date = fixture_date or pd.Timestamp.today().normalize()
        h2h_matches = h2h_matches or []

        # Dynamic EMA and streaks calculations
        home_form_ema = FeatureEngine.compute_form_ema(home_matches_df, span=5)
        away_form_ema = FeatureEngine.compute_form_ema(away_matches_df, span=5)

        home_rest = away_rest = 7
        if not home_matches_df.empty:
            home_rest = max(
                1, int((fixture_date - home_matches_df["match_date"].max()).days)
            )
        if not away_matches_df.empty:
            away_rest = max(
                1, int((fixture_date - away_matches_df["match_date"].max()).days)
            )

        rest_days_diff = float(home_rest - away_rest)

        home_cs_streak = FeatureEngine.compute_streak(home_matches_df, "clean_sheet")
        away_cs_streak = FeatureEngine.compute_streak(away_matches_df, "clean_sheet")
        home_score_streak = FeatureEngine.compute_streak(home_matches_df, "scoring")
        away_score_streak = FeatureEngine.compute_streak(away_matches_df, "scoring")

        # Enhanced features: last 5 matches goals
        home_gf_last5, home_ga_last5 = FeatureEngine.compute_goals_avg(
            home_matches_df, "goals_for", "goals_against"
        )
        away_gf_last5, away_ga_last5 = FeatureEngine.compute_goals_avg(
            away_matches_df, "goals_for", "goals_against"
        )

        # Enhanced features: home advantage coefficient
        home_advantage_coeff = FeatureEngine.compute_home_advantage_coeff(
            home_matches_df
        )

        # Enhanced features: H2H goals averages
        h2h_home_avg_goals, h2h_away_avg_goals = FeatureEngine.compute_h2h_goals(
            h2h_matches
        )

        return {
            "home_form": float(home_stats.get("form", 50.0)),
            "home_attack": float(home_stats.get("attack", 50.0)),
            "home_defense": float(home_stats.get("defense", 50.0)),
            "home_xg": float(home_stats.get("xg", 1.2)),
            "away_form": float(away_stats.get("form", 50.0)),
            "away_attack": float(away_stats.get("attack", 50.0)),
            "away_defense": float(away_stats.get("defense", 50.0)),
            "away_xg": float(away_stats.get("xg", 1.2)),
            "home_form_ema": float(home_form_ema),
            "away_form_ema": float(away_form_ema),
            "rest_days_diff": rest_days_diff,
            "home_clean_sheet_streak": float(home_cs_streak),
            "away_clean_sheet_streak": float(away_cs_streak),
            "home_scoring_streak": float(home_score_streak),
            "away_scoring_streak": float(away_score_streak),
            "h2h_home_win_rate": float(h2h_rates.get("home_win_rate", 0.33)),
            "h2h_draw_rate": float(h2h_rates.get("draw_rate", 0.33)),
            "h2h_home_loss_rate": float(h2h_rates.get("home_loss_rate", 0.34)),
            "home_elo": float(home_elo),
            "away_elo": float(away_elo),
            # New enriched features
            "home_advantage_coeff": home_advantage_coeff,
            "home_gf_last5": home_gf_last5,
            "home_ga_last5": home_ga_last5,
            "away_gf_last5": away_gf_last5,
            "away_ga_last5": away_ga_last5,
            "h2h_avg_goals_home": h2h_home_avg_goals,
            "h2h_avg_goals_away": h2h_away_avg_goals,
        }
