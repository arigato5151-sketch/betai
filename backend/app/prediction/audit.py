from typing import List, Dict, Any
from app.db.models import MatchPrediction


class PredictionAuditor:
    @staticmethod
    def calculate_bet_roi(
        prediction: str | None, actual_result: str | None, odd: float | None
    ) -> float:
        """
        Calculates fractional ROI for a single bet.
        If won: Odd - 1
        If lost: -1
        """
        if not prediction or not actual_result or odd is None or odd <= 1.0:
            return 0.0

        if prediction == actual_result:
            return round(odd - 1.0, 4)
        return -1.0

    @staticmethod
    def calculate_clv(placed_odds: float | None, closing_odds: float | None) -> float:
        """
        Calculates Closing Line Value (CLV) as a fractional percentage.
        CLV = (Placed Odds / Closing Odds) - 1.0
        """
        if not placed_odds or not closing_odds or closing_odds <= 0:
            return 0.0
        return round((placed_odds / closing_odds) - 1.0, 4)

    @staticmethod
    def audit_predictions(predictions: List[MatchPrediction]) -> Dict[str, Any]:
        """
        Runs a full statistical audit on a list of resolved predictions.
        Computes:
        - Win Rate
        - Total ROI (based on unit staking)
        - Multi-class Brier Score (HOME_WIN, DRAW, AWAY_WIN)
        - Average CLV
        - Total bets placed
        """
        resolved = [p for p in predictions if p.actual_result is not None]
        total_bets = len(resolved)

        if total_bets == 0:
            return {
                "total_predictions": 0,
                "win_rate_pct": 0.0,
                "total_roi_pct": 0.0,
                "brier_score": 0.0,
                "avg_clv_pct": 0.0,
            }

        correct_bets = sum(1 for p in resolved if p.prediction == p.actual_result)
        win_rate = (correct_bets / total_bets) * 100.0

        # ROI calculations (unit betting)
        total_profit_loss = 0.0
        for p in resolved:
            p_roi = PredictionAuditor.calculate_bet_roi(
                p.prediction, p.actual_result, p.odd
            )
            total_profit_loss += p_roi
        avg_roi = (total_profit_loss / total_bets) * 100.0

        # Multi-class Brier Score calculation
        # BS = (1/N) * sum_t( sum_i( (f_ti - o_ti)^2 ) )
        # Outcomes: HOME_WIN, DRAW, AWAY_WIN
        brier_sum = 0.0
        clv_sum = 0.0
        clv_count = 0

        for p in resolved:
            # Model probabilities (scaled to 0.0 - 1.0)
            f_home = (p.prob_home or 33.33) / 100.0
            f_draw = (p.prob_draw or 33.33) / 100.0
            f_away = (p.prob_away or 33.33) / 100.0

            # Actual outcome one-hot encoding
            o_home = 1.0 if p.actual_result == "HOME_WIN" else 0.0
            o_draw = 1.0 if p.actual_result == "DRAW" else 0.0
            o_away = 1.0 if p.actual_result == "AWAY_WIN" else 0.0

            brier_sum += (
                (f_home - o_home) ** 2 + (f_draw - o_draw) ** 2 + (f_away - o_away) ** 2
            )

            if p.closing_odds and p.closing_odds > 0:
                clv_sum += PredictionAuditor.calculate_clv(p.odd, p.closing_odds)
                clv_count += 1

        brier_score = brier_sum / total_bets
        avg_clv = (clv_sum / clv_count) * 100.0 if clv_count > 0 else 0.0

        return {
            "total_predictions": total_bets,
            "win_rate_pct": round(win_rate, 2),
            "total_roi_pct": round(avg_roi, 2),
            "brier_score": round(brier_score, 4),
            "avg_clv_pct": round(avg_clv, 2),
        }
