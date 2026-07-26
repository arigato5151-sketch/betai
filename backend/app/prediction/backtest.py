import math
import logging
import numpy as np
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from typing import List, Dict, Any, Tuple
from app.db.models import MatchPrediction
from app.prediction.audit import PredictionAuditor

logger = logging.getLogger("bet-ai-pro.backtest")


class BacktestEngine:
    OUTCOMES = ("HOME_WIN", "DRAW", "AWAY_WIN")

    @classmethod
    def multiclass_brier_score(
        cls,
        forecasts: Iterable[tuple[Mapping[str, float], str]],
    ) -> float:
        """Return the mean multiclass Brier score for normalized probabilities."""
        scores: list[float] = []
        for probabilities, actual_result in forecasts:
            if actual_result not in cls.OUTCOMES:
                raise ValueError(f"Unsupported actual result: {actual_result!r}")
            try:
                values = np.asarray(
                    [float(probabilities[outcome]) for outcome in cls.OUTCOMES],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Forecast must contain three numeric outcomes"
                ) from exc
            if (
                not np.all(np.isfinite(values))
                or np.any(values < 0)
                or values.sum() <= 0
            ):
                raise ValueError(
                    "Forecast probabilities must be finite and non-negative"
                )
            values /= values.sum()
            target = np.zeros(len(cls.OUTCOMES), dtype=float)
            target[cls.OUTCOMES.index(actual_result)] = 1.0
            scores.append(float(np.sum((values - target) ** 2)))
        if not scores:
            raise ValueError("At least one forecast is required")
        return float(np.mean(scores))

    @staticmethod
    def run_simulation(
        predictions: List[MatchPrediction],
        initial_bankroll: float = 1000.0,
        strategy: str = "kelly",  # "kelly", "flat", "fractional_kelly"
        flat_stake_amount: float = 10.0,
        kelly_fraction: float = 0.25,
        min_edge_pct: float = 3.0,
        commission_pct: float = 0.0,
        max_stake_pct: float = 100.0,
        max_daily_exposure_pct: float = 100.0,
        require_closing_odds: bool = False,
        exclude_post_kickoff: bool = True,
    ) -> Dict[str, Any]:
        """
        Kronolojik tahmin simulasyonu: geçmiş tahminleri test et.
        Metriks: doğruluk, ROI, Sharpe, max drawdown, kalibrasyon
        """
        if initial_bankroll <= 0:
            raise ValueError("initial_bankroll must be positive")
        if strategy not in {"kelly", "flat", "fractional_kelly"}:
            raise ValueError("unsupported backtest strategy")
        if flat_stake_amount <= 0:
            raise ValueError("flat_stake_amount must be positive")
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        if min_edge_pct < 0:
            raise ValueError("min_edge_pct cannot be negative")
        if not 0 <= commission_pct <= 20:
            raise ValueError("commission_pct must be between 0 and 20")
        if not 0 < max_stake_pct <= 100:
            raise ValueError("max_stake_pct must be between 0 and 100")
        if not 0 < max_daily_exposure_pct <= 100:
            raise ValueError("max_daily_exposure_pct must be between 0 and 100")

        # Chronologically sort resolved predictions
        resolved = sorted(
            [p for p in predictions if p.actual_result is not None],
            key=lambda prediction: BacktestEngine._timestamp(
                prediction.analyzed_at or prediction.created_at
            ),
        )

        if not resolved:
            logger.warning("No resolved predictions for backtest")
            return {
                "initial_bankroll": initial_bankroll,
                "final_bankroll": initial_bankroll,
                "total_roi_pct": 0.0,
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "accuracy_pct": 0.0,
                "calibration_score": 0.0,
                "profit_factor": 0.0,
                "risk_of_ruin_pct": 0.0,
                "closing_odds_coverage_pct": 0.0,
                "total_staked": 0.0,
                "skipped_reasons": {},
                "bankroll_history": [initial_bankroll],
            }

        current_bankroll = initial_bankroll
        bankroll_history = [initial_bankroll]
        returns = []
        drawdowns = []

        peak = initial_bankroll
        max_drawdown = 0.0
        wins = 0
        losses = 0
        total_staked = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        skipped: Counter[str] = Counter()
        current_day: date | None = None
        day_start_bankroll = initial_bankroll
        day_exposure = 0.0

        # For calibration: store predicted vs actual
        calibration_data = []

        for p in resolved:
            analysis_time = p.analyzed_at or p.created_at
            if (
                exclude_post_kickoff
                and analysis_time is not None
                and p.kickoff is not None
                and BacktestEngine._timestamp(analysis_time)
                >= BacktestEngine._timestamp(p.kickoff)
            ):
                skipped["post_kickoff_analysis"] += 1
                continue
            bet_day = analysis_time.date() if analysis_time else date.min
            if bet_day != current_day:
                current_day = bet_day
                day_start_bankroll = current_bankroll
                day_exposure = 0.0

            # Check edge threshold
            if p.edge is None or p.edge < min_edge_pct:
                skipped["below_edge"] += 1
                continue
            if p.odd is None or p.odd <= 1.0:
                skipped["invalid_odds"] += 1
                continue
            if require_closing_odds and (
                p.closing_odds is None or p.closing_odds <= 1.0
            ):
                skipped["missing_closing_odds"] += 1
                continue

            # Determine bet stake
            if strategy == "flat":
                stake = flat_stake_amount
            elif strategy == "kelly":
                # Defense-in-depth for legacy/imported records: never risk over 5%.
                stake_pct = min(max(p.kelly_stake or 0.0, 0.0), 5.0)
                stake = current_bankroll * (stake_pct / 100.0)
            elif strategy == "fractional_kelly":
                full_kelly_pct = min(max(p.kelly_stake or 0.0, 0.0), 5.0)
                stake_pct = full_kelly_pct * kelly_fraction
                stake = current_bankroll * (stake_pct / 100.0)

            # Portfolio-level guards prevent a single signal or busy day from
            # consuming more capital than configured.
            stake = min(stake, current_bankroll * max_stake_pct / 100.0)
            daily_budget = day_start_bankroll * max_daily_exposure_pct / 100.0
            remaining_daily_budget = max(0.0, daily_budget - day_exposure)
            stake = min(stake, current_bankroll, remaining_daily_budget)
            if stake <= 0:
                skipped["daily_exposure_limit"] += 1
                continue

            # Outcome check
            is_win = p.prediction == p.actual_result
            bet_roi = PredictionAuditor.calculate_bet_roi(
                p.prediction, p.actual_result, p.odd
            )
            gross_return = stake * bet_roi
            net_return = (
                gross_return * (1.0 - commission_pct / 100.0)
                if gross_return > 0
                else gross_return
            )

            current_bankroll += net_return
            bankroll_history.append(current_bankroll)
            day_exposure += stake
            total_staked += stake
            if net_return > 0:
                gross_profit += net_return
            else:
                gross_loss += abs(net_return)

            # Save fractional return
            returns.append(net_return / max(1.0, current_bankroll - net_return))

            if is_win:
                wins += 1
            else:
                losses += 1

            # Drawdown tracking
            if current_bankroll > peak:
                peak = current_bankroll

            dd = (peak - current_bankroll) / peak if peak > 0 else 0.0
            drawdowns.append(dd)
            if dd > max_drawdown:
                max_drawdown = dd

            # Calibration data: predicted prob vs actual outcome
            pred_prob = (p.probability or 50.0) / 100.0  # 0-1 range
            actual_outcome = 1 if is_win else 0
            calibration_data.append((pred_prob, actual_outcome))

            # Early termination if bankrupt
            if current_bankroll <= 0:
                current_bankroll = 0.0
                break

        total_bets = wins + losses
        total_roi = (
            ((current_bankroll - initial_bankroll) / initial_bankroll) * 100.0
            if initial_bankroll > 0
            else 0.0
        )

        # Risk-adjusted metrics
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

        if len(returns) > 1:
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))

            # Sharpe Ratio (risk-free rate = 0)
            if std_return > 0:
                sharpe_ratio = (mean_return / std_return) * math.sqrt(252)  # Annualized

            # Sortino Ratio (only downside risk)
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 0:
                downside_std = float(np.std(downside_returns))
                if downside_std > 0:
                    sortino_ratio = (mean_return / downside_std) * math.sqrt(252)

        # Accuracy: correct predictions / total predictions
        accuracy_pct = (wins / total_bets * 100) if total_bets > 0 else 0.0

        # Calibration score: ECE (Expected Calibration Error)
        calibration_score = BacktestEngine._compute_calibration_error(calibration_data)
        closing_count = sum(
            1 for prediction in resolved if (prediction.closing_odds or 0.0) > 1.0
        )
        closing_coverage = closing_count / len(resolved) * 100.0
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else gross_profit if gross_profit > 0 else 0.0
        )
        risk_of_ruin = BacktestEngine._bootstrap_risk_of_ruin(returns)

        logger.info(
            f"Backtest completed: {wins}W {losses}L, ROI {total_roi:.2f}%, "
            f"Accuracy {accuracy_pct:.1f}%, Calibration {calibration_score:.3f}"
        )

        return {
            "initial_bankroll": initial_bankroll,
            "final_bankroll": round(current_bankroll, 2),
            "total_roi_pct": round(total_roi, 2),
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(accuracy_pct, 2),
            "accuracy_pct": round(accuracy_pct, 2),  # Alias
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "sortino_ratio": round(sortino_ratio, 4),
            "calibration_score": round(calibration_score, 4),
            "profit_factor": round(profit_factor, 4),
            "risk_of_ruin_pct": round(risk_of_ruin, 2),
            "closing_odds_coverage_pct": round(closing_coverage, 2),
            "total_staked": round(total_staked, 2),
            "commission_pct": commission_pct,
            "max_stake_pct": max_stake_pct,
            "max_daily_exposure_pct": max_daily_exposure_pct,
            "exclude_post_kickoff": exclude_post_kickoff,
            "skipped_reasons": dict(sorted(skipped.items())),
            "bankroll_history": [round(b, 2) for b in bankroll_history],
        }

    @staticmethod
    def _timestamp(value: datetime | None) -> float:
        if value is None:
            return 0.0
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return normalized.timestamp()

    @staticmethod
    def _bootstrap_risk_of_ruin(
        returns: List[float],
        *,
        simulations: int = 1000,
        ruin_threshold: float = 0.2,
    ) -> float:
        """Estimate path risk by bootstrapping the observed fractional returns."""
        if not returns:
            return 0.0
        rng = np.random.default_rng(42)
        samples = rng.choice(
            np.asarray(returns, dtype=float),
            size=(simulations, len(returns)),
            replace=True,
        )
        paths = np.cumprod(1.0 + samples, axis=1)
        ruined = np.any(paths <= ruin_threshold, axis=1)
        return float(np.mean(ruined) * 100.0)

    @staticmethod
    def _compute_calibration_error(
        calibration_data: List[Tuple[float, int]], n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error: predicted prob vs actual win rate.
        Bin prob predictions, compare avg predicted vs actual in each bin.
        Lower is better (0 = perfect calibration).
        """
        if not calibration_data or len(calibration_data) < 2:
            return 0.0

        # Bin predictions into n_bins
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(calibration_data)

        for i in range(n_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]

            # Find predictions in this bin
            bin_predictions = [
                (pred, actual)
                for pred, actual in calibration_data
                if bin_lower <= pred < bin_upper or (i == n_bins - 1 and pred == 1.0)
            ]

            if not bin_predictions:
                continue

            # Calibration error for this bin
            avg_pred = float(np.mean([p for p, _ in bin_predictions]))
            actual_rate = float(np.mean([a for _, a in bin_predictions]))

            bin_weight = len(bin_predictions) / total_samples
            ece += bin_weight * abs(avg_pred - actual_rate)

        return ece
