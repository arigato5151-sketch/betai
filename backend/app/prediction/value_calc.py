import math
from typing import Dict, List, Optional


class ValueCalc:
    KELLY_FRACTION = 0.25
    OUTCOME_KEYS = ("HOME_WIN", "DRAW", "AWAY_WIN")
    API_VALUES = {"Home": "HOME_WIN", "Draw": "DRAW", "Away": "AWAY_WIN"}

    OUTCOME_LABELS = {
        "HOME_WIN": "Ev Galibiyeti (1)",
        "DRAW": "Beraberlik (X)",
        "AWAY_WIN": "Deplasman (2)",
    }

    @staticmethod
    def devig_1x2(home_odd: float, draw_odd: float, away_odd: float) -> Dict:
        try:
            raw = {
                "HOME_WIN": float(home_odd),
                "DRAW": float(draw_odd),
                "AWAY_WIN": float(away_odd),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("1X2 odds must be numeric") from exc
        if any(not math.isfinite(odd) or odd <= 1.0 for odd in raw.values()):
            raise ValueError("1X2 odds must be finite and greater than 1.0")

        implied = {k: 1.0 / v for k, v in raw.items()}
        total_implied = sum(implied.values())
        fair_prob = {k: round((implied[k] / total_implied) * 100, 2) for k in implied}

        fair_odds = {
            k: round(100.0 / fair_prob[k], 2) if fair_prob[k] > 0 else 0.0
            for k in fair_prob
        }
        overround = round((total_implied * 100.0 - 100.0), 2)

        return {
            "raw_odds": {k: round(v, 2) for k, v in raw.items()},
            "implied_probability": {k: round(implied[k] * 100, 2) for k in implied},
            "fair_probability": fair_prob,
            "fair_odds": fair_odds,
            "overround_pct": overround,
            "method": "proportional_devig",
        }

    @staticmethod
    def parse_from_api_bets(bets: list) -> Optional[Dict]:
        for market in bets:
            if market.get("name") != "Match Winner":
                continue
            odds_map = {}
            for val in market.get("values", []):
                key = ValueCalc.API_VALUES.get(val.get("value"))
                if key:
                    try:
                        odds_map[key] = float(val.get("odd"))
                    except (TypeError, ValueError):
                        pass
            if len(odds_map) == 3:
                try:
                    return ValueCalc.devig_1x2(
                        odds_map["HOME_WIN"],
                        odds_map["DRAW"],
                        odds_map["AWAY_WIN"],
                    )
                except ValueError:
                    continue
        return None

    @staticmethod
    def best_market_from_bookmakers(bookmakers: list) -> Optional[Dict]:
        best: Optional[Dict] = None
        best_overround = float("inf")

        for bookmaker in bookmakers:
            bets = bookmaker.get("bets", [])
            market = ValueCalc.parse_from_api_bets(bets)
            if not market:
                continue
            overround = float(market.get("overround_pct", 999))
            if overround < best_overround:
                best_overround = overround
                best = market
                best["bookmaker"] = bookmaker.get("name")

        return best

    @staticmethod
    def default_market_from_model(
        model_probs: Dict[str, float],
        home_odd_hint: Optional[float] = None,
        overround_pct: float = 5.0,
    ) -> Dict:
        if not math.isfinite(overround_pct) or not 0 <= overround_pct <= 100:
            raise ValueError("overround_pct must be between 0 and 100")

        probs = {}
        for key in ValueCalc.OUTCOME_KEYS:
            probability = float(model_probs.get(key, 33.33))
            if not math.isfinite(probability) or probability < 0:
                raise ValueError("Model probabilities must be finite and non-negative")
            probs[key] = max(0.01, probability)
        total = sum(probs.values())
        fair_prob = {k: round((probs[k] / total) * 100, 2) for k in probs}

        factor = 1.0 + (overround_pct / 100.0)
        if home_odd_hint is not None:
            if not math.isfinite(home_odd_hint) or home_odd_hint <= 1.0:
                raise ValueError("home_odd_hint must be finite and greater than 1.0")
            home_implied = 1.0 / home_odd_hint
            remaining_implied = factor - home_implied
            remaining_probability = fair_prob["DRAW"] + fair_prob["AWAY_WIN"]
            draw_share = fair_prob["DRAW"] / remaining_probability
            draw_implied = remaining_implied * draw_share
            away_implied = remaining_implied - draw_implied
            raw_odds = {
                "HOME_WIN": round(home_odd_hint, 2),
                "DRAW": round(1.0 / draw_implied, 2),
                "AWAY_WIN": round(1.0 / away_implied, 2),
            }
        else:
            raw_odds = {k: round((100.0 / fair_prob[k]) / factor, 2) for k in fair_prob}

        return ValueCalc.devig_1x2(
            raw_odds["HOME_WIN"],
            raw_odds["DRAW"],
            raw_odds["AWAY_WIN"],
        )

    @staticmethod
    def default_market(
        home_odd: float = 1.85, model_probs: Optional[Dict[str, float]] = None
    ) -> Dict:
        if model_probs:
            return ValueCalc.default_market_from_model(
                model_probs, home_odd_hint=home_odd
            )
        neutral = {"HOME_WIN": 40.0, "DRAW": 28.0, "AWAY_WIN": 32.0}
        return ValueCalc.default_market_from_model(neutral, home_odd_hint=home_odd)

    @staticmethod
    def _edge_threshold_ratio(implied_probability: float) -> float:
        """ROI based minimum edge threshold ratio."""
        if implied_probability > 60.0:
            return 0.03
        if implied_probability >= 30.0:
            return 0.05
        return 0.08

    @staticmethod
    def _max_kelly_pct(odd: float) -> float:
        if odd < 2.0:
            return 5.0
        if odd <= 3.0:
            return 3.0
        return 1.5

    @staticmethod
    def _is_value_bet(edge_pct: float, implied_probability: float) -> bool:
        if (
            not math.isfinite(edge_pct)
            or not math.isfinite(implied_probability)
            or implied_probability <= 0
        ):
            return False
        # Convert edge back to raw fraction for ratio checking
        edge_fraction = edge_pct / 100.0
        implied_fraction = implied_probability / 100.0
        ratio = edge_fraction / implied_fraction
        return ratio >= ValueCalc._edge_threshold_ratio(implied_probability)

    @staticmethod
    def calculate_professional(
        analysis: dict,
        market: Optional[Dict],
        fallback_odd: Optional[float] = None,
    ) -> dict:
        model_probs = analysis["all_probabilities"]

        if market and market.get("fair_probability"):
            return ValueCalc._evaluate_with_market(model_probs, market)

        if fallback_odd and fallback_odd > 1.0:
            synthetic = ValueCalc.default_market(fallback_odd, model_probs=model_probs)
            synthetic["raw_odds"]["HOME_WIN"] = fallback_odd
            result = ValueCalc._evaluate_with_market(model_probs, synthetic)
            result["market"] = "SINGLE_ODD"
            return result

        return {
            "value_bet": False,
            "edge": 0.0,
            "implied_probability": 0.0,
            "fair_odd": 0.0,
            "best_pick": None,
            "value_options": [],
            "market": None,
        }

    @staticmethod
    def _evaluate_with_market(model_probs: Dict[str, float], market: Dict) -> dict:
        fair_probs = market["fair_probability"]
        raw_odds = market.get("raw_odds", {})
        candidates: List[dict] = []

        for outcome in ValueCalc.OUTCOME_KEYS:
            model_p = model_probs.get(outcome, 0.0)
            odd = raw_odds.get(outcome, 0.0)

            # Strict edge formula: Edge = Model_Prob * Bookmaker_Odds - 1
            # Here model_p is in percentage, so we divide by 100
            model_p_frac = model_p / 100.0
            edge = model_p_frac * odd - 1.0 if odd > 1.0 else 0.0
            edge_pct = round(edge * 100.0, 2)

            implied_p = (100.0 / odd) if odd > 1.0 else 0.0

            item = {
                "outcome": outcome,
                "label": ValueCalc.OUTCOME_LABELS[outcome],
                "probability": model_p,
                "market_probability": fair_probs.get(outcome, 0.0),
                "edge": edge_pct,
                "implied_probability": round(implied_p, 2),
                "fair_odd": market.get("fair_odds", {}).get(outcome, 0.0),
                "raw_odd": odd,
                "value_bet": (
                    ValueCalc._is_value_bet(edge_pct, implied_p) if odd > 1.0 else False
                ),
                "kelly_stake_pct": (
                    ValueCalc._kelly_stake(model_p, odd) if odd > 1.0 else 0.0
                ),
            }
            if item["value_bet"]:
                candidates.append(item)

        best_pick = max(candidates, key=lambda x: x["edge"]) if candidates else None
        main_outcome = max(model_probs, key=lambda outcome: model_probs[outcome])
        main_odd = raw_odds.get(main_outcome, 0.0)

        main_edge = (
            (model_probs[main_outcome] / 100.0) * main_odd - 1.0
            if main_odd > 1.0
            else 0.0
        )
        main_implied = (100.0 / main_odd) if main_odd > 1.0 else 0.0

        return {
            "value_bet": bool(best_pick),
            "edge": best_pick["edge"] if best_pick else round(main_edge * 100, 2),
            "implied_probability": round(main_implied, 2),
            "fair_odd": market.get("fair_odds", {}).get(main_outcome, 0.0),
            "best_pick": best_pick,
            "value_options": sorted(candidates, key=lambda x: x["edge"], reverse=True),
            "market": market,
            "market_comparison": {
                k: round(model_probs.get(k, 0) - fair_probs.get(k, 0), 1)
                for k in ValueCalc.OUTCOME_KEYS
            },
            "overround_pct": market.get("overround_pct", 0.0),
            "devig_method": market.get("method", "proportional_devig"),
        }

    @staticmethod
    def _kelly_stake(model_prob_pct: float, odd: float) -> float:
        if not math.isfinite(model_prob_pct) or not 0 <= model_prob_pct <= 100:
            raise ValueError("Model probability must be between 0 and 100")
        if not math.isfinite(odd):
            raise ValueError("Odd must be finite")
        p = model_prob_pct / 100.0
        b = odd - 1.0
        if b <= 0 or p <= 0:
            return 0.0
        q = 1.0 - p
        kelly = (b * p - q) / b
        if kelly <= 0:
            return 0.0
        stake = kelly * ValueCalc.KELLY_FRACTION * 100
        return round(min(ValueCalc._max_kelly_pct(odd), stake), 2)
