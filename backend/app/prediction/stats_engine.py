import math
from typing import Dict, List, Optional

from app.core.config import settings

MAX_GOALS = 7
MODEL_VERSION = "poisson_dixon_coles_v3"


def build_team_profile(api_data: Optional[Dict], venue: str) -> Dict:
    """
    venue: 'home' | 'away' - perspective of the team profile.
    """
    if not api_data:
        return _default_profile(venue, source="fallback_default")

    goals = api_data.get("goals", {})
    goals_for = _avg_goals(goals.get("for", {}), venue)
    goals_against = _avg_goals(goals.get("against", {}), venue)
    form_string = api_data.get("form", "")

    attack_strength = max(
        0.55, min(1.75, goals_for / settings.LEAGUE_BASELINE_GOALS)
    )
    defense_strength = max(
        0.55, min(1.75, goals_against / settings.LEAGUE_BASELINE_GOALS)
    )

    form_score = _decay_form_score(form_string)
    clean_sheets = _extract_count(api_data.get("clean_sheet", {}), venue)
    failed_score = _extract_count(api_data.get("failed_to_score", {}), venue)
    played = _extract_played(api_data.get("fixtures", {}), venue)

    xg = _estimate_xg(
        goals_for, goals_against, attack_strength, defense_strength, venue
    )
    attack_index = _to_index(attack_strength)
    defense_index = _to_index(2.0 - defense_strength)
    strength = round(
        (
            attack_index * settings.STRENGTH_ATTACK_WEIGHT
            + defense_index * settings.STRENGTH_DEFENSE_WEIGHT
            + form_score * settings.STRENGTH_FORM_WEIGHT
        ),
        1,
    )

    return {
        "form": form_score,
        "attack": attack_index,
        "defense": defense_index,
        "xg": xg,
        "attack_strength": round(attack_strength, 3),
        "defense_strength": round(defense_strength, 3),
        "goals_for_avg": round(goals_for, 2),
        "goals_against_avg": round(goals_against, 2),
        "strength_rating": strength,
        "clean_sheet_rate": round((clean_sheets / played * 100) if played else 0, 1),
        "failed_to_score_rate": round(
            (failed_score / played * 100) if played else 0, 1
        ),
        "venue": venue,
        "source": "api_football_season_stats",
        "method": "home_away_split_decay_form",
    }


def _avg_goals(goals_side: Dict, venue: str) -> float:
    average = goals_side.get("average", {})
    raw = average.get(venue) or average.get("total") or 1.2
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.2


def _decay_form_score(form_string: str) -> int:
    if not form_string:
        return 50
    recent = form_string.strip()[-5:].upper()
    points_map = {"W": 3, "D": 1, "L": 0}
    weighted_points = 0.0
    max_points = 0.0
    for idx, char in enumerate(reversed(recent)):
        weight = (
            settings.FORM_DECAY_WEIGHTS[idx]
            if idx < len(settings.FORM_DECAY_WEIGHTS)
            else settings.FORM_DECAY_FALLBACK_WEIGHT
        )
        pts = points_map.get(char, 1)
        weighted_points += pts * weight
        max_points += 3 * weight
    if max_points <= 0:
        return 50
    return int(round((weighted_points / max_points) * 100))


def _estimate_xg(
    goals_for: float,
    goals_against: float,
    attack_strength: float,
    defense_strength: float,
    venue: str,
) -> float:
    base = (
        goals_for * settings.XG_OBSERVED_GOALS_WEIGHT
        + settings.LEAGUE_BASELINE_GOALS
        * attack_strength
        * settings.XG_ATTACK_BASELINE_WEIGHT
    )
    if venue == "home":
        base *= settings.HOME_ATTACK_BOOST
    else:
        base *= settings.AWAY_ATTACK_PENALTY
    consistency = 1.0 - min(
        settings.XG_CONSISTENCY_MAX_PENALTY,
        abs(goals_for - goals_against) * settings.XG_CONSISTENCY_PENALTY_WEIGHT,
    )
    return round(max(0.35, min(3.5, base * consistency)), 2)


def _to_index(strength: float) -> int:
    return int(max(0, min(100, round((strength - 0.5) / 1.25 * 100))))


def _extract_count(block: Dict, venue: str) -> int:
    if not block:
        return 0
    val = block.get(venue) or block.get("total") or 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _extract_played(fixtures: Dict, venue: str) -> int:
    played = fixtures.get("played", {}) if fixtures else {}
    val = played.get(venue) or played.get("total") or 1
    try:
        return max(1, int(val))
    except (TypeError, ValueError):
        return 1


def _default_profile(venue: str, source: str) -> Dict:
    return {
        "form": 50,
        "attack": 50,
        "defense": 50,
        "xg": 1.2,
        "attack_strength": 1.0,
        "defense_strength": 1.0,
        "goals_for_avg": 1.2,
        "goals_against_avg": 1.2,
        "strength_rating": 50.0,
        "clean_sheet_rate": 0.0,
        "failed_to_score_rate": 0.0,
        "venue": venue,
        "source": source,
        "method": "fallback",
    }


class StatsEngine:
    OUTCOME_LABELS = {
        "HOME_WIN": "Ev Sahibi Galibiyeti",
        "DRAW": "Beraberlik",
        "AWAY_WIN": "Deplasman Galibiyeti",
    }

    PROFILE_LABELS = {
        "HOME_FAVORITE": "Ev Sahibi Öne Çıkıyor",
        "AWAY_FAVORITE": "Deplasman Öne Çıkıyor",
        "BALANCED": "Dengeli Maç",
        "DRAW_LIKELY": "Beraberlik Adayı",
        "HIGH_SCORING": "Göllü Maç Beklentisi",
        "LOW_SCORING": "Düşük Skor Beklentisi",
        "OPEN_GAME": "Açık Oyun / KG Var Eğilimi",
    }

    @staticmethod
    def analyze_match(
        home_stats: dict,
        away_stats: dict,
        league_id: Optional[int] = None,
    ) -> dict:
        rho = (
            settings.LEAGUE_DIXON_COLES_RHO.get(
                league_id, settings.DEFAULT_DIXON_COLES_RHO
            )
            if league_id
            else settings.DEFAULT_DIXON_COLES_RHO
        )
        home_lambda = StatsEngine._expected_goals(home_stats, away_stats, is_home=True)
        away_lambda = StatsEngine._expected_goals(away_stats, home_stats, is_home=False)

        matrix = StatsEngine._score_probability_matrix(
            home_lambda, away_lambda, rho=rho
        )
        result_probs = StatsEngine._result_probabilities(matrix)
        over_under = StatsEngine._over_under_probs(matrix)
        btts = StatsEngine._btts_probs(matrix)
        expected_score = StatsEngine._most_likely_score(matrix)
        score_band = StatsEngine._score_band(matrix)

        final_prediction = max(result_probs, key=lambda outcome: result_probs[outcome])
        sorted_outcomes = sorted(result_probs.items(), key=lambda x: x[1], reverse=True)

        alternate_picks = [
            {
                "outcome": key,
                "label": StatsEngine.OUTCOME_LABELS[key],
                "probability": value,
            }
            for key, value in sorted_outcomes
        ]

        secondary_markets = StatsEngine._build_secondary_markets(
            over_under, btts, home_lambda, away_lambda
        )
        match_profile = StatsEngine._match_profile(
            result_probs, home_lambda, away_lambda, home_stats, away_stats
        )

        confidence_gap = sorted_outcomes[0][1] - sorted_outcomes[1][1]

        return {
            "model": MODEL_VERSION,
            "prediction": final_prediction,
            "probability": result_probs[final_prediction],
            "all_probabilities": result_probs,
            "confidence_gap": round(confidence_gap, 2),
            "confidence_tier": StatsEngine._confidence_tier(confidence_gap),
            "expected_goals": {
                "home": round(home_lambda, 2),
                "away": round(away_lambda, 2),
                "total": round(home_lambda + away_lambda, 2),
            },
            "expected_score": expected_score,
            "score_band": score_band,
            "alternate_picks": alternate_picks,
            "secondary_markets": secondary_markets,
            "match_profile": match_profile,
        }

    @staticmethod
    def _expected_goals(team: dict, opponent: dict, is_home: bool) -> float:
        """Calculate lambda utilizing Poisson regression style logic from team profiles."""
        if team.get("attack_strength") and opponent.get("defense_strength"):
            attack_s = float(team["attack_strength"])
            defense_weakness = float(opponent["defense_strength"])
            form_factor = settings.PROFILE_FORM_FACTOR_BASE + (
                float(team.get("form", 50)) / 100.0
            ) * settings.PROFILE_FORM_FACTOR_WEIGHT
            lambda_goals = (
                settings.LEAGUE_BASELINE_GOALS
                * attack_s
                * defense_weakness
                * form_factor
            )
        else:
            attack_factor = settings.LEGACY_ATTACK_FACTOR_BASE + (
                team["attack"] / 100.0
            ) * settings.LEGACY_ATTACK_FACTOR_WEIGHT
            defense_factor = settings.LEGACY_DEFENSE_FACTOR_BASE + (
                (100 - opponent["defense"]) / 100.0
            ) * settings.LEGACY_DEFENSE_FACTOR_WEIGHT
            form_factor = settings.LEGACY_FORM_FACTOR_BASE + (
                team["form"] / 100.0
            ) * settings.LEGACY_FORM_FACTOR_WEIGHT
            xg_base = (
                team["xg"] * settings.LEGACY_XG_OBSERVED_WEIGHT
                + settings.LEAGUE_BASELINE_GOALS
                * settings.LEGACY_XG_BASELINE_WEIGHT
            )
            lambda_goals = xg_base * attack_factor * defense_factor * form_factor

        if is_home:
            lambda_goals *= StatsEngine._home_advantage_multiplier(team, opponent)
        else:
            lambda_goals *= settings.AWAY_ATTACK_PENALTY

        return max(0.35, min(3.4, lambda_goals))

    @staticmethod
    def _home_advantage_multiplier(team: dict, opponent: dict) -> float:
        home_gf = float(team.get("goals_for_avg") or 0)
        away_ga = float(opponent.get("goals_against_avg") or 0)
        if home_gf > 0 and away_ga > 0:
            ratio = home_gf / max(
                settings.HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR, away_ga
            )
            return max(
                settings.HOME_ADVANTAGE_MIN_MULTIPLIER,
                min(settings.HOME_ADVANTAGE_MAX_MULTIPLIER, ratio),
            )
        form_boost = max(
            0.0,
            (float(team.get("form", 50)) - 50.0)
            / settings.HOME_FORM_BOOST_DIVISOR,
        )
        return settings.HOME_FORM_BASE_MULTIPLIER + form_boost

    @staticmethod
    def _dixon_coles_adjustment(
        home_goals: int,
        away_goals: int,
        home_lambda: float,
        away_lambda: float,
        rho: float = settings.DEFAULT_DIXON_COLES_RHO,
    ) -> float:
        if home_goals == 0 and away_goals == 0:
            return 1.0 - (home_lambda * away_lambda * rho)
        if home_goals == 0 and away_goals == 1:
            return 1.0 + (away_lambda * rho)
        if home_goals == 1 and away_goals == 0:
            return 1.0 + (home_lambda * rho)
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0

    @staticmethod
    def _poisson_pmf(rate: float, goals: int) -> float:
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("Poisson rate must be finite and non-negative")
        if goals < 0:
            raise ValueError("Goal count cannot be negative")
        return math.exp(-rate) * (rate**goals) / math.factorial(goals)

    @staticmethod
    def _score_probability_matrix(
        home_lambda: float,
        away_lambda: float,
        rho: float = settings.DEFAULT_DIXON_COLES_RHO,
    ) -> List[List[float]]:
        if not math.isfinite(rho):
            raise ValueError("Dixon-Coles rho must be finite")
        matrix: List[List[float]] = []
        for home_goals in range(MAX_GOALS + 1):
            row = []
            p_home = StatsEngine._poisson_pmf(home_lambda, home_goals)
            for away_goals in range(MAX_GOALS + 1):
                p_away = StatsEngine._poisson_pmf(away_lambda, away_goals)
                tau = StatsEngine._dixon_coles_adjustment(
                    home_goals, away_goals, home_lambda, away_lambda, rho=rho
                )
                if tau < 0:
                    raise ValueError(
                        "Dixon-Coles adjustment produced a negative weight"
                    )
                row.append(p_home * p_away * tau)
            matrix.append(row)

        total = sum(sum(r) for r in matrix)
        if total <= 0:
            return matrix
        return [[cell / total for cell in row] for row in matrix]

    @staticmethod
    def _result_probabilities(matrix: List[List[float]]) -> Dict[str, float]:
        home_win = draw = away_win = 0.0
        for home_goals, row in enumerate(matrix):
            for away_goals, prob in enumerate(row):
                if home_goals > away_goals:
                    home_win += prob
                elif home_goals == away_goals:
                    draw += prob
                else:
                    away_win += prob

        raw = {"HOME_WIN": home_win, "DRAW": draw, "AWAY_WIN": away_win}
        return {key: round(value * 100, 2) for key, value in raw.items()}

    @staticmethod
    def _over_under_probs(matrix: List[List[float]]) -> Dict[str, float]:
        over_25 = under_25 = over_15 = 0.0
        for home_goals, row in enumerate(matrix):
            for away_goals, prob in enumerate(row):
                total_goals = home_goals + away_goals
                if total_goals > 2.5:
                    over_25 += prob
                else:
                    under_25 += prob
                if total_goals > 1.5:
                    over_15 += prob

        return {
            "over_2_5": round(over_25 * 100, 2),
            "under_2_5": round(under_25 * 100, 2),
            "over_1_5": round(over_15 * 100, 2),
        }

    @staticmethod
    def _btts_probs(matrix: List[List[float]]) -> Dict[str, float]:
        yes = 0.0
        for home_goals, row in enumerate(matrix):
            for away_goals, prob in enumerate(row):
                if home_goals >= 1 and away_goals >= 1:
                    yes += prob
        no = 1.0 - yes
        return {"yes": round(yes * 100, 2), "no": round(no * 100, 2)}

    @staticmethod
    def _most_likely_score(matrix: List[List[float]]) -> Dict[str, object]:
        best_prob = -1.0
        best_home = best_away = 0
        for home_goals, row in enumerate(matrix):
            for away_goals, prob in enumerate(row):
                if prob > best_prob:
                    best_prob = prob
                    best_home = home_goals
                    best_away = away_goals

        return {
            "home": best_home,
            "away": best_away,
            "label": f"{best_home}-{best_away}",
            "probability": round(best_prob * 100, 2),
        }

    @staticmethod
    def _score_band(matrix: List[List[float]]) -> str:
        low = mid = high = 0.0
        for home_goals, row in enumerate(matrix):
            for away_goals, prob in enumerate(row):
                total = home_goals + away_goals
                if total <= 2:
                    low += prob
                elif total <= 4:
                    mid += prob
                else:
                    high += prob

        bands = {"0-2 Gol": low, "3-4 Gol": mid, "5+ Gol": high}
        return max(bands, key=lambda band: bands[band])

    @staticmethod
    def _build_secondary_markets(
        over_under: Dict[str, float],
        btts: Dict[str, float],
        home_lambda: float,
        away_lambda: float,
    ) -> List[Dict[str, object]]:
        markets = [
            {
                "market": "OVER_2_5",
                "label": "Üst 2.5 Gol",
                "pick": "UST" if over_under["over_2_5"] >= 50 else "ALT",
                "probability": max(over_under["over_2_5"], over_under["under_2_5"]),
            },
            {
                "market": "BTTS",
                "label": "Karşılıklı Gol (KG)",
                "pick": "VAR" if btts["yes"] >= 50 else "YOK",
                "probability": max(btts["yes"], btts["no"]),
            },
            {
                "market": "OVER_1_5",
                "label": "Üst 1.5 Gol",
                "pick": "UST",
                "probability": over_under["over_1_5"],
            },
        ]

        if home_lambda >= 1.55 and away_lambda >= 1.35:
            markets.append(
                {
                    "market": "DOUBLE_CHANCE_1X",
                    "label": "Çifte Şans 1-X",
                    "pick": "1-X",
                    "probability": round(
                        min(
                            95.0,
                            55
                            + (home_lambda - away_lambda)
                            * settings.DOUBLE_CHANCE_HOME_DIFFERENCE_WEIGHT,
                        ),
                        2,
                    ),
                }
            )
        elif away_lambda > home_lambda + 0.25:
            markets.append(
                {
                    "market": "DOUBLE_CHANCE_X2",
                    "label": "Çifte Şans X-2",
                    "pick": "X-2",
                    "probability": round(
                        min(
                            95.0,
                            52
                            + (away_lambda - home_lambda)
                            * settings.DOUBLE_CHANCE_AWAY_DIFFERENCE_WEIGHT,
                        ),
                        2,
                    ),
                }
            )

        return markets

    @staticmethod
    def _match_profile(
        result_probs: Dict[str, float],
        home_lambda: float,
        away_lambda: float,
        home_stats: dict,
        away_stats: dict,
    ) -> Dict[str, str]:
        total_xg = home_lambda + away_lambda
        draw_prob = result_probs["DRAW"]
        home_prob = result_probs["HOME_WIN"]
        away_prob = result_probs["AWAY_WIN"]

        if total_xg >= 3.1:
            key = "HIGH_SCORING"
        elif total_xg <= 2.1:
            key = "LOW_SCORING"
        elif draw_prob >= 30 and abs(home_prob - away_prob) < 8:
            key = "DRAW_LIKELY"
        elif home_prob >= away_prob + 14:
            key = "HOME_FAVORITE"
        elif away_prob >= home_prob + 12:
            key = "AWAY_FAVORITE"
        elif home_stats.get("attack", 50) > 72 and away_stats.get("attack", 50) > 72:
            key = "OPEN_GAME"
        else:
            key = "BALANCED"

        return {
            "code": key,
            "label": StatsEngine.PROFILE_LABELS[key],
            "summary": StatsEngine._profile_summary(key, total_xg, draw_prob),
        }

    @staticmethod
    def build_insights(analysis: dict, value_data: dict) -> List[str]:
        insights: List[str] = []

        raw_prediction = analysis.get("prediction")
        prediction = raw_prediction if isinstance(raw_prediction, str) else ""
        prediction_label = StatsEngine.OUTCOME_LABELS.get(
            prediction, prediction or "Tahmin"
        )
        probability = analysis.get("probability", 0)
        confidence_tier = analysis.get("confidence_tier", "DUSUK")
        confidence_gap = analysis.get("confidence_gap", 0)
        insights.append(
            f"{prediction_label} olasiligi %{probability}; guven seviyesi {confidence_tier} "
            f"(fark %{confidence_gap})."
        )

        expected_goals = analysis.get("expected_goals") or {}
        if expected_goals:
            insights.append(
                "Beklenen gol dengesi: "
                f"ev {expected_goals.get('home', 0)}, deplasman {expected_goals.get('away', 0)}, "
                f"toplam {expected_goals.get('total', 0)}."
            )

        expected_score = analysis.get("expected_score") or {}
        if expected_score.get("label"):
            insights.append(
                f"En olasi skor {expected_score['label']} "
                f"(%{expected_score.get('probability', 0)})."
            )

        match_profile = analysis.get("match_profile") or {}
        if match_profile.get("summary"):
            insights.append(match_profile["summary"])

        best_pick = value_data.get("best_pick") if value_data else None
        if best_pick:
            insights.append(
                f"Value sinyali: {best_pick.get('label')} icin edge %{best_pick.get('edge')} "
                f"ve Kelly onerisi %{best_pick.get('kelly_stake_pct', 0)}."
            )
        elif value_data:
            insights.append(
                f"Value sinyali zayif; ana edge %{value_data.get('edge', 0)} seviyesinde."
            )

        return insights

    @staticmethod
    def _profile_summary(key: str, total_xg: float, draw_prob: float) -> str:
        summaries = {
            "HOME_FAVORITE": "Ev sahibi gol beklentisi ve formuyla öne çıkıyor.",
            "AWAY_FAVORITE": "Deplasman takımı istatistiksel üstünlük kuruyor.",
            "BALANCED": "Takımlar birbirine yakın; sonuç dalgalanabilir.",
            "DRAW_LIKELY": "Benzer güç profili beraberliği destekliyor.",
            "HIGH_SCORING": f"Toplam gol beklentisi yüksek ({total_xg:.1f}).",
            "LOW_SCORING": f"Defans ağırlıklı, düşük skor profili ({total_xg:.1f}).",
            "OPEN_GAME": "Her iki taraf da gol üretme potansiyeline sahip.",
        }
        return summaries.get(key, f"Beraberlik ihtimali %{draw_prob:.0f} civarında.")

    @staticmethod
    def _confidence_tier(gap: float) -> str:
        if gap >= 18:
            return "YUKSEK"
        if gap >= 10:
            return "ORTA"
        return "DUSUK"
