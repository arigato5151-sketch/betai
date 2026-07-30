import math
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from app.core.config import settings
from app.prediction.player_impact import TeamStrengthImpact

MAX_GOALS = 7
MODEL_VERSION = "poisson_dixon_coles_v5"


def time_weighted_goal_averages(
    match_history: pd.DataFrame | None,
    *,
    as_of: datetime | pd.Timestamp | None = None,
    decay_factor: float | None = None,
) -> tuple[float | None, float | None]:
    """Return exponentially weighted goals-for/against averages without leakage."""
    if (
        not isinstance(match_history, pd.DataFrame)
        or match_history.empty
        or "match_date" not in match_history.columns
    ):
        return (None, None)

    factor = (
        settings.GOAL_TIME_DECAY_FACTOR if decay_factor is None else float(decay_factor)
    )
    if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError("decay_factor must be finite and between 0 and 1")

    dates = pd.to_datetime(match_history["match_date"], errors="coerce", utc=True)
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return (None, None)

    if as_of is None:
        reference = valid_dates.max()
    else:
        try:
            reference = pd.Timestamp(as_of)
        except (TypeError, ValueError):
            return (None, None)
        if pd.isna(reference):
            return (None, None)
        reference = (
            reference.tz_localize("UTC")
            if reference.tzinfo is None
            else reference.tz_convert("UTC")
        )

    days_ago = (reference - dates).dt.total_seconds() / 86400.0
    historical_mask = dates.notna() & days_ago.ge(0.0)

    def weighted_average(column: str) -> float | None:
        if column not in match_history.columns:
            return None
        values = pd.to_numeric(match_history[column], errors="coerce")
        observations: list[tuple[float, float]] = []
        for value, age, is_historical in zip(
            values,
            days_ago,
            historical_mask,
        ):
            if not bool(is_historical) or pd.isna(value) or pd.isna(age):
                continue
            numeric_value = float(value)
            numeric_age = float(age)
            if (
                not math.isfinite(numeric_value)
                or not math.isfinite(numeric_age)
                or numeric_value < 0
            ):
                continue
            observations.append((numeric_value, numeric_age))

        if not observations:
            return None

        # Shifting by the youngest age preserves the normalized average and
        # prevents every weight from underflowing for very old histories.
        youngest_age = min(age for _, age in observations)
        weighted_values: list[float] = []
        weights: list[float] = []
        for value, age in observations:
            weight = math.exp(-factor * (age - youngest_age))
            weighted_values.append(value * weight)
            weights.append(weight)
        denominator = math.fsum(weights)
        if denominator <= 0 or not math.isfinite(denominator):
            return None
        return math.fsum(weighted_values) / denominator

    return (
        weighted_average("goals_for"),
        weighted_average("goals_against"),
    )


def build_team_profile(
    api_data: Optional[Dict],
    venue: str,
    *,
    match_history: pd.DataFrame | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    decay_factor: float | None = None,
) -> Dict:
    """
    venue: 'home' | 'away' - perspective of the team profile.
    """
    if not api_data and (
        not isinstance(match_history, pd.DataFrame) or match_history.empty
    ):
        return _default_profile(venue, source="fallback_default")

    api_data = api_data or {}
    goals = api_data.get("goals", {})
    weighted_for, weighted_against = time_weighted_goal_averages(
        match_history,
        as_of=as_of,
        decay_factor=decay_factor,
    )
    goals_for = (
        weighted_for
        if weighted_for is not None
        else _avg_goals(goals.get("for", {}), venue)
    )
    goals_against = (
        weighted_against
        if weighted_against is not None
        else _avg_goals(goals.get("against", {}), venue)
    )
    form_string = api_data.get("form", "")

    attack_strength = max(0.55, min(1.75, goals_for / settings.LEAGUE_BASELINE_GOALS))
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
        "source": (
            "time_weighted_match_history"
            if weighted_for is not None or weighted_against is not None
            else "api_football_season_stats"
        ),
        "method": (
            "time_weighted_goal_decay"
            if weighted_for is not None or weighted_against is not None
            else "home_away_split_decay_form"
        ),
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


def _apply_time_weighted_goal_profile(
    team_stats: dict,
    match_history: pd.DataFrame | None,
    *,
    as_of: datetime | pd.Timestamp | None,
) -> dict:
    goals_for, goals_against = time_weighted_goal_averages(
        match_history,
        as_of=as_of,
    )
    if goals_for is None and goals_against is None:
        return dict(team_stats)

    profile = dict(team_stats)
    if goals_for is not None:
        profile["goals_for_avg"] = round(goals_for, 2)
        profile["attack_strength"] = round(
            max(0.55, min(1.75, goals_for / settings.LEAGUE_BASELINE_GOALS)),
            3,
        )
    if goals_against is not None:
        profile["goals_against_avg"] = round(goals_against, 2)
        profile["defense_strength"] = round(
            max(0.55, min(1.75, goals_against / settings.LEAGUE_BASELINE_GOALS)),
            3,
        )
    return profile


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
        *,
        home_match_history: pd.DataFrame | None = None,
        away_match_history: pd.DataFrame | None = None,
        as_of: datetime | pd.Timestamp | None = None,
        home_player_impact: TeamStrengthImpact | None = None,
        away_player_impact: TeamStrengthImpact | None = None,
    ) -> dict:
        home_stats = _apply_time_weighted_goal_profile(
            home_stats,
            home_match_history,
            as_of=as_of,
        )
        away_stats = _apply_time_weighted_goal_profile(
            away_stats,
            away_match_history,
            as_of=as_of,
        )
        rho = (
            settings.LEAGUE_DIXON_COLES_RHO.get(
                league_id, settings.DEFAULT_DIXON_COLES_RHO
            )
            if league_id
            else settings.DEFAULT_DIXON_COLES_RHO
        )
        home_multiplier = (
            home_player_impact.xg_multiplier if home_player_impact is not None else 1.0
        )
        away_multiplier = (
            away_player_impact.xg_multiplier if away_player_impact is not None else 1.0
        )
        home_lambda = StatsEngine._expected_goals(
            home_stats,
            away_stats,
            is_home=True,
            player_xg_multiplier=home_multiplier,
        )
        away_lambda = StatsEngine._expected_goals(
            away_stats,
            home_stats,
            is_home=False,
            player_xg_multiplier=away_multiplier,
        )

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
            "player_impact": {
                "home": StatsEngine._player_impact_diagnostics(home_player_impact),
                "away": StatsEngine._player_impact_diagnostics(away_player_impact),
            },
            "expected_score": expected_score,
            "score_band": score_band,
            "alternate_picks": alternate_picks,
            "secondary_markets": secondary_markets,
            "match_profile": match_profile,
        }

    @staticmethod
    def _expected_goals(
        team: dict,
        opponent: dict,
        is_home: bool,
        player_xg_multiplier: float = 1.0,
    ) -> float:
        """Calculate lambda utilizing Poisson regression style logic from team profiles."""
        if team.get("attack_strength") and opponent.get("defense_strength"):
            attack_s = float(team["attack_strength"])
            defense_weakness = float(opponent["defense_strength"])
            form_factor = (
                settings.PROFILE_FORM_FACTOR_BASE
                + (float(team.get("form", 50)) / 100.0)
                * settings.PROFILE_FORM_FACTOR_WEIGHT
            )
            lambda_goals = (
                settings.LEAGUE_BASELINE_GOALS
                * attack_s
                * defense_weakness
                * form_factor
            )
        else:
            attack_factor = (
                settings.LEGACY_ATTACK_FACTOR_BASE
                + (team["attack"] / 100.0) * settings.LEGACY_ATTACK_FACTOR_WEIGHT
            )
            defense_factor = (
                settings.LEGACY_DEFENSE_FACTOR_BASE
                + ((100 - opponent["defense"]) / 100.0)
                * settings.LEGACY_DEFENSE_FACTOR_WEIGHT
            )
            form_factor = (
                settings.LEGACY_FORM_FACTOR_BASE
                + (team["form"] / 100.0) * settings.LEGACY_FORM_FACTOR_WEIGHT
            )
            xg_base = (
                team["xg"] * settings.LEGACY_XG_OBSERVED_WEIGHT
                + settings.LEAGUE_BASELINE_GOALS * settings.LEGACY_XG_BASELINE_WEIGHT
            )
            lambda_goals = xg_base * attack_factor * defense_factor * form_factor

        if is_home:
            lambda_goals *= StatsEngine._home_advantage_multiplier(team, opponent)
        else:
            lambda_goals *= settings.AWAY_ATTACK_PENALTY

        try:
            multiplier = float(player_xg_multiplier)
        except (TypeError, ValueError):
            multiplier = 1.0
        if not math.isfinite(multiplier):
            multiplier = 1.0
        multiplier = max(
            settings.PLAYER_IMPACT_MIN_XG_MULTIPLIER,
            min(settings.PLAYER_IMPACT_MAX_STRENGTH_RATIO, multiplier),
        )
        lambda_goals *= multiplier

        return max(0.35, min(3.4, lambda_goals))

    @staticmethod
    def _player_impact_diagnostics(
        impact: TeamStrengthImpact | None,
    ) -> dict[str, object]:
        if impact is None:
            return {
                "data_available": False,
                "team_strength_ratio": 1.0,
                "xg_multiplier": 1.0,
                "critical_missing_count": 0,
            }
        return {
            "data_available": impact.data_available,
            "team_strength_ratio": impact.team_strength_ratio,
            "xg_multiplier": impact.xg_multiplier,
            "critical_missing_count": impact.critical_missing_count,
            "critical_missing_player_ids": list(impact.critical_missing_player_ids),
        }

    @staticmethod
    def _home_advantage_multiplier(team: dict, opponent: dict) -> float:
        home_gf = float(team.get("goals_for_avg") or 0)
        away_ga = float(opponent.get("goals_against_avg") or 0)
        if home_gf > 0 and away_ga > 0:
            ratio = home_gf / max(settings.HOME_ADVANTAGE_OPPONENT_GOALS_FLOOR, away_ga)
            return max(
                settings.HOME_ADVANTAGE_MIN_MULTIPLIER,
                min(settings.HOME_ADVANTAGE_MAX_MULTIPLIER, ratio),
            )
        form_boost = max(
            0.0,
            (float(team.get("form", 50)) - 50.0) / settings.HOME_FORM_BOOST_DIVISOR,
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
        confidence_tier = str(analysis.get("confidence_tier", "DUSUK")).upper()
        confidence_label = {
            "YUKSEK": "yüksek",
            "ORTA": "orta",
            "DUSUK": "düşük",
        }.get(confidence_tier, "bilinmiyor")
        confidence_gap = analysis.get("confidence_gap", 0)
        insights.append(
            f"{prediction_label} olasılığı %{probability}; güven seviyesi {confidence_label} "
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
                f"En olası skor {expected_score['label']} "
                f"(%{expected_score.get('probability', 0)})."
            )

        match_profile = analysis.get("match_profile") or {}
        if match_profile.get("summary"):
            insights.append(match_profile["summary"])

        best_pick = value_data.get("best_pick") if value_data else None
        if best_pick:
            insights.append(
                f"Değer sinyali: {best_pick.get('label')} için avantaj "
                f"%{best_pick.get('edge')} ve Kelly önerisi "
                f"%{best_pick.get('kelly_stake_pct', 0)}."
            )
        elif value_data:
            insights.append(
                f"Değer sinyali zayıf; ana avantaj "
                f"%{value_data.get('edge', 0)} seviyesinde."
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
