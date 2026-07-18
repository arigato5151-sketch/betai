DEMO_LIVE_FIXTURES = [
    {
        "fixture_id": 900001,
        "league": "Super Lig (Demo)",
        "home_team": "Fenerbahce",
        "away_team": "Galatasaray",
        "home_team_id": 611,
        "away_team_id": 645,
        "league_id": 203,
        "season": 2024,
        "minute": 67,
        "score": "2 - 1",
        "is_demo": True,
    },
    {
        "fixture_id": 900002,
        "league": "Premier League (Demo)",
        "home_team": "Arsenal",
        "away_team": "Liverpool",
        "home_team_id": 42,
        "away_team_id": 40,
        "league_id": 39,
        "season": 2024,
        "minute": 54,
        "score": "1 - 1",
        "is_demo": True,
    },
    {
        "fixture_id": 900003,
        "league": "La Liga (Demo)",
        "home_team": "Real Madrid",
        "away_team": "Barcelona",
        "home_team_id": 541,
        "away_team_id": 529,
        "league_id": 140,
        "season": 2024,
        "minute": 38,
        "score": "0 - 0",
        "is_demo": True,
    },
]


def _demo_profile(form, attack, defense, xg, gf, ga):
    atk_s = round(gf / 1.32, 3)
    def_s = round(ga / 1.32, 3)
    return {
        "form": form,
        "attack": attack,
        "defense": defense,
        "xg": xg,
        "attack_strength": atk_s,
        "defense_strength": def_s,
        "goals_for_avg": gf,
        "goals_against_avg": ga,
        "strength_rating": round((form + attack + defense) / 3, 1),
        "source": "demo_professional_profile",
        "method": "home_away_split_decay_form",
    }


DEMO_TEAM_STATS = {
    611: _demo_profile(88, 86, 82, 2.05, 1.95, 1.05),
    645: _demo_profile(84, 90, 76, 1.95, 2.05, 1.25),
    42: _demo_profile(91, 88, 85, 2.20, 2.10, 0.95),
    40: _demo_profile(87, 92, 78, 2.10, 2.15, 1.10),
    541: _demo_profile(93, 94, 88, 2.35, 2.30, 0.85),
    529: _demo_profile(89, 91, 80, 2.05, 2.05, 1.00),
}

DEMO_UPCOMING_FIXTURES = [
    {
        "fixture_id": 900010,
        "league": "Super Lig (Demo)",
        "home_team": "Besiktas",
        "away_team": "Trabzonspor",
        "home_team_id": 549,
        "away_team_id": 998,
        "league_id": 203,
        "season": 2024,
        "kickoff": "2026-05-25T19:00:00+00:00",
        "kickoff_label": "25.05 22:00",
        "status": "NS",
        "is_live": False,
        "is_demo": True,
    },
    {
        "fixture_id": 900011,
        "league": "Serie A (Demo)",
        "home_team": "Juventus",
        "away_team": "AC Milan",
        "home_team_id": 496,
        "away_team_id": 489,
        "league_id": 135,
        "season": 2024,
        "kickoff": "2026-05-25T21:45:00+00:00",
        "kickoff_label": "26.05 00:45",
        "status": "NS",
        "is_live": False,
        "is_demo": True,
    },
    {
        "fixture_id": 900012,
        "league": "Bundesliga (Demo)",
        "home_team": "Bayern Munich",
        "away_team": "Borussia Dortmund",
        "home_team_id": 157,
        "away_team_id": 165,
        "league_id": 78,
        "season": 2024,
        "kickoff": "2026-05-26T18:30:00+00:00",
        "kickoff_label": "26.05 21:30",
        "status": "NS",
        "is_live": False,
        "is_demo": True,
    },
    {
        "fixture_id": 900013,
        "league": "Ligue 1 (Demo)",
        "home_team": "PSG",
        "away_team": "Marseille",
        "home_team_id": 85,
        "away_team_id": 81,
        "league_id": 61,
        "season": 2024,
        "kickoff": "2026-05-26T20:00:00+00:00",
        "kickoff_label": "26.05 23:00",
        "status": "NS",
        "is_live": False,
        "is_demo": True,
    },
]

DEMO_FIXTURE_ODDS = {
    900001: 2.35,
    900002: 2.10,
    900003: 2.55,
    900010: 2.20,
    900011: 2.45,
    900012: 1.95,
    900013: 1.72,
}

DEMO_TEAM_STATS.update(
    {
        549: _demo_profile(76, 78, 74, 1.75, 1.70, 1.20),
        998: _demo_profile(72, 80, 70, 1.82, 1.78, 1.28),
        496: _demo_profile(82, 84, 86, 1.88, 1.72, 0.98),
        489: _demo_profile(79, 86, 80, 1.92, 1.88, 1.12),
        157: _demo_profile(88, 92, 84, 2.25, 2.35, 0.92),
        165: _demo_profile(81, 88, 76, 2.05, 2.05, 1.18),
        85: _demo_profile(90, 93, 82, 2.30, 2.28, 1.05),
        81: _demo_profile(74, 79, 72, 1.70, 1.65, 1.22),
    }
)
