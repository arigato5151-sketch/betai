from app.core.allowed_leagues import (
    ALLOWED_LEAGUE_IDS,
    ALLOWED_LEAGUES,
    LEAGUE_PRIORITY,
)
from app.core.config import settings

UEFA_COMPETITIONS = {
    2: "UEFA Champions League",
    3: "UEFA Europa League",
    848: "UEFA Conference League",
}

NEW_DOMESTIC_LEAGUES = {
    179: ("Scottish Premiership", "Scotland"),
    218: ("Austrian Bundesliga", "Austria"),
    207: ("Swiss Super League", "Switzerland"),
    197: ("Super League 1", "Greece"),
    119: ("Superliga", "Denmark"),
}


def test_uefa_competitions_are_allowed_with_high_priority() -> None:
    leagues_by_id = {league["id"]: league for league in ALLOWED_LEAGUES}

    assert len(leagues_by_id) == len(ALLOWED_LEAGUES)
    assert set(UEFA_COMPETITIONS).issubset(ALLOWED_LEAGUE_IDS)
    assert [league["id"] for league in ALLOWED_LEAGUES[:3]] == [2, 3, 848]

    for league_id, expected_name in UEFA_COMPETITIONS.items():
        league = leagues_by_id[league_id]
        assert league["name"] == expected_name
        assert league["country"] == "Europe"
        assert league["tier"] == "Avrupa Kupası"
        assert league["dixon_coles_rho"] == settings.DEFAULT_DIXON_COLES_RHO

    assert LEAGUE_PRIORITY[2] < LEAGUE_PRIORITY[3] < LEAGUE_PRIORITY[848]


def test_uncalibrated_uefa_competitions_use_global_dixon_coles_default() -> None:
    for league_id in UEFA_COMPETITIONS:
        rho = settings.LEAGUE_DIXON_COLES_RHO.get(
            league_id,
            settings.DEFAULT_DIXON_COLES_RHO,
        )
        assert rho == settings.DEFAULT_DIXON_COLES_RHO


def test_new_domestic_leagues_are_allowed_with_safe_global_calibration() -> None:
    leagues_by_id = {league["id"]: league for league in ALLOWED_LEAGUES}

    assert set(NEW_DOMESTIC_LEAGUES).issubset(ALLOWED_LEAGUE_IDS)
    for league_id, (expected_name, expected_country) in NEW_DOMESTIC_LEAGUES.items():
        league = leagues_by_id[league_id]
        assert league["name"] == expected_name
        assert league["country"] == expected_country
        assert league["tier"] == "1. Lig"
        assert league["dixon_coles_rho"] == settings.DEFAULT_DIXON_COLES_RHO
        assert league_id not in settings.LEAGUE_DIXON_COLES_RHO
