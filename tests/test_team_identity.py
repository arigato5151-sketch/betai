import pytest

from app.core.team_identity import normalize_team_name, stable_team_name_key


def test_hannover_96_collapses_onto_hannover() -> None:
    assert normalize_team_name("Hannover 96") == normalize_team_name("Hannover")
    assert normalize_team_name("Schalke 04") == normalize_team_name("Schalke")
    assert normalize_team_name("Mainz 05") == normalize_team_name("Mainz")


def test_ordinal_1_prefix_is_stripped() -> None:
    assert normalize_team_name("1 FC Koln") == normalize_team_name("FC Koln")
    assert normalize_team_name("1 FK Wien") == normalize_team_name("FK Wien")


def test_leading_year_identity_is_preserved() -> None:
    assert normalize_team_name("1860 Munich") == "1860 munich"
    assert normalize_team_name("1860 Munich") != normalize_team_name("Bayern Munich")
    assert normalize_team_name("1893 Wien") == "1893 wien"


def test_existing_alias_rules_still_apply() -> None:
    assert normalize_team_name("Man City") == normalize_team_name("Manchester City")
    assert normalize_team_name("FC Rostov") == normalize_team_name("FK Rostov")


def test_stable_key_unchanged_by_year_stripping() -> None:
    assert stable_team_name_key("Hannover 96") == "hannover 96"
    assert stable_team_name_key("Hannover") == "hannover"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Bayern München", "Bayern Munich"),
        ("AEK Athen", "AEK Athens"),
        ("PSV", "PSV Eindhoven"),
        ("Sporting CP", "Sp Lisbon"),
        ("Sporting CP", "Sporting Lisbon"),
        ("Sp Braga", "Braga"),
        ("AFC Bournemouth", "Bournemouth"),
        ("AC Ajaccio", "Ajaccio"),
        ("FSV Mainz 05", "Mainz"),
        ("Eintracht Frankfurt", "Ein Frankfurt"),
        ("Eintracht Frankfurt", "Frankfurt"),
        ("Hellas Verona", "Verona"),
        ("Hertha Berlin", "Hertha"),
        ("Olympiacos", "Olympiakos Piraeus"),
        ("PAOK", "PAOK Saloniki"),
        ("Tottenham", "Tottenham Hotspur"),
        ("Leicester", "Leicester City"),
        ("B. Dortmund", "Borussia Dortmund"),
        ("B. Dortmund", "Dortmund"),
        ("Hamburger SV", "Hamburg"),
        ("Vitória SC", "Guimaraes"),
        ("PEC Zwolle", "Zwolle"),
        ("Fortuna Sittard", "For Sittard"),
        ("Clermont Foot 63", "Clermont"),
        ("Dijon FCO", "Dijon"),
        ("US Boulogne CO", "Boulogne"),
        ("USL Dunkerque", "Dunkerque"),
        ("Montpellier Hérault SC", "Montpellier"),
        ("Erzurumspor FK", "Erzurumspor"),
        ("Birmingham City", "Birmingham"),
        ("Stade Lavallois MFC", "Laval"),
        ("Kasımpaşa", "Kasimpasa"),
        ("Cardiff City", "Cardiff"),
        ("Estoril Praia", "Estoril"),
        ("Rio Ave FC", "Rio Ave"),
        ("Toulouse FC", "Toulouse"),
        ("Olympique Lyonnais", "Lyon"),
        ("FC Barcelona", "Barcelona"),
        ("Heart of Midlothian", "Hearts"),
        ("NEC Nijmegen", "Nijmegen"),
        ("FC Porto", "Porto"),
        ("Atlético de Madrid", "Atletico Madrid"),
        ("Havre Athletic Club", "Le Havre"),
        ("Stade Rennais FC", "Rennes"),
        ("Zulte Waregem", "Waregem"),
        ("SK Beveren", "Beveren"),
        ("Hull City", "Hull"),
        ("Lincoln City", "Lincoln"),
        ("Blackburn Rovers", "Blackburn"),
        ("Sheffield Utd", "Sheffield United"),
        ("Preston North End", "Preston"),
        ("Bolton Wanderers", "Bolton"),
        ("Charlton Athletic", "Charlton"),
        ("Internazionale", "Inter"),
        ("Spurs", "Tottenham"),
        ("Çaykur Rizespor", "Rizespor"),
        ("AZ", "AZ Alkmaar"),
        ("N.E.C. Nijmegen", "NEC Nijmegen"),
        ("Excelsior Rotterdam", "Excelsior"),
        ("AS Saint-Étienne", "Saint Etienne"),
        ("Grenoble Foot 38", "Grenoble"),
        ("RCD Espanyol de Barcelona", "Espanyol"),
        ("West Brom", "West Bromwich Albion"),
        ("Gençlerbirliği S.K.", "Gençlerbirligi"),
        ("Stade de Reims", "Reims"),
        ("DSC Arminia Bielefeld", "Arminia Bielefeld"),
        ("SV 07 Elversberg", "Elversberg"),
        ("Bayer 04 Leverkusen", "Leverkusen"),
        ("TSG Hoffenheim", "Hoffenheim"),
        ("1. FSV Mainz 05", "Mainz"),
        ("En Avant Guingamp", "Guingamp"),
        ("FC Sochaux-Montbéliard", "Sochaux"),
    ],
)
def test_cross_league_name_variants_collapse(left: str, right: str) -> None:
    assert normalize_team_name(left) == normalize_team_name(right)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Distinct clubs must never merge via prefix stripping.
        ("1860 Munich", "Bayern Munich"),
        ("Sporting CP", "Sporting Gijón"),
        ("Inter", "Eintracht Frankfurt"),
    ],
)
def test_distinct_clubs_remain_separate(first: str, second: str) -> None:
    assert normalize_team_name(first) != normalize_team_name(second)
