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
