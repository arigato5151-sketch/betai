from __future__ import annotations

import unicodedata

_TEAM_ALIASES = {
    "ac ajaccio": "ajaccio",
    "ac milan": "milan",
    "ad demirspor": "adana demirspor",
    "akron": "akron togliatti",
    "aris saloniki": "aris",
    "ath bilbao": "athletic bilbao",
    "athletic club": "athletic bilbao",
    "ath madrid": "atletico madrid",
    "atleti": "atletico madrid",
    "b dortmund": "dortmund",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia m gladbach": "m gladbach",
    "borussia monchengladbach": "m gladbach",
    "buyuksehyr": "basaksehir",
    "cd numancia de soria": "numancia",
    "celta vigo": "celta",
    "clermont foot": "clermont",
    "derby county": "derby",
    "dinamo moscow": "dynamo moscow",
    "eintracht frankfurt": "frankfurt",
    "ein frankfurt": "frankfurt",
    "espanyol": "espanol",
    "fc cologne": "fc koln",
    "fc copenhagen": "copenhagen",
    "fc heidenheim": "heidenheim",
    "fc krasnodar": "krasnodar",
    "fc orenburg": "orenburg",
    "fk akhmat": "akhmat grozny",
    "fsv mainz 05": "mainz",
    "hamburger sv": "hamburg",
    "hellas verona": "verona",
    "hertha berlin": "hertha",
    "inter": "inter milan",
    "krylya sovetov samara": "krylya sovetov",
    "leicester city": "leicester",
    "levante ud": "levante",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "mainz 05": "mainz",
    "newcastle": "newcastle united",
    "nizhny novgorod": "pari nn",
    "norwich city": "norwich",
    "nott m forest": "nottingham forest",
    "olympique de marseille": "marseille",
    "olympiakos piraeus": "olympiakos",
    "paok saloniki": "paok",
    "paris sg": "paris saint germain",
    "parma calcio 1913": "parma",
    "pfc sochi": "sochi",
    "psv eindhoven": "psv",
    "qpr": "queens park rangers",
    "rasenballsport leipzig": "rb leipzig",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real oviedo": "oviedo",
    "real sociedad": "sociedad",
    "fortuna sittard": "for sittard",
    "pec zwolle": "zwolle",
    "sheffield weds": "sheffield wednesday",
    "sp braga": "braga",
    "sp lisbon": "sporting lisbon",
    "sporting braga": "braga",
    "sporting cp": "sporting lisbon",
    "stade brestois": "brest",
    "stade malherbe caen": "caen",
    "stoke city": "stoke",
    "swansea city": "swansea",
    "tottenham hotspur": "tottenham",
    "vfb stuttgart": "stuttgart",
    "vitoria sc": "guimaraes",
    "west ham united": "west ham",
    "wolves": "wolverhampton wanderers",
    "zenit st petersburg": "zenit",
}

_TEAM_TOKEN_ALIASES = {
    "athen": "athens",
    "moskva": "moscow",
    "munchen": "munich",
    "olympiacos": "olympiakos",
    "praha": "prague",
}


def stable_team_name_key(value: str) -> str:
    """Return an immutable source identity key without mutable alias rules."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(
        "".join(character if character.isalnum() else " " for character in ascii_value)
        .casefold()
        .split()
    )


def normalize_team_name(value: str) -> str:
    """Return an alias-aware key for matching names across providers."""
    normalized = stable_team_name_key(value)
    normalized = _TEAM_ALIASES.get(normalized, normalized)
    tokens = _strip_club_year_tokens(normalized.split())
    if tokens and tokens[0] in {"fc", "fk"}:
        # API-Football and CSV feeds use FC/FK interchangeably for some clubs.
        tokens[0] = "fc"
    return " ".join(_TEAM_TOKEN_ALIASES.get(token, token) for token in tokens)


def _strip_club_year_tokens(tokens: list[str]) -> list[str]:
    """Drop leading prefixes and trailing founding-year digits.

    Leading years identify separate clubs (e.g. "1860 Munich") and are kept so
    they never collapse onto the senior namesake club. Leading "1", "ac" and
    "afc" prefixes are pure form prefixes ("1. FC Köln", "AC Ajaccio",
    "AFC Bournemouth") and safe to remove.
    """
    if not tokens:
        return tokens
    while tokens and tokens[0] in {"1", "ac", "afc"}:
        tokens = tokens[1:]
    while tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    return tokens
