from __future__ import annotations

import unicodedata

_TEAM_ALIASES = {
    "ac milan": "milan",
    "ath bilbao": "athletic bilbao",
    "athletic club": "athletic bilbao",
    "ath madrid": "atletico madrid",
    "bayer leverkusen": "leverkusen",
    "borussia dortmund": "dortmund",
    "borussia m gladbach": "m gladbach",
    "celta vigo": "celta",
    "espanyol": "espanol",
    "ein frankfurt": "eintracht frankfurt",
    "fc cologne": "fc koln",
    "fc heidenheim": "heidenheim",
    "hamburger sv": "hamburg",
    "inter": "inter milan",
    "man city": "manchester city",
    "man united": "manchester united",
    "mainz 05": "mainz",
    "newcastle": "newcastle united",
    "nott m forest": "nottingham forest",
    "parma calcio 1913": "parma",
    "paris sg": "paris saint germain",
    "rasenballsport leipzig": "rb leipzig",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real oviedo": "oviedo",
    "real sociedad": "sociedad",
    "sp lisbon": "sporting lisbon",
    "vfb stuttgart": "stuttgart",
    "wolves": "wolverhampton wanderers",
}

_TEAM_TOKEN_ALIASES = {
    "moskva": "moscow",
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
    tokens = normalized.split()
    if tokens and tokens[0] in {"fc", "fk"}:
        # API-Football and CSV feeds use FC/FK interchangeably for some clubs.
        tokens[0] = "fc"
    return " ".join(_TEAM_TOKEN_ALIASES.get(token, token) for token in tokens)
