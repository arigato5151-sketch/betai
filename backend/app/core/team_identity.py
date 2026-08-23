from __future__ import annotations

import unicodedata

_LETTER_FOLD = str.maketrans(
    {
        "æ": "ae",
        "đ": "d",
        "ð": "d",
        "ı": "i",
        "ł": "l",
        "ø": "o",
        "œ": "oe",
        "ß": "ss",
        "þ": "th",
    }
)

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
    "bayer 04 leverkusen": "leverkusen",
    "birmingham city": "birmingham",
    "blackburn rovers": "blackburn",
    "bolton wanderers": "bolton",
    "besiktas jk": "besiktas",
    "besiktas istanbul": "besiktas",
    "borussia dortmund": "dortmund",
    "borussia m gladbach": "m gladbach",
    "borussia monchengladbach": "m gladbach",
    "buyuksehyr": "basaksehir",
    "cd numancia de soria": "numancia",
    "celta vigo": "celta",
    "cardiff city": "cardiff",
    "charlton athletic": "charlton",
    "clermont foot": "clermont",
    "dsc arminia bielefeld": "arminia bielefeld",
    "dijon fco": "dijon",
    "corum fk": "corum",
    "derby county": "derby",
    "dinamo moscow": "dynamo moscow",
    "eintracht frankfurt": "frankfurt",
    "ein frankfurt": "frankfurt",
    "espanyol": "espanol",
    "estoril praia": "estoril",
    "erzurumspor fk": "erzurumspor",
    "fc cologne": "fc koln",
    "fc copenhagen": "copenhagen",
    "fc heidenheim": "heidenheim",
    "fc krasnodar": "krasnodar",
    "fc orenburg": "orenburg",
    "fc nantes": "nantes",
    "fc sochaux montbeliard": "sochaux",
    "fk akhmat": "akhmat grozny",
    "fsv mainz 05": "mainz",
    "fsv mainz": "mainz",
    "genclerbirligi s k": "genclerbirligi",
    "hamburger sv": "hamburg",
    "hellas verona": "verona",
    "hertha berlin": "hertha",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "krylya sovetov samara": "krylya sovetov",
    "leicester city": "leicester",
    "levante ud": "levante",
    "lincoln city": "lincoln",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "mainz 05": "mainz",
    "hull city": "hull",
    "newcastle": "newcastle united",
    "nizhny novgorod": "pari nn",
    "norwich city": "norwich",
    "nott m forest": "nottingham forest",
    "olympique de marseille": "marseille",
    "olympique lyonnais": "lyon",
    "olympiakos piraeus": "olympiakos",
    "paok saloniki": "paok",
    "paris sg": "paris saint germain",
    "parma calcio 1913": "parma",
    "pfc sochi": "sochi",
    "psv eindhoven": "psv",
    "qpr": "queens park rangers",
    "rodez aveyron football": "rodez",
    "rc strasbourg alsace": "strasbourg",
    "rio ave fc": "rio ave",
    "rasenballsport leipzig": "rb leipzig",
    "rayo vallecano": "vallecano",
    "real betis": "betis",
    "real oviedo": "oviedo",
    "real sociedad": "sociedad",
    "fortuna sittard": "for sittard",
    "pec zwolle": "zwolle",
    "preston north end": "preston",
    "sheffield weds": "sheffield wednesday",
    "sheffield utd": "sheffield united",
    "sp braga": "braga",
    "sp lisbon": "sporting lisbon",
    "sporting braga": "braga",
    "sporting cp": "sporting lisbon",
    "stade brestois": "brest",
    "stade de reims": "reims",
    "stade lavallois mfc": "laval",
    "stade malherbe caen": "caen",
    "stoke city": "stoke",
    "sv 07 elversberg": "elversberg",
    "sv elversberg": "elversberg",
    "swansea city": "swansea",
    "us boulogne co": "boulogne",
    "usl dunkerque": "dunkerque",
    "tsg hoffenheim": "hoffenheim",
    "en avant guingamp": "guingamp",
    "as nancy lorraine": "nancy",
    "montpellier herault sc": "montpellier",
    "n e c nijmegen": "nec nijmegen",
    "excelsior rotterdam": "excelsior",
    "grenoble foot": "grenoble",
    "as saint etienne": "saint etienne",
    "aj auxerre": "auxerre",
    "rc lens": "lens",
    "ogc nice": "nice",
    "spurs": "tottenham",
    "caykur rizespor": "rizespor",
    "az": "az alkmaar",
    "dundee utd": "dundee united",
    "academico viseu": "academico",
    "maritimo m": "maritimo",
    "rcd espanyol de barcelona": "espanol",
    "tottenham hotspur": "tottenham",
    "toulouse fc": "toulouse",
    "vfb stuttgart": "stuttgart",
    "vitoria sc": "guimaraes",
    "west ham united": "west ham",
    "west brom": "west bromwich albion",
    "wolves": "wolverhampton wanderers",
    "zenit st petersburg": "zenit",
    # Verified cross-provider variants present in the historical fixture store.
    "angers sco": "angers",
    "atletico de madrid": "atletico madrid",
    "ca osasuna": "osasuna",
    "casa pia ac": "casa pia",
    "cd nacional": "nacional",
    "estac troyes": "troyes",
    "fc arouca": "arouca",
    "fc barcelona": "barcelona",
    "fc basel": "basel",
    "fc famalicao": "famalicao",
    "fc groningen": "groningen",
    "fc luzern": "luzern",
    "fc porto": "porto",
    "fc zurich": "zurich",
    "heart of midlothian": "hearts",
    "havre athletic club": "le havre",
    "istanbul basaksehir": "basaksehir",
    "kv mechelen": "mechelen",
    "lokomotiv": "lokomotiv moscow",
    "losc lille": "lille",
    "nec nijmegen": "nijmegen",
    "sc cambuur": "cambuur",
    "sk beveren": "beveren",
    "stade rennais fc": "rennes",
    "valencia cf": "valencia",
    "villarreal cf": "villarreal",
    "zulte waregem": "waregem",
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
    decomposed = unicodedata.normalize("NFKD", value.casefold().translate(_LETTER_FOLD))
    ascii_value = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(
        "".join(character if character.isalnum() else " " for character in ascii_value)
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
    normalized = " ".join(_TEAM_TOKEN_ALIASES.get(token, token) for token in tokens)
    # Removing provider-specific years/prefixes can reveal another known alias.
    return _TEAM_ALIASES.get(normalized, normalized)


def team_name_variants(value: str) -> set[str]:
    """Return known provider spellings for a canonical team identity."""
    canonical = normalize_team_name(value)
    if not canonical:
        return set()
    return {
        candidate
        for candidate in {value.casefold(), canonical, *_TEAM_ALIASES}
        if normalize_team_name(candidate) == canonical
    }


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
