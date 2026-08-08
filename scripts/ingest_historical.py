"""Bulk-import completed league seasons into ``historical_fixtures``.

Sources (combined, as chosen for the platform):
* football-data.co.uk CSV per league-season for results, shots, corners, fouls
  and bookmaker odds;
* optional API-Football enrichment that resolves Football-Data rows to API
  fixture ids and backfills missing shot/corner/foul statistics (requires
  ``FOOTBALL_API_KEY``; pass ``--no-resolve`` to skip it when no key is set).

Run from the repository root:
``python scripts/ingest_historical.py --seasons 2425 2324 2223``

By default it writes to the local SQLite database the backend app uses
(``backend/matches.db``) so it can be run on a dev machine without PostgreSQL.
Override with ``--db-url`` or the ``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import UTC, datetime, time
from io import StringIO
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/matches.db")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

import pandas as pd  # noqa: E402

from app.db.historical_repository import HistoricalFixtureRepository  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from data_pipeline import FootballDataFetcher  # noqa: E402
from data_enricher import FootballDataEnricher  # noqa: E402
from fixture_resolver import FixtureResolver  # noqa: E402

_FTR_TO_RESULT = {"H": "HOME_WIN", "D": "DRAW", "A": "AWAY_WIN"}

LEAGUE_IDS: dict[str, int] = {
    "Premier_League": 39,
    "La_Liga": 140,
    "Serie_A": 135,
    "Bundesliga": 78,
    "Ligue_1": 61,
    "Super_Lig": 203,
    "Eredivisie": 88,
    "Liga_Portugal": 94,
}

_STAT_COLUMNS = (
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HTHG",
    "HTAG",
)

_ODDS_COLUMNS = (
    "B365H",
    "B365D",
    "B365A",
    "PSH",
    "PSD",
    "PSA",
)


def season_start_year(code: str) -> int:
    """Map a football-data season code (``2425`` or ``2024``) to a start year."""
    digits = "".join(character for character in str(code) if character.isdigit())
    if len(digits) != 4:
        raise ValueError(f"Unsupported season code: {code!r}")
    value = int(digits)
    if 1900 <= value <= 2100:
        return value
    first_two = int(digits[:2])
    return 1900 + first_two if first_two >= 50 else 2000 + first_two


def _stable_id(*parts: str) -> int:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big")


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def fetch_league_season(
    fetcher: FootballDataFetcher, league_key: str, season: str
) -> pd.DataFrame:
    """Download one league-season and keep every usable match/odds column."""
    try:
        league_code = fetcher.LEAGUE_MAP[league_key]
    except KeyError as exc:
        supported = ", ".join(fetcher.LEAGUE_MAP)
        raise ValueError(
            f"Unsupported league '{league_key}'. Supported leagues: {supported}"
        ) from exc
    raw_csv = fetcher.fetch_raw_csv(season, league_code)
    frame = pd.read_csv(StringIO(raw_csv))
    desired = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    desired.update(_STAT_COLUMNS)
    desired.update(_ODDS_COLUMNS)
    frame = frame.loc[
        :, [column for column in desired if column in frame.columns]
    ].copy()
    return frame.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])


def build_fixture_rows(
    dataframe: pd.DataFrame,
    *,
    league_id: int,
    season_start: int,
) -> list[dict[str, object]]:
    """Convert an enriched league-season frame into ``historical_fixtures`` rows."""
    rows: list[dict[str, object]] = []
    for _, frame_row in dataframe.iterrows():
        match_date = FixtureResolver.parse_football_data_date(frame_row["Date"])
        home_score = _optional_int(frame_row.get("FTHG"))
        away_score = _optional_int(frame_row.get("FTAG"))
        ftr = str(frame_row.get("FTR") or "").strip().upper()
        home_team = str(frame_row.get("HomeTeam") or "").strip()
        away_team = str(frame_row.get("AwayTeam") or "").strip()
        season_start_value = frame_row.get("_season_start")
        if (
            isinstance(season_start_value, (int, float))
            and not isinstance(season_start_value, bool)
            and not pd.isna(season_start_value)
        ):
            season_start = int(season_start_value)
        result = _FTR_TO_RESULT.get(ftr)
        if (
            match_date is None
            or result is None
            or home_score is None
            or away_score is None
        ):
            continue
        if not home_team or not away_team:
            continue

        fixture_id = frame_row.get("fixture_id")
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            fixture_id = _synthetic_fixture_id(
                league_id, season_start, match_date, home_team, away_team
            )

        kickoff = datetime.combine(match_date, time(15, 0), tzinfo=UTC)
        row: dict[str, object] = {
            "fixture_id": fixture_id,
            "league_id": int(league_id or 0),
            "season": int(season_start or 0),
            "kickoff": kickoff,
            "home_team_id": _synthetic_team_id(league_id, home_team),
            "away_team_id": _synthetic_team_id(league_id, away_team),
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_score,
            "away_goals": away_score,
            "half_time_home_goals": _optional_int(_cell(frame_row, "HTHG")),
            "half_time_away_goals": _optional_int(_cell(frame_row, "HTAG")),
            "home_shots": _optional_int(_cell(frame_row, "HS")),
            "away_shots": _optional_int(_cell(frame_row, "AS")),
            "home_shots_on_target": _optional_int(_cell(frame_row, "HST")),
            "away_shots_on_target": _optional_int(_cell(frame_row, "AST")),
            "home_corners": _optional_int(_cell(frame_row, "HC")),
            "away_corners": _optional_int(_cell(frame_row, "AC")),
            "home_fouls": _optional_int(_cell(frame_row, "HF")),
            "away_fouls": _optional_int(_cell(frame_row, "AF")),
            "opening_home_odd": _odds_value(frame_row, "B365H", "PSH"),
            "opening_draw_odd": _odds_value(frame_row, "B365D", "PSD"),
            "opening_away_odd": _odds_value(frame_row, "B365A", "PSA"),
            "closing_home_odd": _odds_value(frame_row, "PSH", "B365H"),
            "closing_draw_odd": _odds_value(frame_row, "PSD", "B365D"),
            "closing_away_odd": _odds_value(frame_row, "PSA", "B365A"),
            "actual_result": result,
            "status": "completed",
            "data_source": "football_data_csv",
        }
        rows.append(row)
    return rows


def _cell(frame_row: pd.Series, column: str) -> object:
    return frame_row.get(column) if column in frame_row.index else None


def _odds_value(frame_row: pd.Series, primary: str, fallback: str) -> float | None:
    for column in (primary, fallback):
        if column in frame_row.index:
            value = _optional_float(frame_row[column])
            if value is not None:
                return value
    return None


def _synthetic_team_id(league_id: int, team_name: str) -> int:
    return _stable_id(str(league_id), "team", team_name.strip().lower())


def _synthetic_fixture_id(
    league_id: int, season_start: int, match_date, home: str, away: str
) -> int:
    return _stable_id(
        str(league_id),
        "fixture",
        str(season_start),
        match_date.isoformat(),
        home.strip().lower(),
        away.strip().lower(),
    )


def ingest_league_seasons(
    *,
    fetcher: FootballDataFetcher,
    league_key: str,
    seasons: list[str],
    resolve_ids: bool,
) -> tuple[int, int]:
    """Fetch, (optionally) enrich and persist one league's seasons. Returns
    (rows_fetched, rows_persisted)."""
    league_id = LEAGUE_IDS[league_key]
    frames: list[pd.DataFrame] = []
    for season in seasons:
        frame = fetch_league_season(fetcher, league_key, season)
        frame["_season_start"] = season_start_year(season)
        frames.append(frame)
        print(f"  {league_key} {season}: {len(frame)} matches fetched")
    combined = pd.concat(frames, ignore_index=True)

    if resolve_ids:
        api_key = (os.getenv("FOOTBALL_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "FOOTBALL_API_KEY is required for id resolution/enrichment; "
                "pass --no-resolve to skip it"
            )
        vector = FixtureResolver().resolve_dataframe(combined)
        combined = FootballDataEnricher().enrich_missing_statistics(vector)

    rows = build_fixture_rows(
        combined,
        league_id=league_id,
        season_start=0,
    )
    with SessionLocal() as db:
        persisted = HistoricalFixtureRepository(db).upsert_many(rows)
    print(f"  {league_key}: {len(rows)} rows built, {persisted} persisted")
    return len(combined), persisted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import historical seasons.")
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(LEAGUE_IDS),
        default=sorted(LEAGUE_IDS),
        help="Which supported leagues to import.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2425", "2324", "2223"],
        help="football-data season codes, e.g. 2425.",
    )
    parser.add_argument(
        "--no-resolve",
        action="store_true",
        help="Skip API-Football fixture-id resolution and statistics enrichment.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Override the SQLAlchemy database URL for this run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url
    Base.metadata.create_all(engine)
    resolve_ids = not args.no_resolve
    if resolve_ids and not (os.getenv("FOOTBALL_API_KEY") or "").strip():
        print(
            "FOOTBALL_API_KEY yok; API-Football zenginleştirmesi atlanıyor (CSV tabanı import ediliyor)."
        )
        resolve_ids = False
    fetcher = FootballDataFetcher()
    total_rows = 0
    total_persisted = 0
    for league_key in args.leagues:
        try:
            fetched, persisted = ingest_league_seasons(
                fetcher=fetcher,
                league_key=league_key,
                seasons=list(args.seasons),
                resolve_ids=resolve_ids,
            )
        except Exception as exc:
            print(f"  {league_key}: failed ({type(exc).__name__}: {exc})")
            continue
        total_rows += fetched
        total_persisted += persisted
    print(
        f"Done. fetched={total_rows} persisted={total_persisted} "
        f"(db={engine.url.render_as_string(hide_password=True)})"
    )


if __name__ == "__main__":
    main()
