"""Bulk-import completed league seasons into ``historical_fixtures``.

It uses the platform's canonical ``FootballDataCSVClient`` so every imported
row carries the same stable (negative, signed) team/fixture ids, normalized
statistics, bookmaker odds and ``FT`` final status as the live sync tasks.

Sources:
* football-data.co.uk CSV feeds: standard season files for the main leagues
  (e.g. ``/mmz4281/2425/E0.csv``) and rolling files for Austria, Switzerland,
  Denmark and Russia (``/new/{AUT,SWZ,DNK,RUS}.csv``).

Run from the repository root:
``python scripts/ingest_historical.py --seasons 2425 2324 2223``

By default it writes to the local SQLite database the backend app uses
(``backend/matches.db``) so it can be run on a dev machine without PostgreSQL.
Override with ``--db-url`` or the ``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
os.environ.setdefault("DATABASE_URL", "sqlite:///backend/matches.db")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from app.db.historical_repository import HistoricalFixtureRepository  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.services.football_data_csv import FootballDataCSVClient  # noqa: E402

LEAGUE_IDS: dict[str, int] = {
    "Premier_League": 39,
    "Championship": 40,
    "La_Liga": 140,
    "Serie_A": 135,
    "Serie_B": 136,
    "Bundesliga": 78,
    "Bundesliga_2": 79,
    "Ligue_1": 61,
    "Ligue_2": 62,
    "Super_Lig": 203,
    "Eredivisie": 88,
    "Liga_Portugal": 94,
    "Jupiler_Pro_League": 144,
    "Scottish_Premiership": 179,
    "Greek_Super_League": 197,
    "Russian_Premier_Liga": 235,
    "Austrian_Bundesliga": 218,
    "Swiss_Super_League": 207,
    "Danish_Superliga": 119,
}


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


def _run_async(coro):
    return asyncio.run(coro)


def fetch_fixture_rows(
    client: FootballDataCSVClient,
    league_key: str,
    seasons: list[str],
) -> tuple[int, list[dict[str, object]]]:
    """Download normalized fixture rows for one league across seasons.

    Returns ``(rows_fetched, rows)`` where every row already matches the
    ``HistoricalFixture`` column model with canonical ids and ``FT`` status.
    """
    try:
        league_id = LEAGUE_IDS[league_key]
    except KeyError as exc:
        supported = ", ".join(LEAGUE_IDS)
        raise ValueError(
            f"Unsupported league '{league_key}'. Supported leagues: {supported}"
        ) from exc

    rows: list[dict[str, object]] = []
    fetched = 0
    for season in seasons:
        season_start = season_start_year(season)
        imported = _run_async(client.get_completed_fixtures(league_id, season_start))
        season_rows = imported.fixtures
        rows.extend(season_rows)
        fetched += len(season_rows)
        print(
            f"  {league_key} {season}: {len(season_rows)} matches fetched"
            f" ({imported.skipped_rows} incomplete rows skipped)"
        )
    return fetched, rows


def ingest_league_seasons(
    client: FootballDataCSVClient,
    league_key: str,
    seasons: list[str],
) -> tuple[int, int]:
    """Fetch, (optionally) enrich and persist one league's seasons. Returns
    (rows_fetched, rows_persisted)."""
    fetched, rows = fetch_fixture_rows(client, league_key, seasons)
    with SessionLocal() as db:
        persisted = HistoricalFixtureRepository(db).upsert_many(rows)
    print(f"  {league_key}: {fetched} rows built, {persisted} persisted")
    return fetched, persisted


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
    client = FootballDataCSVClient()
    total_rows = 0
    total_persisted = 0
    for league_key in args.leagues:
        try:
            fetched, persisted = ingest_league_seasons(
                client=client,
                league_key=league_key,
                seasons=list(args.seasons),
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
