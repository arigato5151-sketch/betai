from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.core.namespaced_ids import hashed_id
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.session import SessionLocal

MANUAL_FIXTURE_OFFSET = 7_000_000_000
DATA_SOURCE = "manual_verified"


def manual_fixture_id(league_id, season, kickoff_utc, home, away):
    natural_key = f"manual:{league_id}:{season}:{kickoff_utc.isoformat()}:{home}:{away}"
    return hashed_id("manual-verified-fixture", natural_key, MANUAL_FIXTURE_OFFSET)


rows = [
    {
        "league_id": 62,
        "season": 2025,
        "kickoff": datetime(2025, 12, 5, 19, 0, tzinfo=ZoneInfo("Europe/Paris")).astimezone(UTC),
        "home_team_id": -698835494763193013,
        "away_team_id": -7421587088341072934,
        "home_team": "Bastia",
        "away_team": "Red Star",
        "home_goals": 0,
        "away_goals": 3,
        "half_time_home_goals": 0,
        "half_time_away_goals": 3,
        "actual_result": "AWAY_WIN",
        "status": "FT",
        "data_source": DATA_SOURCE,
        "home_starting_xi": None,
        "away_starting_xi": None,
    },
    {
        "league_id": 144,
        "season": 2025,
        "kickoff": datetime(2026, 5, 3, 19, 15, tzinfo=ZoneInfo("Europe/Brussels")).astimezone(UTC),
        "home_team_id": -4426724503595692663,
        "away_team_id": -9021290301448745073,
        "home_team": "Dender",
        "away_team": "RAAL La Louviere",
        "home_goals": 2,
        "away_goals": 1,
        "half_time_home_goals": None,
        "half_time_away_goals": None,
        "actual_result": "HOME_WIN",
        "status": "FT",
        "data_source": DATA_SOURCE,
        "home_starting_xi": None,
        "away_starting_xi": None,
    },
]

for row in rows:
    row["fixture_id"] = manual_fixture_id(
        row["league_id"],
        row["season"],
        row["kickoff"],
        row["home_team"],
        row["away_team"],
    )
    print(
        f"row: league={row['league_id']} {row['kickoff']} "
        f"{row['home_team']} {row['home_goals']}-{row['away_goals']} {row['away_team']} "
        f"fixture_id={row['fixture_id']}"
    )

fids = [row["fixture_id"] for row in rows]
if len(set(fids)) != len(fids):
    raise SystemExit("fixture_id collision among new rows")

with SessionLocal() as db:
    existing = set(
        r[0]
        for r in db.execute(
            text("SELECT fixture_id FROM historical_fixtures WHERE fixture_id IN :ids"),
            {"ids": tuple(fids)},
        ).all()
    )
if existing:
    raise SystemExit(f"ABORT: fixture ids already exist: {existing}")

with SessionLocal() as db:
    processed = HistoricalFixtureRepository(db).upsert_many(rows)

with SessionLocal() as db:
    counts = db.execute(
        text(
            "SELECT league_id, COUNT(*) FROM historical_fixtures"
            " WHERE season=2025 AND league_id IN (62,144) GROUP BY league_id ORDER BY league_id"
        )
    ).all()
    new_rows = db.execute(
        text(
            "SELECT league_id, kickoff, home_team, home_goals, away_goals, away_team,"
            " data_source FROM historical_fixtures WHERE fixture_id IN :ids"
        ),
        {"ids": tuple(fids)},
    ).all()

print(f"inserted={processed}")
for league_id, count in counts:
    print(f"  league={league_id} season 2025 total={count}")
for league_id, kickoff, home, hg, ag, away, src in new_rows:
    print(f"  new: {league_id} {kickoff} {home} {hg}-{ag} {away} [{src}]")