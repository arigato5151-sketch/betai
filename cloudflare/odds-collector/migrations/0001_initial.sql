PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tracked_fixtures (
    id TEXT PRIMARY KEY,
    api_fixture_id INTEGER,
    league_id INTEGER NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    kickoff TEXT NOT NULL,
    preferred_bookmaker TEXT,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'completed', 'expired', 'unresolved')),
    closing_attempts INTEGER NOT NULL DEFAULT 0 CHECK (closing_attempts >= 0),
    last_attempt_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracked_fixtures_closing_queue
    ON tracked_fixtures(status, kickoff, closing_attempts);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracked_fixture_id TEXT NOT NULL,
    api_fixture_id INTEGER NOT NULL,
    snapshot_kind TEXT NOT NULL CHECK (snapshot_kind IN ('entry', 'closing')),
    home_odd REAL NOT NULL CHECK (home_odd > 1.0),
    draw_odd REAL NOT NULL CHECK (draw_odd > 1.0),
    away_odd REAL NOT NULL CHECK (away_odd > 1.0),
    bookmaker TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'api_football_odds',
    captured_at TEXT NOT NULL,
    provider_updated_at TEXT,
    payload_sha256 TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (tracked_fixture_id) REFERENCES tracked_fixtures(id) ON DELETE CASCADE,
    UNIQUE (tracked_fixture_id, snapshot_kind)
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_cursor
    ON odds_snapshots(id, captured_at);

CREATE TABLE IF NOT EXISTS provider_fixtures (
    api_fixture_id INTEGER PRIMARY KEY,
    fixture_date TEXT NOT NULL,
    league_id INTEGER,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_team_key TEXT NOT NULL,
    away_team_key TEXT NOT NULL,
    kickoff TEXT NOT NULL,
    cached_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provider_fixtures_lookup
    ON provider_fixtures(fixture_date, home_team_key, away_team_key, kickoff);

CREATE TABLE IF NOT EXISTS provider_day_fetches (
    fixture_date TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    fixture_count INTEGER NOT NULL CHECK (fixture_count >= 0)
);

CREATE TABLE IF NOT EXISTS api_usage (
    usage_date TEXT PRIMARY KEY,
    requests_used INTEGER NOT NULL DEFAULT 0 CHECK (requests_used >= 0),
    provider_remaining INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_type TEXT NOT NULL CHECK (trigger_type IN ('scheduled', 'manual', 'registration')),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    fixtures_considered INTEGER NOT NULL DEFAULT 0,
    snapshots_recorded INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);
