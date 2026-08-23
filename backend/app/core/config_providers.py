"""External data provider configuration fields."""


from typing import Literal

from pydantic import Field


class ProviderFields:
    """API Football, odds providers, Sportmonks, weather, and data source settings."""

    API_FOOTBALL_KEY: str = "DEMO_KEY"
    API_FOOTBALL_PLAN: Literal["free", "pro", "ultra", "mega"] = "free"
    THE_ODDS_API_ENABLED: bool = False
    THE_ODDS_API_KEY: str = ""
    THE_ODDS_API_BASE_URL: str = "https://api.the-odds-api.com"
    THE_ODDS_API_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=60)
    THE_ODDS_API_CACHE_SECONDS: int = Field(default=900, ge=300, le=3600)
    ODDS_API_IO_ENABLED: bool = False
    ODDS_API_IO_KEY: str = ""
    ODDS_API_IO_BASE_URL: str = "https://api.odds-api.io/v3"
    ODDS_API_IO_BOOKMAKERS: str = "Bet365,Unibet"
    ODDS_API_IO_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=60)
    ODDS_API_IO_CACHE_SECONDS: int = Field(default=900, ge=300, le=3600)
    API_FOOTBALL_CIRCUIT_FAILURE_THRESHOLD: int = Field(default=4, ge=2, le=20)
    API_FOOTBALL_CIRCUIT_OPEN_SECONDS: int = Field(default=60, ge=10, le=900)
    API_FOOTBALL_MAX_RETRY_AFTER_SECONDS: int = Field(default=120, ge=10, le=900)
    API_FOOTBALL_BACKOFF_JITTER_RATIO: float = Field(default=0.2, ge=0, le=1)
    API_FOOTBALL_MINUTE_LIMIT: int = Field(default=10, ge=1, le=1000)
    API_FOOTBALL_HISTORICAL_SYNC_ENABLED: bool = False
    CLOUDFLARE_ODDS_COLLECTOR_ENABLED: bool = False
    CLOUDFLARE_ODDS_COLLECTOR_URL: str = ""
    CLOUDFLARE_ODDS_COLLECTOR_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=60)
    CLOUDFLARE_ODDS_SYNC_INTERVAL_SECONDS: int = Field(default=900, ge=300, le=86400)
    FOOTBALL_DATA_BASE_URL: str = "https://www.football-data.co.uk"
    FOOTBALL_DATA_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    OPENFOOTBALL_ENABLED: bool = True
    OPENFOOTBALL_BASE_URL: str = (
        "https://raw.githubusercontent.com/openfootball/football.json/master"
    )
    OPENFOOTBALL_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    FOOTBALL_DATA_ORG_ENABLED: bool = False
    FOOTBALL_DATA_ORG_API_KEY: str = ""
    FOOTBALL_DATA_ORG_BASE_URL: str = "https://api.football-data.org/v4"
    THESPORTSDB_ENABLED: bool = True
    THESPORTSDB_BASE_URL: str = "https://www.thesportsdb.com/api/v1/json/123"
    MULTI_FIXTURE_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=60)
    OPENLIGADB_ENABLED: bool = True
    OPENLIGADB_BASE_URL: str = "https://api.openligadb.de"
    OPENLIGADB_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=60)
    STATSBOMB_OPEN_DATA_ENABLED: bool = True
    STATSBOMB_OPEN_DATA_BASE_URL: str = (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    )
    STATSBOMB_OPEN_DATA_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=120)
    STATSBOMB_OPEN_DATA_CONCURRENCY: int = Field(default=4, ge=1, le=8)
    STATSBOMB_OPEN_DATA_MIN_SEASON: int = Field(default=2004, ge=1950, le=2100)
    STATSBOMB_OPEN_DATA_ENRICH_BATCH_SIZE: int = Field(default=20, ge=1, le=100)
    OPEN_METEO_ENABLED: bool = True
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_HISTORICAL_FORECAST_URL: str = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
    )
    OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_METEO_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=60)
    OPEN_METEO_CONCURRENCY: int = Field(default=2, ge=1, le=4)
    OPEN_METEO_BACKFILL_BATCH_SIZE: int = Field(default=100, ge=1, le=100)
    FIXTURE_DOWNLOAD_BASE_URL: str = "https://fixturedownload.com/feed/json"
    FIXTURE_DOWNLOAD_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    FIXTURE_DOWNLOAD_UPCOMING_ENABLED: bool = True
    FIXTURE_HISTORY_SCOPE_LOOKBACK_FIXTURES: int = Field(
        default=1000, ge=100, le=10000
    )
    UNDERSTAT_ENABLED: bool = False
    UNDERSTAT_BASE_URL: str = "https://understat.com"
    UNDERSTAT_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    UNDERSTAT_MATCH_TOLERANCE_HOURS: int = Field(default=48, ge=1, le=48)
    UNDERSTAT_REQUEST_INTERVAL_SECONDS: float = Field(default=1.5, ge=0.5, le=10)
    DERIVED_XG_ENABLED: bool = True
    DERIVED_XG_MIN_TRAINING_MATCHES: int = Field(default=500, ge=100, le=10000)
    DERIVED_XG_TRAINING_FIXTURE_LIMIT: int = Field(
        default=10000, ge=500, le=100000
    )
    DERIVED_XG_UPDATE_FIXTURE_LIMIT: int = Field(default=5000, ge=100, le=50000)
    DERIVED_XG_MAX_HOLDOUT_MAE: float = Field(default=0.60, gt=0, le=2)
    DERIVED_XG_MIN_BASELINE_IMPROVEMENT: float = Field(default=0.10, ge=0, le=1)
    DERIVED_XG_CONFIDENCE: float = Field(default=0.65, gt=0, lt=0.95)
    CLUBELO_ENABLED: bool = False
    CLUBELO_BASE_URL: str = "http://api.clubelo.com"
    CLUBELO_TIMEOUT_SECONDS: float = Field(default=3.0, gt=0, le=30)
    CLUBELO_FAILURE_COOLDOWN_SECONDS: int = Field(default=300, ge=30, le=3600)
    CLUBELO_CACHE_HOURS: int = Field(default=24, ge=1, le=168)
    CLUBELO_CONFIDENCE: float = Field(default=0.80, gt=0, le=1, allow_inf_nan=False)
    SPORTMONKS_ENABLED: bool = False
    SPORTMONKS_API_TOKEN: str = ""
    SPORTMONKS_BASE_URL: str = "https://api.sportmonks.com/v3/football"
    SPORTMONKS_TIMEOUT_SECONDS: float = Field(default=15.0, gt=0, le=120)
    SPORTMONKS_PLAYER_LOOKBACK_DAYS: int = Field(default=120, ge=30, le=365)
    SPORTMONKS_PLAYER_LOOKBACK_MATCHES: int = Field(default=10, ge=3, le=30)
    ODDS_SNAPSHOT_MIN_INTERVAL_SECONDS: int = Field(default=300, ge=60, le=86400)
    ODDS_SNAPSHOT_CONFIDENCE: float = Field(
        default=0.90, gt=0, le=1, allow_inf_nan=False
    )
    ODDS_COLLECTOR_ENABLED: bool = True
    ODDS_COLLECTOR_RUN_INTERVAL_SECONDS: int = Field(default=21600, ge=900, le=86400)
    ODDS_COLLECTOR_HORIZON_DAYS: int = Field(default=7, ge=1, le=14)
    ODDS_COLLECTOR_MAX_FIXTURES: int = Field(default=20, ge=1, le=200)
    ODDS_COLLECTOR_MARKET_REQUEST_BUDGET: int = Field(default=15, ge=1, le=100)
    ODDS_COLLECTOR_DAILY_QUOTA_RESERVE: int = Field(default=20, ge=0, le=1000)
    ODDS_COLLECTOR_CLOSING_WINDOW_HOURS: int = Field(default=24, ge=1, le=168)
    ODDS_COLLECTOR_CONCURRENCY: int = Field(default=2, ge=1, le=10)
    LINEUP_COLLECTOR_ENABLED: bool = True
    LINEUP_COLLECTOR_RUN_INTERVAL_SECONDS: int = Field(default=3600, ge=900, le=21600)
    LINEUP_COLLECTOR_HORIZON_DAYS: int = Field(default=2, ge=1, le=3)
    LINEUP_COLLECTOR_MAX_FIXTURES: int = Field(default=30, ge=1, le=100)
    LINEUP_COLLECTOR_WINDOW_MINUTES: int = Field(default=120, ge=30, le=360)
    LINEUP_COLLECTOR_CONCURRENCY: int = Field(default=2, ge=1, le=10)
    AUTO_PREDICTION_ENABLED: bool = True
    AUTO_PREDICTION_RUN_INTERVAL_SECONDS: int = Field(default=21600, ge=900, le=86400)
    AUTO_PREDICTION_HORIZON_DAYS: int = Field(default=7, ge=1, le=14)
    AUTO_PREDICTION_MAX_FIXTURES: int = Field(default=25, ge=1, le=200)
    AUTO_PREDICTION_MIN_LEAD_MINUTES: int = Field(default=30, ge=5, le=1440)
    AUTO_PREDICTION_CONCURRENCY: int = Field(default=2, ge=1, le=5)
    AUTO_PREDICTION_MIN_DATA_QUALITY_SCORE: float = Field(
        default=60.0, ge=0, le=100, allow_inf_nan=False
    )
    AUTO_PREDICTION_MIN_DATA_QUALITY_SCORE_WITH_MARKET: float = Field(
        default=50.0, ge=0, le=100, allow_inf_nan=False
    )
    AUTO_PREDICTION_REQUIRE_MARKET: bool = True
    AUTO_PREDICTION_REQUIRE_SUFFICIENT_HISTORY: bool = True
    FIXTURE_CONTEXT_SNAPSHOT_TTL_SECONDS: int = Field(default=300, ge=60, le=3600)
    AUTO_PREDICTION_LOCK_TTL_SECONDS: int = Field(default=3600, ge=60, le=21600)
    MODEL_TRAINING_LOCK_TTL_SECONDS: int = Field(default=21600, ge=600, le=86400)
    TIERED_RETRAIN_MIN_FIXTURES: int = Field(default=200, ge=1, le=100000)
    AUTO_TEAM_LOCATION_ENABLED: bool = True
    WIKIDATA_LOCATION_ENABLED: bool = True
    WIKIDATA_API_URL: str = "https://www.wikidata.org/w/api.php"
    WIKIDATA_TIMEOUT_SECONDS: float = Field(default=20.0, gt=0, le=120)
    WIKIDATA_LOCATION_MAX_TEAMS: int = Field(default=500, ge=1, le=2000)
    WIKIDATA_LOCATION_CONCURRENCY: int = Field(default=1, ge=1, le=4)
    WIKIDATA_REQUEST_INTERVAL_SECONDS: float = Field(default=0.25, ge=0.1, le=5)
    FREE_TEAM_LOCATION_MAX_TEAMS: int = Field(default=70, ge=1, le=90)
