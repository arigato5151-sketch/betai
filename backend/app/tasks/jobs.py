"""Thin re-export layer for backward compatibility.

All task definitions live in domain-specific modules:
  - _helpers.py       — shared utilities
  - fixtures_sync.py  — fixture history sync tasks
  - predictions.py    — prediction, odds, lineup collection tasks
  - enrichment.py     — xG, weather, team location enrichment
  - ml_tasks.py       — model retraining and drift monitoring
  - results.py        — completed match sync and verification

This module re-exports every public and private task symbol so that existing
imports like ``from app.tasks.jobs import sync_missing_fixture_history_task``
and ``from app.tasks import jobs; jobs.settings`` continue to work.
"""

from app.core.config import settings  # noqa: F401
from app.db.session import SessionLocal  # noqa: F401
from app.prediction.ml.model import ml_pipeline  # noqa: F401
from app.prediction.ml.training_data import HistoricalTrainingDataBuilder  # noqa: F401
from app.prediction.ensemble_weights import ensemble_weight_manager  # noqa: F401
from app.services.api_provider_health import api_football_health  # noqa: F401
from app.services.cache import cache  # noqa: F401
from app.services.model_monitoring import ModelMonitoringService  # noqa: F401
from app.services.task_lock import DistributedTaskLock  # noqa: F401
from app.core.allowed_leagues import ALLOWED_LEAGUE_IDS  # noqa: F401
from app.services.api_football import APIFootballClient  # noqa: F401
from app.services.football_data_csv import FootballDataCSVClient  # noqa: F401
from app.providers.understat import UnderstatClient  # noqa: F401

from app.tasks._helpers import (  # noqa: F401
    _current_football_season,
    _fixture_kickoff,
    _missing_fixture_history_scopes,
    _missing_fixture_team_targets,
    _missing_api_team_targets,
    _run_async,
)
from app.tasks.fixtures_sync import (  # noqa: F401
    sync_missing_fixture_history_task,
    sync_historical_fixtures_task,
    sync_football_data_fixtures_task,
    sync_openfootball_fixtures_task,
    sync_uefa_fixtures_task,
    sync_statsbomb_open_data_task,
    _sync_historical_fixtures_impl,
)
from app.tasks.predictions import (  # noqa: F401
    generate_upcoming_predictions_task,
    collect_upcoming_odds_task,
    sync_cloudflare_odds_snapshots_task,
    collect_upcoming_lineups_task,
    _enrich_historical_player_context,
    _generate_upcoming_predictions,
    _collect_upcoming_odds,
    _collect_upcoming_lineups,
)
from app.tasks.enrichment import (  # noqa: F401
    sync_understat_xg_task,
    derive_historical_xg_task,
    sync_wikidata_team_locations_task,
    sync_free_team_locations_task,
    sync_open_meteo_weather_task,
)
from app.tasks.ml_tasks import (  # noqa: F401
    retrain_ml_model_task,
    monitor_model_drift_task,
)
from app.tasks.results import (  # noqa: F401
    sync_completed_matches_task,
    _sync_completed_matches,
)
