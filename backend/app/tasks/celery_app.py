from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "bet_ai_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.jobs"],
)

# Custom Celery configurations
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Istanbul",
    enable_utc=True,
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=None,
    broker_transport_options={"visibility_timeout": 3600},
    result_backend_transport_options={"retry_policy": {"timeout": 5.0}},
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    # Celery Beat schedules
    beat_schedule={
        "sync-football-data-fixtures-daily": {
            "task": "app.tasks.jobs.sync_football_data_fixtures_task",
            "schedule": 86400.0,  # 24 hours
        },
        "sync-understat-xg-daily": {
            "task": "app.tasks.jobs.sync_understat_xg_task",
            "schedule": 86400.0,
        },
        "derive-historical-xg-daily": {
            "task": "app.tasks.jobs.derive_historical_xg_task",
            "schedule": 86400.0,
        },
        "sync-current-season-primary-weekly": {
            "task": "app.tasks.jobs.sync_historical_fixtures_task",
            "schedule": 604800.0,
            # One fixture call per supported league; player calls run in a separate job.
            "kwargs": {"enrich_player_context": False},
        },
        "sync-uefa-fixtures-daily": {
            "task": "app.tasks.jobs.sync_uefa_fixtures_task",
            "schedule": 86400.0,
        },
        "sync-statsbomb-open-daily": {
            "task": "app.tasks.jobs.sync_statsbomb_open_data_task",
            "schedule": 86400.0,
        },
        "sync-open-meteo-weather-daily": {
            "task": "app.tasks.jobs.sync_open_meteo_weather_task",
            "schedule": 86400.0,
        },
        "sync-wikidata-team-locations-weekly": {
            "task": "app.tasks.jobs.sync_wikidata_team_locations_task",
            "schedule": 604800.0,
        },
        "sync-free-team-locations-weekly": {
            "task": "app.tasks.jobs.sync_free_team_locations_task",
            "schedule": 604800.0,
        },
        "sync-completed-matches-daily": {
            "task": "app.tasks.jobs.sync_completed_matches_task",
            "schedule": 86400.0,  # 24 hours
        },
        "collect-upcoming-odds": {
            "task": "app.tasks.jobs.collect_upcoming_odds_task",
            "schedule": float(settings.ODDS_COLLECTOR_RUN_INTERVAL_SECONDS),
        },
        "collect-upcoming-lineups": {
            "task": "app.tasks.jobs.collect_upcoming_lineups_task",
            "schedule": float(settings.LINEUP_COLLECTOR_RUN_INTERVAL_SECONDS),
        },
        "retrain-ml-model-weekly": {
            "task": "app.tasks.jobs.retrain_ml_model_task",
            "schedule": 604800.0,  # Weekly
        },
        "monitor-model-drift-daily": {
            "task": "app.tasks.jobs.monitor_model_drift_task",
            "schedule": 86400.0,
        },
    },
)
