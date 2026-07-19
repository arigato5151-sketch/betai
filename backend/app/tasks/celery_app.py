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
        "sync-historical-fixtures-daily": {
            "task": "app.tasks.jobs.sync_historical_fixtures_task",
            "schedule": 86400.0,  # 24 hours
        },
        "sync-completed-matches-daily": {
            "task": "app.tasks.jobs.sync_completed_matches_task",
            "schedule": 86400.0,  # 24 hours
        },
        "retrain-ml-model-weekly": {
            "task": "app.tasks.jobs.retrain_ml_model_task",
            "schedule": 604800.0,  # Weekly
        },
    },
)
