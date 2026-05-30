from celery import Celery

from app.config import settings

celery_app = Celery(
    "ct_seg",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.inference_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)
