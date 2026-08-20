import os

from celery import Celery
from app.config import settings

# Local LLMs are slow — 5 CrewAI agents per PDF can take 10-30 min each.
# Default: 1 hour hard limit, 50 min soft limit.
TASK_TIME_LIMIT = int(os.environ.get("TASK_TIME_LIMIT", "3600"))
TASK_SOFT_TIME_LIMIT = int(os.environ.get("TASK_SOFT_TIME_LIMIT", "3000"))

celery_app = Celery("api", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=TASK_TIME_LIMIT,
    task_soft_time_limit=TASK_SOFT_TIME_LIMIT,
    timezone="UTC",
    imports=["app.worker.tasks"],
)
