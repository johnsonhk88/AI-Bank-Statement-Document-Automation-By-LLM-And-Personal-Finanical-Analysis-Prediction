import asyncio
import uuid

from app.worker.celery_app import celery_app
from app.worker.runner import run_item_async


@celery_app.task(name="agent.run_item", bind=True, max_retries=2)
def run_agent_item(self, agent_run_item_id: str) -> None:
    try:
        asyncio.run(run_item_async(uuid.UUID(agent_run_item_id)))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
