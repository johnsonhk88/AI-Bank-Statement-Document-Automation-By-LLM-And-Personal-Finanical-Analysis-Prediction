import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import AsyncSessionLocal
from app.models.agent_run import AgentRun, AgentRunItem
from app.models.document import Document
from app.agents.registry import agent_registry


async def _refresh_run_status(db: AsyncSession, run_id: UUID) -> None:
    run_result = await db.execute(select(AgentRun).where(AgentRun.id == run_id))
    run = run_result.scalar_one_or_none()
    if not run:
        return

    items_result = await db.execute(select(AgentRunItem).where(AgentRunItem.run_id == run_id))
    items = items_result.scalars().all()

    statuses = [i.status for i in items]
    if all(s == "pending" for s in statuses):
        run.status = "pending"
    elif any(s == "running" for s in statuses):
        run.status = "running"
        if run.started_at is None:
            run.started_at = datetime.datetime.now(datetime.UTC)
    elif all(s == "succeeded" for s in statuses):
        run.status = "succeeded"
    elif all(s == "failed" for s in statuses):
        run.status = "failed"
    else:
        run.status = "partial"

    if run.status in ("succeeded", "failed", "partial") and run.finished_at is None:
        run.finished_at = datetime.datetime.now(datetime.UTC)

    await db.commit()


async def run_item_async(item_id: UUID) -> None:
    async with AsyncSessionLocal() as db:
        item_result = await db.execute(select(AgentRunItem).where(AgentRunItem.id == item_id))
        item = item_result.scalar_one_or_none()
        if not item:
            raise ValueError(f"AgentRunItem not found: {item_id}")

        run_result = await db.execute(select(AgentRun).where(AgentRun.id == item.run_id))
        run = run_result.scalar_one_or_none()
        if not run:
            raise ValueError(f"AgentRun not found: {item.run_id}")

        doc_result = await db.execute(select(Document).where(Document.id == item.document_id))
        doc = doc_result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document not found: {item.document_id}")

        item.status = "running"
        item.started_at = datetime.datetime.now(datetime.UTC)
        await db.commit()

        full_path = settings.UPLOAD_ROOT.parent / doc.storage_path

        adapter = agent_registry.get(run.agent)

        try:
            result = await adapter.run(
                pdf_path=str(full_path),
                question=run.question,
                llm_provider_id=run.llm_provider,
                llm_model_id=run.llm_model,
                agent_run_item_id=item.id,
            )
            item.status = "succeeded"
            item.markdown_report = result.markdown_report
            item.transactions = [t.model_dump() for t in result.transactions]
        except Exception as exc:
            item.status = "failed"
            item.error = str(exc)[:4000]

        item.finished_at = datetime.datetime.now(datetime.UTC)
        await db.commit()

        await _refresh_run_status(db, run.id)
