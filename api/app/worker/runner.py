import datetime
import logging
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import AsyncSessionLocal
from app.models.agent_run import AgentRun, AgentRunItem
from app.models.document import Document
from app.agents.registry import agent_registry
from app.services.matcher import auto_match_receipt
from app.services.normalizer import normalize_agent_run_item, parse_date
from app.services.receipt_parser import parse_receipt_and_save

logger = logging.getLogger(__name__)


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


async def _post_process_item(
    db: AsyncSession,
    *,
    run: AgentRun,
    item: AgentRunItem,
    doc: Document,
) -> None:
    """Post-process a succeeded agent run item.

    For bank/CC statements: normalize transactions into the transactions table.
    For receipts/bills: extract line items and auto-match to transactions.
    """
    doc_type = getattr(doc, "doc_type", "bank_statement")

    if doc_type in ("bank_statement", "credit_card_statement") and item.transactions:
        try:
            await normalize_agent_run_item(
                db,
                owner_id=run.owner_id,
                document_id=item.document_id,
                agent_run_item_id=item.id,
                raw_transactions=item.transactions,
            )
            await db.commit()
        except Exception:
            logger.exception(
                "Failed to normalize transactions for agent_run_item %s", item.id,
            )

    elif doc_type == "receipt":
        try:
            # Extract line items from the receipt markdown report
            receipt_text = item.markdown_report or ""
            line_items = await parse_receipt_and_save(
                db,
                receipt_document_id=item.document_id,
                receipt_text=receipt_text,
            )
            await db.commit()

            # Auto-match to a credit card / bank statement transaction
            if line_items:
                receipt_total = sum(
                    (li.total for li in line_items), Decimal("0"),
                )
                # Try to get receipt date from the first transaction or today
                receipt_date = None
                if item.transactions:
                    try:
                        receipt_date = parse_date(item.transactions[0].get("date", ""))
                    except ValueError:
                        pass
                if receipt_date is None:
                    receipt_date = datetime.date.today()

                matched, linked = await auto_match_receipt(
                    db,
                    owner_id=run.owner_id,
                    receipt_document_id=item.document_id,
                    receipt_date=receipt_date,
                    receipt_total=receipt_total,
                    receipt_text=receipt_text,
                )
                await db.commit()
                if matched:
                    logger.info(
                        "Receipt %s matched to transaction %s (%d line items linked)",
                        item.document_id, matched.id, linked,
                    )
        except Exception:
            logger.exception(
                "Failed to process receipt for agent_run_item %s", item.id,
            )


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

        # Post-processing: normalize transactions or parse receipts
        if item.status == "succeeded":
            await _post_process_item(db, run=run, item=item, doc=doc)

        await _refresh_run_status(db, run.id)
