import datetime, logging, os
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.registry import agent_registry
from app.api.llm import get_llm_registry
from app.config import settings
from app.core.hashing import sha256_file
from app.core.storage import save_upload
from app.deps import get_db, get_current_user
from app.models import AgentRun, AgentRunItem, Document, User
from app.schemas.agent_run import (
    AgentRunCreate,
    AgentRunItemOut,
    AgentRunListResponse,
    AgentRunOut,
    BulkProcessResponse,
    RetryResponse,
)
from app.worker.celery_app import celery_app
from app.worker.tasks import run_agent_item

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent-runs", tags=["agent-runs"])


@router.post("", response_model=AgentRunOut, status_code=201)
async def create_agent_run(
    body: AgentRunCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not agent_registry.is_valid(body.agent):
        raise HTTPException(
            422,
            detail={"error": {"code": "VALIDATION", "message": f"Unknown or disabled agent: {body.agent}"}},
        )

    try:
        get_llm_registry().resolve(body.llm_provider_id, body.llm_model_id)
    except ValueError as e:
        raise HTTPException(
            422,
            detail={"error": {"code": "VALIDATION", "message": str(e)}},
        )

    if len(body.document_ids) < 1:
        raise HTTPException(
            422,
            detail={"error": {"code": "VALIDATION", "message": "At least one document_id is required"}},
        )

    for doc_id in body.document_ids:
        result = await db.execute(
            select(Document).where(
                Document.id == doc_id,
                Document.owner_id == current_user.id,
                Document.deleted_at.is_(None),
            )
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                404,
                detail={"error": {"code": "NOT_FOUND", "message": f"Document {doc_id} not found"}},
            )

    run = AgentRun(
        owner_id=current_user.id,
        agent=body.agent,
        question=body.question,
        llm_provider=body.llm_provider_id,
        llm_model=body.llm_model_id,
    )
    db.add(run)

    items = []
    for doc_id in body.document_ids:
        item = AgentRunItem(run_id=run.id, document_id=doc_id)
        db.add(item)
        items.append(item)

    await db.flush()

    for item in items:
        task = run_agent_item.delay(str(item.id))
        item.celery_task_id = task.id

    await db.commit()

    result = await db.execute(
        select(AgentRun)
        .options(selectinload(AgentRun.items))
        .where(AgentRun.id == run.id)
    )
    run = result.scalar_one()

    return run


@router.post("/bulk-process", response_model=BulkProcessResponse, status_code=202)
async def bulk_upload_and_process(
    files: List[UploadFile] = File(...),
    question: str = Form("Extract all transactions and provide a financial summary."),
    agent: str = Form("crewai"),
    llm_provider_id: str = Form("llamacpp"),
    llm_model_id: str = Form("openai/local-model"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload multiple bank statement PDFs and immediately process them all."""
    if not agent_registry.is_valid(agent):
        raise HTTPException(422, detail={"error": {"code": "VALIDATION", "message": f"Unknown or disabled agent: {agent}"}})

    try:
        get_llm_registry().resolve(llm_provider_id, llm_model_id)
    except ValueError as e:
        raise HTTPException(422, detail={"error": {"code": "VALIDATION", "message": str(e)}})

    doc_ids: list[UUID] = []
    skipped = 0
    errors: list[str] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            errors.append(f"{file.filename or 'unknown'}: not a PDF, skipped")
            continue
        try:
            storage_path = save_upload(file, current_user.id)
        except ValueError as exc:
            errors.append(f"{file.filename}: {exc}")
            continue

        full_path = settings.UPLOAD_ROOT.parent / storage_path
        sha256 = sha256_file(str(full_path))
        size_bytes = full_path.stat().st_size

        existing = await db.execute(
            select(Document).where(Document.content_sha256 == sha256, Document.deleted_at.is_(None))
        )
        dup = existing.scalar_one_or_none()
        if dup:
            os.remove(str(full_path))
            skipped += 1
            doc_ids.append(dup.id)
            continue

        doc = Document(
            owner_id=current_user.id,
            original_filename=file.filename,
            storage_path=storage_path,
            content_sha256=sha256,
            mime_type=file.content_type or "application/pdf",
            size_bytes=size_bytes,
        )
        db.add(doc)
        await db.flush()
        doc_ids.append(doc.id)

    if not doc_ids:
        await db.commit()
        return BulkProcessResponse(
            uploaded_count=0, skipped_duplicates=skipped, upload_errors=errors, agent_run=None,
        )

    run = AgentRun(
        owner_id=current_user.id,
        agent=agent,
        question=question,
        llm_provider=llm_provider_id,
        llm_model=llm_model_id,
    )
    db.add(run)

    items = []
    for doc_id in doc_ids:
        item = AgentRunItem(run_id=run.id, document_id=doc_id)
        db.add(item)
        items.append(item)

    await db.flush()

    for item in items:
        task = run_agent_item.delay(str(item.id))
        item.celery_task_id = task.id

    await db.commit()

    result = await db.execute(
        select(AgentRun).options(selectinload(AgentRun.items)).where(AgentRun.id == run.id)
    )
    run = result.scalar_one()

    logger.info("Bulk process: %d docs queued, %d skipped, %d errors", len(doc_ids), skipped, len(errors))

    return BulkProcessResponse(
        uploaded_count=len(doc_ids) - skipped,
        skipped_duplicates=skipped,
        upload_errors=errors,
        agent_run=AgentRunOut.model_validate(run),
    )


@router.get("", response_model=AgentRunListResponse)
async def list_agent_runs(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if limit > 100:
        limit = 100

    where = [AgentRun.owner_id == current_user.id, AgentRun.deleted_at.is_(None)]
    if status:
        where.append(AgentRun.status == status)

    count_query = select(func.count(AgentRun.id)).where(*where)
    total = (await db.execute(count_query)).scalar() or 0

    result = await db.execute(
        select(AgentRun)
        .options(selectinload(AgentRun.items))
        .where(*where)
        .order_by(AgentRun.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    runs = result.scalars().all()

    items = [AgentRunOut.model_validate(r) for r in runs]
    return AgentRunListResponse(items=items, total=total)


@router.get("/{run_id}", response_model=AgentRunOut)
async def get_agent_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(AgentRun)
        .options(selectinload(AgentRun.items))
        .where(
            AgentRun.id == run_id,
            AgentRun.owner_id == current_user.id,
            AgentRun.deleted_at.is_(None),
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(
            404,
            detail={"error": {"code": "NOT_FOUND", "message": "Agent run not found"}},
        )
    return run


@router.post("/{run_id}/retry", response_model=RetryResponse, status_code=202)
async def retry_agent_run_items(
    run_id: UUID,
    item_ids: list[UUID] | None = Query(None, alias="item_ids"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(AgentRun)
        .options(selectinload(AgentRun.items))
        .where(
            AgentRun.id == run_id,
            AgentRun.owner_id == current_user.id,
            AgentRun.deleted_at.is_(None),
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(
            404,
            detail={"error": {"code": "NOT_FOUND", "message": "Agent run not found"}},
        )

    target_ids = set(item_ids) if item_ids else None
    retried: list[UUID] = []

    now = datetime.datetime.now(datetime.UTC)
    for item in run.items:
        if target_ids is not None and item.id not in target_ids:
            continue
        if item.status == "failed":
            item.status = "pending"
            item.error = None
            item.markdown_report = None
            item.transactions = None
            item.started_at = None
            item.finished_at = None
            task = run_agent_item.delay(str(item.id))
            item.celery_task_id = task.id
            item.updated_at = now
            retried.append(item.id)

    if target_ids:
        missing = target_ids - set(item.id for item in run.items)
        if missing:
            raise HTTPException(
                404,
                detail={"error": {"code": "NOT_FOUND", "message": f"Item(s) not found: {missing}"}},
            )
        non_failed = []
        for item in run.items:
            if target_ids is not None and item.id in target_ids and item.status != "failed":
                non_failed.append(str(item.id))
        if non_failed:
            raise HTTPException(
                409,
                detail={"error": {"code": "CONFLICT", "message": f"Item(s) are not in failed status: {non_failed}"}},
            )

    await db.commit()
    return RetryResponse(retried_item_ids=retried)


@router.delete("/{run_id}", status_code=204)
async def delete_agent_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(AgentRun)
        .options(selectinload(AgentRun.items))
        .where(
            AgentRun.id == run_id,
            AgentRun.owner_id == current_user.id,
            AgentRun.deleted_at.is_(None),
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(
            404,
            detail={"error": {"code": "NOT_FOUND", "message": "Agent run not found"}},
        )

    for item in run.items:
        if item.celery_task_id and item.status in ("pending", "running"):
            celery_app.control.revoke(item.celery_task_id, terminate=True)

    run.deleted_at = datetime.datetime.now(datetime.UTC)
    await db.commit()
