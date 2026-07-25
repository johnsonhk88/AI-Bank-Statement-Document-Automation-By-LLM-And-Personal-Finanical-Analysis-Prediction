import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.registry import agent_registry
from app.api.llm import get_llm_registry
from app.deps import get_db, get_current_user
from app.models import AgentRun, AgentRunItem, Document, User
from app.schemas.agent_run import (
    AgentRunCreate,
    AgentRunItemOut,
    AgentRunListResponse,
    AgentRunOut,
    RetryResponse,
)
from app.worker.celery_app import celery_app
from app.worker.tasks import run_agent_item

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
