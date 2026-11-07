import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_current_user, get_db
from app.models import AgentRun, AgentRunItem, User
from app.schemas.analytics import AnalyticsSummary
from app.services.analytics import summarize_transactions, summary_markdown

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


async def _load_rows(db: AsyncSession, owner_id) -> list[dict]:
    result = await db.execute(
        select(AgentRunItem.transactions)
        .join(AgentRun, AgentRun.id == AgentRunItem.run_id)
        .where(
            AgentRun.owner_id == owner_id,
            AgentRun.deleted_at.is_(None),
            AgentRunItem.status == "succeeded",
            AgentRunItem.transactions.isnot(None),
        )
    )
    rows: list[dict] = []
    for (payload,) in result.all():
        if isinstance(payload, list):
            rows.extend(x for x in payload if isinstance(x, dict))
        elif isinstance(payload, dict):
            inner = payload.get("transactions")
            if isinstance(inner, list):
                rows.extend(x for x in inner if isinstance(x, dict))
    return rows


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(
    year: int = Query(default_factory=lambda: datetime.date.today().year),
    quarter: int | None = Query(default=None, ge=1, le=4),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await _load_rows(db, current_user.id)
    return AnalyticsSummary(**summarize_transactions(rows, year=year, quarter=quarter))


@router.get("/export.md")
async def analytics_export_md(
    year: int = Query(default_factory=lambda: datetime.date.today().year),
    quarter: int | None = Query(default=None, ge=1, le=4),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = await _load_rows(db, current_user.id)
    summary = summarize_transactions(rows, year=year, quarter=quarter)
    body = summary_markdown(summary)
    filename = f"bankai-{year}{f'-Q{quarter}' if quarter else ''}.md"
    return PlainTextResponse(
        body,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
