"""Cashflow API — summary, categories, trends, forecast, transactions.

All endpoints filter by owner_id (from JWT), date range, and currency.
"""
import logging
from datetime import date, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, case, extract
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.deps import get_db, get_current_user
from app.models import User
from app.models.transaction import Transaction, LineItem
from app.schemas.cashflow import (
    CashFlowCategories,
    CashFlowForecast,
    CashFlowSummary,
    CashFlowTrends,
    CategoryAmount,
    LineItemOut,
    MonthlyTrend,
    TransactionList,
    TransactionOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cashflow", tags=["cashflow"])

# Default date range: 1 year back
_DEFAULT_LOOKBACK_DAYS = 365


def _default_date_from() -> date:
    return date.today() - timedelta(days=_DEFAULT_LOOKBACK_DAYS)


def _default_date_to() -> date:
    return date.today()


# ---------------------------------------------------------------------------
# GET /api/cashflow/summary
# ---------------------------------------------------------------------------

@router.get("/summary", response_model=CashFlowSummary)
async def cashflow_summary(
    date_from: date = Query(default=None),
    date_to: date = Query(default=None),
    currency: str = Query(default="HKD"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Overall income, expenses, net, and top expense categories."""
    d_from = date_from or _default_date_from()
    d_to = date_to or _default_date_to()

    base_where = [
        Transaction.owner_id == current_user.id,
        Transaction.date >= d_from,
        Transaction.date <= d_to,
        Transaction.currency == currency,
    ]

    # Aggregate income / expenses
    agg = await db.execute(
        select(
            func.coalesce(func.sum(case((Transaction.amount > 0, Transaction.amount), else_=Decimal("0"))), Decimal("0")).label("income"),
            func.coalesce(func.sum(case((Transaction.amount < 0, func.abs(Transaction.amount)), else_=Decimal("0"))), Decimal("0")).label("expenses"),
            func.count(Transaction.id).label("tx_count"),
        ).where(*base_where)
    )
    row = agg.one()
    total_income = row.income
    total_expenses = row.expenses
    net = total_income - total_expenses

    # Top 5 expense categories
    cat_q = await db.execute(
        select(
            Transaction.category,
            func.sum(func.abs(Transaction.amount)).label("total"),
            func.count(Transaction.id).label("cnt"),
        )
        .where(*base_where, Transaction.amount < 0)
        .group_by(Transaction.category)
        .order_by(func.sum(func.abs(Transaction.amount)).desc())
        .limit(5)
    )
    top_cats = []
    for cat_row in cat_q.all():
        pct = (cat_row.total / total_expenses * 100) if total_expenses > 0 else Decimal("0")
        top_cats.append(CategoryAmount(
            category=cat_row.category,
            amount=cat_row.total,
            percentage=pct.quantize(Decimal("0.1")),
            transaction_count=cat_row.cnt,
        ))

    return CashFlowSummary(
        date_from=d_from,
        date_to=d_to,
        currency=currency,
        total_income=total_income,
        total_expenses=total_expenses,
        net=net,
        transaction_count=row.tx_count,
        top_expense_categories=top_cats,
    )


# ---------------------------------------------------------------------------
# GET /api/cashflow/categories
# ---------------------------------------------------------------------------

@router.get("/categories", response_model=CashFlowCategories)
async def cashflow_categories(
    date_from: date = Query(default=None),
    date_to: date = Query(default=None),
    currency: str = Query(default="HKD"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Breakdown of spending by category."""
    d_from = date_from or _default_date_from()
    d_to = date_to or _default_date_to()

    base_where = [
        Transaction.owner_id == current_user.id,
        Transaction.date >= d_from,
        Transaction.date <= d_to,
        Transaction.currency == currency,
    ]

    # Total expenses for percentage calculation
    total_q = await db.execute(
        select(func.coalesce(func.sum(func.abs(Transaction.amount)), Decimal("0")))
        .where(*base_where, Transaction.amount < 0)
    )
    total_expenses = total_q.scalar() or Decimal("0")

    cat_q = await db.execute(
        select(
            Transaction.category,
            func.sum(func.abs(Transaction.amount)).label("total"),
            func.count(Transaction.id).label("cnt"),
        )
        .where(*base_where, Transaction.amount < 0)
        .group_by(Transaction.category)
        .order_by(func.sum(func.abs(Transaction.amount)).desc())
    )

    categories = []
    for row in cat_q.all():
        pct = (row.total / total_expenses * 100) if total_expenses > 0 else Decimal("0")
        categories.append(CategoryAmount(
            category=row.category,
            amount=row.total,
            percentage=pct.quantize(Decimal("0.1")),
            transaction_count=row.cnt,
        ))

    return CashFlowCategories(
        date_from=d_from,
        date_to=d_to,
        currency=currency,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# GET /api/cashflow/trends
# ---------------------------------------------------------------------------

@router.get("/trends", response_model=CashFlowTrends)
async def cashflow_trends(
    date_from: date = Query(default=None),
    date_to: date = Query(default=None),
    currency: str = Query(default="HKD"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Monthly income vs expenses time series."""
    d_from = date_from or _default_date_from()
    d_to = date_to or _default_date_to()

    base_where = [
        Transaction.owner_id == current_user.id,
        Transaction.date >= d_from,
        Transaction.date <= d_to,
        Transaction.currency == currency,
    ]

    year_col = extract("year", Transaction.date).label("yr")
    month_col = extract("month", Transaction.date).label("mo")

    q = await db.execute(
        select(
            year_col,
            month_col,
            func.coalesce(func.sum(case((Transaction.amount > 0, Transaction.amount), else_=Decimal("0"))), Decimal("0")).label("income"),
            func.coalesce(func.sum(case((Transaction.amount < 0, func.abs(Transaction.amount)), else_=Decimal("0"))), Decimal("0")).label("expenses"),
        )
        .where(*base_where)
        .group_by(year_col, month_col)
        .order_by(year_col, month_col)
    )

    months = []
    for row in q.all():
        month_str = f"{int(row.yr):04d}-{int(row.mo):02d}"
        income = row.income
        expenses = row.expenses
        months.append(MonthlyTrend(
            month=month_str,
            income=income,
            expenses=expenses,
            net=income - expenses,
        ))

    return CashFlowTrends(
        date_from=d_from,
        date_to=d_to,
        currency=currency,
        months=months,
    )


# ---------------------------------------------------------------------------
# GET /api/cashflow/forecast
# ---------------------------------------------------------------------------

@router.get("/forecast", response_model=CashFlowForecast)
async def cashflow_forecast(
    horizon_months: int = Query(default=6, ge=1, le=24),
    currency: str = Query(default="HKD"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Prophet-based forecast of future income/expenses."""
    # Get all historical monthly data for this user
    year_col = extract("year", Transaction.date).label("yr")
    month_col = extract("month", Transaction.date).label("mo")

    base_where = [
        Transaction.owner_id == current_user.id,
        Transaction.currency == currency,
    ]

    q = await db.execute(
        select(
            year_col,
            month_col,
            func.coalesce(func.sum(case((Transaction.amount > 0, Transaction.amount), else_=Decimal("0"))), Decimal("0")).label("income"),
            func.coalesce(func.sum(case((Transaction.amount < 0, func.abs(Transaction.amount)), else_=Decimal("0"))), Decimal("0")).label("expenses"),
        )
        .where(*base_where)
        .group_by(year_col, month_col)
        .order_by(year_col, month_col)
    )
    rows = q.all()

    if len(rows) < 3:
        raise HTTPException(
            400,
            detail={"error": {"code": "INSUFFICIENT_DATA", "message": "Need at least 3 months of data for forecasting"}},
        )

    # Build time series
    monthly_data = []
    for row in rows:
        month_str = f"{int(row.yr):04d}-{int(row.mo):02d}"
        monthly_data.append({
            "month": month_str,
            "income": float(row.income),
            "expenses": float(row.expenses),
            "net": float(row.income - row.expenses),
        })

    from app.services.forecaster import forecast_cashflow
    result = forecast_cashflow(monthly_data, horizon_months)

    return CashFlowForecast(
        currency=currency,
        horizon_months=horizon_months,
        income=result["income"],
        expenses=result["expenses"],
        net=result["net"],
    )


# ---------------------------------------------------------------------------
# GET /api/cashflow/transactions
# ---------------------------------------------------------------------------

@router.get("/transactions", response_model=TransactionList)
async def cashflow_transactions(
    date_from: date = Query(default=None),
    date_to: date = Query(default=None),
    currency: str = Query(default="HKD"),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Paginated transactions with optional category/search filters and line items."""
    d_from = date_from or _default_date_from()
    d_to = date_to or _default_date_to()

    base_where = [
        Transaction.owner_id == current_user.id,
        Transaction.date >= d_from,
        Transaction.date <= d_to,
        Transaction.currency == currency,
    ]

    if category:
        base_where.append(Transaction.category == category)

    if search:
        search_pattern = f"%{search}%"
        base_where.append(
            Transaction.description.ilike(search_pattern)
            | Transaction.merchant.ilike(search_pattern)
        )

    # Count
    count_q = await db.execute(
        select(func.count(Transaction.id)).where(*base_where)
    )
    total = count_q.scalar() or 0

    # Fetch with line items eager-loaded
    q = await db.execute(
        select(Transaction)
        .options(selectinload(Transaction.line_items))
        .where(*base_where)
        .order_by(Transaction.date.desc(), Transaction.id)
        .offset(offset)
        .limit(limit)
    )
    txs = q.scalars().all()

    items = []
    for tx in txs:
        line_items_out = [
            LineItemOut(
                id=str(li.id),
                item_name=li.item_name,
                quantity=li.quantity,
                unit_price=li.unit_price,
                total=li.total,
                sub_category=li.sub_category,
            )
            for li in tx.line_items
        ]
        items.append(TransactionOut(
            id=str(tx.id),
            date=tx.date,
            description=tx.description,
            merchant=tx.merchant,
            amount=tx.amount,
            balance=tx.balance,
            currency=tx.currency,
            category=tx.category,
            receipt_id=str(tx.receipt_id) if tx.receipt_id else None,
            match_confidence=tx.match_confidence,
            line_items=line_items_out,
        ))

    return TransactionList(items=items, total=total, limit=limit, offset=offset)
