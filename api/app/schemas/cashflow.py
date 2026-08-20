"""Pydantic response schemas for cashflow API endpoints."""
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class CashFlowSummary(BaseModel):
    date_from: date
    date_to: date
    currency: str
    total_income: Decimal
    total_expenses: Decimal
    net: Decimal
    transaction_count: int
    top_expense_categories: list["CategoryAmount"]


class CategoryAmount(BaseModel):
    category: str
    amount: Decimal
    percentage: Decimal
    transaction_count: int


class CashFlowCategories(BaseModel):
    date_from: date
    date_to: date
    currency: str
    categories: list[CategoryAmount]


class MonthlyTrend(BaseModel):
    month: str  # "2024-01"
    income: Decimal
    expenses: Decimal
    net: Decimal


class CashFlowTrends(BaseModel):
    date_from: date
    date_to: date
    currency: str
    months: list[MonthlyTrend]


class ForecastPoint(BaseModel):
    month: str  # "2026-09"
    predicted: Decimal
    lower: Decimal
    upper: Decimal


class CashFlowForecast(BaseModel):
    currency: str
    horizon_months: int
    income: list[ForecastPoint]
    expenses: list[ForecastPoint]
    net: list[ForecastPoint]


class LineItemOut(BaseModel):
    id: str
    item_name: str
    quantity: Decimal
    unit_price: Decimal
    total: Decimal
    sub_category: str


class TransactionOut(BaseModel):
    id: str
    date: date
    description: str
    merchant: str
    amount: Decimal
    balance: Decimal | None
    currency: str
    category: str
    receipt_id: str | None
    match_confidence: Decimal | None
    line_items: list[LineItemOut] = Field(default_factory=list)


class TransactionList(BaseModel):
    items: list[TransactionOut]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Savings-capacity signals (Phase 1 — TA integration)
# ---------------------------------------------------------------------------


class SignalMonthDetail(BaseModel):
    month: str
    net: float
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    rsi: float | None


SignalValue = Literal["INVEST", "HOLD", "ALERT"]
BBPosition = Literal[
    "above_upper", "upper_half", "lower_half", "below_lower", "insufficient_data"
]


class BollingerPosition(BaseModel):
    pband: float | None
    position: BBPosition


class SavingsCapacityResponse(BaseModel):
    currency: str
    current_signal: SignalValue
    rsi: float | None
    bollinger: BollingerPosition
    recommendation: str
    months: list[SignalMonthDetail]


# ---------------------------------------------------------------------------
# Spending-trend alerts (Phase 2 — MACD + SMA crossover)
# ---------------------------------------------------------------------------

SpendingTrend = Literal["accelerating", "decelerating", "stable"]
SMACrossover = Literal["golden_cross", "death_cross", "none"]


class SpendingMonthDetail(BaseModel):
    month: str
    amount: float
    macd: float | None
    signal: float | None
    short_sma: float | None
    long_sma: float | None


class CategorySpendingAlert(BaseModel):
    category: str
    trend: SpendingTrend
    alert: bool
    macd_histogram: float | None
    sma_crossover: SMACrossover
    recommendation: str
    months: list[SpendingMonthDetail]


class SpendingAlertsResponse(BaseModel):
    currency: str
    categories: list[CategorySpendingAlert]
    alert_count: int
    total_categories: int
