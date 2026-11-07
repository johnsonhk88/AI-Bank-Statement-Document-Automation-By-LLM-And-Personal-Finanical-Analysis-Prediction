from pydantic import BaseModel


class MonthBucket(BaseModel):
    year: int
    month: int
    credits: float
    debits: float
    net: float
    count: int


class TopItem(BaseModel):
    description: str
    credits: float
    debits: float
    count: int


class AnalyticsSummary(BaseModel):
    year: int
    quarter: int | None = None
    credits: float
    debits: float
    net: float
    count: int
    by_month: list[MonthBucket]
    top: list[TopItem]
