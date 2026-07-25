import datetime
from uuid import UUID
from pydantic import BaseModel


class AgentRunCreate(BaseModel):
    document_ids: list[UUID]
    agent: str = "crewai"
    question: str
    llm_provider_id: str
    llm_model_id: str


class AgentRunItemOut(BaseModel):
    id: UUID
    document_id: UUID
    status: str
    error: str | None = None
    markdown_report: str | None = None
    transactions: list[dict] | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    model_config = {"from_attributes": True}


class AgentRunOut(BaseModel):
    id: UUID
    agent: str
    question: str
    status: str
    llm_provider: str
    llm_model: str
    created_at: datetime.datetime
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    items: list[AgentRunItemOut]
    model_config = {"from_attributes": True}


class AgentRunListResponse(BaseModel):
    items: list[AgentRunOut]
    total: int


class RetryResponse(BaseModel):
    retried_item_ids: list[UUID]
