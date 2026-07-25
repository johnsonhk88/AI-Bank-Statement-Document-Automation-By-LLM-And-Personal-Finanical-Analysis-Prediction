from abc import ABC, abstractmethod
from uuid import UUID
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    date: str = ""
    description: str = ""
    credit: float | None = None
    debit: float | None = None
    balance: float | None = None
    currency: str = "HKD"


class AgentResult(BaseModel):
    markdown_report: str
    transactions: list[Transaction] = Field(default_factory=list)
    raw: dict | None = None


class AgentInfo(BaseModel):
    name: str
    display_name: str
    enabled: bool
    description: str


class BaseAgentAdapter(ABC):
    name: str
    display_name: str
    enabled: bool = True
    description: str = ""

    @abstractmethod
    async def run(self, *, pdf_path: str, question: str, llm_provider_id: str, llm_model_id: str, agent_run_item_id: UUID) -> AgentResult:
        ...
