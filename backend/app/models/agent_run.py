from __future__ import annotations
import uuid
from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base
from app.models.mixins import SoftDeleteMixin, TimestampMixin

class AgentRun(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "agent_runs"
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent: Mapped[str] = mapped_column(String(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at = mapped_column(DateTime(timezone=True), nullable=True)
    items: Mapped[list["AgentRunItem"]] = relationship("AgentRunItem", back_populates="run", cascade="all, delete-orphan")

class AgentRunItem(Base, TimestampMixin):
    __tablename__ = "agent_run_items"
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    markdown_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    transactions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at = mapped_column(DateTime(timezone=True), nullable=True)
    run: Mapped[AgentRun] = relationship("AgentRun", back_populates="items")
