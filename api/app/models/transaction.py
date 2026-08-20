from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import Date, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin


class Transaction(Base, TimestampMixin):
    """Normalized transaction extracted from a bank/credit card statement."""

    __tablename__ = "transactions"

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False,
    )
    agent_run_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_run_items.id"), nullable=True,
    )

    date: Mapped[str] = mapped_column(Date, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    merchant: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    balance: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="HKD")
    category: Mapped[str] = mapped_column(String(100), nullable=False, default="Uncategorized")

    # Matched receipt (bill) document
    receipt_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True,
    )
    match_confidence: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)

    line_items: Mapped[list["LineItem"]] = relationship(
        "LineItem", back_populates="transaction", cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_transactions_owner_date", "owner_id", "date"),
        Index("ix_transactions_category", "category"),
        Index("ix_transactions_document", "document_id"),
        Index("ix_transactions_receipt", "receipt_id"),
    )


class LineItem(Base, TimestampMixin):
    """Individual item extracted from a receipt/bill."""

    __tablename__ = "line_items"

    receipt_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False,
    )
    transaction_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.id", ondelete="SET NULL"), nullable=True,
    )

    item_name: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(8, 2), nullable=False, default=1)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    total: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    sub_category: Mapped[str] = mapped_column(String(100), nullable=False, default="")

    transaction: Mapped[Transaction | None] = relationship(
        "Transaction", back_populates="line_items",
    )

    __table_args__ = (
        Index("ix_line_items_receipt", "receipt_id"),
        Index("ix_line_items_transaction", "transaction_id"),
    )
