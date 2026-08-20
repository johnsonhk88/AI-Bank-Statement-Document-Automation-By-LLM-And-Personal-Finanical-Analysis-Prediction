"""Transaction normalizer — converts JSONB extraction blobs into normalized rows.

After CrewAI extracts transactions from a PDF (stored as JSONB in agent_run_items),
this service:
  1. Parses each raw transaction dict (date, description, credit, debit, balance, currency)
  2. Computes a signed amount (positive = credit/income, negative = debit/expense)
  3. Extracts a merchant name heuristic from the description
  4. Calls the categorizer (keyword rules first, LLM for unknowns)
  5. Inserts normalized Transaction rows linked to owner, document, and agent_run_item
"""
import logging
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.transaction import Transaction
from app.services.categorizer import categorize_transactions

logger = logging.getLogger(__name__)

# Common date formats found in bank statements
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
]


def parse_date(raw: str) -> date:
    """Parse a date string trying multiple formats."""
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {raw!r}")


def _to_decimal(value: float | int | str | None) -> Decimal | None:
    """Safely convert a value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _compute_amount(credit: float | None, debit: float | None) -> Decimal:
    """Compute signed amount: positive for credits, negative for debits."""
    c = _to_decimal(credit) or Decimal("0")
    d = _to_decimal(debit) or Decimal("0")
    if c > 0:
        return c
    if d > 0:
        return -d
    return c - d


# Patterns for extracting merchant from description
_MERCHANT_SEPARATORS = re.compile(r"\s[-–—/|]\s")


def _extract_merchant(description: str) -> str:
    """Extract merchant name heuristic from transaction description.

    Bank statements often have formats like:
      "POS Purchase - PARK N SHOP - 12345"
      "VISA DEBIT CARD PURCHASE/STARBUCKS HONG KONG"
    We take the most meaningful segment.
    """
    if not description:
        return ""
    # Split on common separators and take the longest segment
    parts = _MERCHANT_SEPARATORS.split(description)
    if len(parts) > 1:
        # Skip very short prefix tokens like "POS", "ATM", "VISA"
        candidates = [p.strip() for p in parts if len(p.strip()) > 4]
        if candidates:
            return candidates[0][:255]
    return description.strip()[:255]


async def normalize_agent_run_item(
    db: AsyncSession,
    *,
    owner_id: UUID,
    document_id: UUID,
    agent_run_item_id: UUID,
    raw_transactions: list[dict],
) -> list[Transaction]:
    """Convert raw JSONB transaction dicts into normalized Transaction rows.

    Returns the list of created Transaction objects (already added to session,
    but caller is responsible for commit).
    """
    if not raw_transactions:
        return []

    # Build dicts for categorizer (it expects description + merchant keys)
    parsed: list[dict] = []
    for raw in raw_transactions:
        try:
            tx_date = parse_date(raw.get("date", ""))
        except ValueError:
            logger.warning("Skipping transaction with unparseable date: %s", raw.get("date"))
            continue

        description = raw.get("description", "")
        merchant = _extract_merchant(description)
        amount = _compute_amount(raw.get("credit"), raw.get("debit"))
        balance = _to_decimal(raw.get("balance"))
        currency = raw.get("currency", "HKD") or "HKD"

        parsed.append({
            "date": tx_date,
            "description": description,
            "merchant": merchant,
            "amount": amount,
            "balance": balance,
            "currency": currency,
        })

    if not parsed:
        return []

    # Categorize all at once (keyword rules first, LLM batch for unknowns)
    categories = await categorize_transactions(parsed)

    created: list[Transaction] = []
    for tx_data, category in zip(parsed, categories):
        tx = Transaction(
            owner_id=owner_id,
            document_id=document_id,
            agent_run_item_id=agent_run_item_id,
            date=tx_data["date"],
            description=tx_data["description"],
            merchant=tx_data["merchant"],
            amount=tx_data["amount"],
            balance=tx_data["balance"],
            currency=tx_data["currency"],
            category=category,
        )
        db.add(tx)
        created.append(tx)

    logger.info(
        "Normalized %d transactions for document %s (agent_run_item %s)",
        len(created), document_id, agent_run_item_id,
    )
    return created
