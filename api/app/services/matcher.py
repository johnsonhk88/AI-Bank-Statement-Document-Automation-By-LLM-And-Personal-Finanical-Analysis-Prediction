"""Receipt-to-transaction matcher — auto-pairs bills to credit card statements.

Matching strategy:
  1. Date match:     receipt date within ±1 day of transaction date
  2. Amount match:   receipt total within ±$0.50 of transaction amount (absolute)
  3. Merchant match: fuzzy substring match on merchant/description (bonus confidence)

Confidence scoring:
  - Date ±0 days:   +0.40
  - Date ±1 day:    +0.25
  - Amount ±$0.10:  +0.40
  - Amount ±$0.50:  +0.25
  - Merchant match:  +0.20

Threshold: ≥ 0.50 to auto-pair (date + amount match is sufficient).
"""
import logging
from datetime import date, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.transaction import LineItem, Transaction

logger = logging.getLogger(__name__)

# Matching thresholds
DATE_TOLERANCE_DAYS = 1
AMOUNT_TOLERANCE = Decimal("0.50")
AMOUNT_TIGHT_TOLERANCE = Decimal("0.10")
MIN_CONFIDENCE = Decimal("0.50")


def _merchant_overlap(merchant: str, description: str, receipt_text: str) -> bool:
    """Check if merchant name appears in receipt text (case-insensitive)."""
    if not merchant or len(merchant) < 3:
        return False
    merchant_lower = merchant.lower()
    receipt_lower = receipt_text.lower()
    # Check if any significant word from merchant appears in receipt
    words = [w for w in merchant_lower.split() if len(w) > 3]
    return any(w in receipt_lower for w in words)


def _compute_confidence(
    tx_date: date,
    tx_amount: Decimal,
    receipt_date: date,
    receipt_total: Decimal,
    merchant: str,
    description: str,
    receipt_text: str,
) -> Decimal:
    """Compute match confidence between a transaction and a receipt."""
    confidence = Decimal("0")

    # Date score
    day_diff = abs((tx_date - receipt_date).days)
    if day_diff == 0:
        confidence += Decimal("0.40")
    elif day_diff <= DATE_TOLERANCE_DAYS:
        confidence += Decimal("0.25")
    else:
        return Decimal("0")  # Outside date range, no match

    # Amount score (compare absolute values)
    amount_diff = abs(abs(tx_amount) - receipt_total)
    if amount_diff <= AMOUNT_TIGHT_TOLERANCE:
        confidence += Decimal("0.40")
    elif amount_diff <= AMOUNT_TOLERANCE:
        confidence += Decimal("0.25")
    else:
        return Decimal("0")  # Outside amount range, no match

    # Merchant bonus
    if _merchant_overlap(merchant, description, receipt_text):
        confidence += Decimal("0.20")

    return confidence.quantize(Decimal("0.01"))


async def match_receipt_to_transactions(
    db: AsyncSession,
    *,
    owner_id: UUID,
    receipt_document_id: UUID,
    receipt_date: date,
    receipt_total: Decimal,
    receipt_text: str = "",
) -> Transaction | None:
    """Find the best matching transaction for a receipt.

    Searches unmatched transactions within the date window and amount tolerance.
    Returns the matched Transaction (with receipt_id and match_confidence set)
    or None if no match meets the threshold.
    """
    date_start = receipt_date - timedelta(days=DATE_TOLERANCE_DAYS)
    date_end = receipt_date + timedelta(days=DATE_TOLERANCE_DAYS)

    stmt = (
        select(Transaction)
        .where(
            Transaction.owner_id == owner_id,
            Transaction.date >= date_start,
            Transaction.date <= date_end,
            Transaction.receipt_id.is_(None),  # Only unmatched transactions
        )
    )
    result = await db.execute(stmt)
    candidates = result.scalars().all()

    if not candidates:
        return None

    best_match: Transaction | None = None
    best_confidence = Decimal("0")

    for tx in candidates:
        confidence = _compute_confidence(
            tx_date=tx.date,
            tx_amount=tx.amount,
            receipt_date=receipt_date,
            receipt_total=receipt_total,
            merchant=tx.merchant,
            description=tx.description,
            receipt_text=receipt_text,
        )
        if confidence >= MIN_CONFIDENCE and confidence > best_confidence:
            best_confidence = confidence
            best_match = tx

    if best_match is not None:
        best_match.receipt_id = receipt_document_id
        best_match.match_confidence = best_confidence
        logger.info(
            "Matched receipt %s to transaction %s (confidence: %s)",
            receipt_document_id, best_match.id, best_confidence,
        )
        return best_match

    logger.info(
        "No match found for receipt %s (date=%s, total=%s, candidates=%d)",
        receipt_document_id, receipt_date, receipt_total, len(candidates),
    )
    return None


async def link_line_items_to_transaction(
    db: AsyncSession,
    *,
    transaction_id: UUID,
    receipt_document_id: UUID,
) -> int:
    """Link all line items from a receipt to a matched transaction.

    Returns the number of line items linked.
    """
    stmt = (
        select(LineItem)
        .where(
            LineItem.receipt_id == receipt_document_id,
            LineItem.transaction_id.is_(None),
        )
    )
    result = await db.execute(stmt)
    items = result.scalars().all()

    for item in items:
        item.transaction_id = transaction_id

    if items:
        logger.info(
            "Linked %d line items from receipt %s to transaction %s",
            len(items), receipt_document_id, transaction_id,
        )
    return len(items)


async def auto_match_receipt(
    db: AsyncSession,
    *,
    owner_id: UUID,
    receipt_document_id: UUID,
    receipt_date: date,
    receipt_total: Decimal,
    receipt_text: str = "",
) -> tuple[Transaction | None, int]:
    """Full auto-match pipeline: find transaction match, then link line items.

    Returns (matched_transaction, line_items_linked_count).
    """
    matched = await match_receipt_to_transactions(
        db,
        owner_id=owner_id,
        receipt_document_id=receipt_document_id,
        receipt_date=receipt_date,
        receipt_total=receipt_total,
        receipt_text=receipt_text,
    )

    linked_count = 0
    if matched is not None:
        linked_count = await link_line_items_to_transaction(
            db,
            transaction_id=matched.id,
            receipt_document_id=receipt_document_id,
        )

    return matched, linked_count
