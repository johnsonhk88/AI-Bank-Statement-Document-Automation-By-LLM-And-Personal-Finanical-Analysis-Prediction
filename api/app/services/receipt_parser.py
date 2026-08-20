"""Receipt parser — extracts line items from receipt/bill PDFs.

Always routes to local LLM (llama.cpp) because receipts contain PII.
Produces LineItem rows linked to the receipt document.
"""
import logging
from decimal import Decimal, InvalidOperation
from uuid import UUID

from litellm import acompletion
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.transaction import LineItem
from app.services.llm_router import LLMRouter, TaskType

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Extract all line items from this receipt/bill text.

For each item, provide:
- item_name: product or service name
- quantity: number of units (default 1)
- unit_price: price per unit
- total: quantity × unit_price
- sub_category: type of item (e.g. "produce", "dairy", "beverage", "appetizer", "main course", "dessert", "tax", "tip", "service charge")

Reply with ONLY a numbered list in this exact format:
1. item_name | quantity | unit_price | total | sub_category
2. item_name | quantity | unit_price | total | sub_category

Example:
1. Organic Milk 1L | 1 | 25.90 | 25.90 | dairy
2. Brown Rice 2kg | 2 | 18.50 | 37.00 | grains
3. Service Charge 10% | 1 | 45.00 | 45.00 | service charge"""


def _to_decimal(value: str, default: Decimal = Decimal("0")) -> Decimal:
    """Parse a string to Decimal, stripping currency symbols."""
    cleaned = value.strip().replace("$", "").replace(",", "").replace("HK", "")
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return default


def _parse_line_items_response(raw: str) -> list[dict]:
    """Parse the numbered pipe-delimited LLM response into dicts."""
    items: list[dict] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove number prefix "1. "
        parts = line.split(".", 1)
        content = parts[-1].strip() if len(parts) > 1 else line

        fields = [f.strip() for f in content.split("|")]
        if len(fields) < 4:
            continue

        item_name = fields[0]
        quantity = _to_decimal(fields[1], Decimal("1"))
        unit_price = _to_decimal(fields[2])
        total = _to_decimal(fields[3])
        sub_category = fields[4] if len(fields) > 4 else ""

        # Sanity: if total is 0 but we have unit_price and quantity, compute it
        if total == 0 and unit_price > 0:
            total = (quantity * unit_price).quantize(Decimal("0.01"))

        items.append({
            "item_name": item_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": total,
            "sub_category": sub_category,
        })
    return items


async def extract_line_items_from_text(
    receipt_text: str,
    llm_router: LLMRouter | None = None,
) -> list[dict]:
    """Extract line items from receipt text using local LLM.

    Returns list of dicts with keys: item_name, quantity, unit_price, total, sub_category.
    """
    router = llm_router or LLMRouter()
    model, base_url, api_key = router.resolve(TaskType.RECEIPT)

    try:
        response = await acompletion(
            model=model,
            api_base=base_url or None,
            api_key=api_key,
            messages=[
                {"role": "system", "content": "You extract line items from receipts and bills. Reply with only the numbered list, nothing else."},
                {"role": "user", "content": f"{_EXTRACTION_PROMPT}\n\nReceipt text:\n{receipt_text}"},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        return _parse_line_items_response(raw)
    except Exception:
        logger.exception("Receipt line-item extraction failed")
        return []


async def parse_receipt_and_save(
    db: AsyncSession,
    *,
    receipt_document_id: UUID,
    receipt_text: str,
    llm_router: LLMRouter | None = None,
) -> list[LineItem]:
    """Extract line items from receipt text and save to database.

    Returns created LineItem objects (added to session, caller commits).
    """
    raw_items = await extract_line_items_from_text(receipt_text, llm_router)
    if not raw_items:
        return []

    created: list[LineItem] = []
    for item_data in raw_items:
        line_item = LineItem(
            receipt_id=receipt_document_id,
            item_name=item_data["item_name"],
            quantity=item_data["quantity"],
            unit_price=item_data["unit_price"],
            total=item_data["total"],
            sub_category=item_data["sub_category"],
        )
        db.add(line_item)
        created.append(line_item)

    logger.info(
        "Extracted %d line items from receipt %s",
        len(created), receipt_document_id,
    )
    return created
