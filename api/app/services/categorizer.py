"""Transaction categorizer — keyword rules first, LLM for unknowns.

Keyword rules handle ~80% of common merchants instantly (no LLM call).
Remaining unknowns are batched and sent to the cloud LLM (scrubbed of PII)
or local LLM as fallback.
"""
import logging
from litellm import acompletion

from app.services.pii_scrubber import scrub_transactions_batch
from app.services.llm_router import LLMRouter, TaskType

logger = logging.getLogger(__name__)

# Keyword → category mapping. Checked case-insensitively against description + merchant.
# Add your bank-specific merchants here.
KEYWORD_RULES: dict[str, list[str]] = {
    "Income": [
        "salary", "payroll", "wages", "dividend", "interest earned",
        "refund", "cashback", "bonus", "commission",
    ],
    "Groceries": [
        "wellcome", "park n shop", "parknshop", "pns", "hktv", "marketplace",
        "wet market", "city super", "taste", "great", "fusion", "dch food",
        "ztore", "iherb", "costco", "walmart", "tesco", "aldi", "lidl",
    ],
    "Dining": [
        "restaurant", "cafe", "coffee", "mcdonald", "starbucks", "kfc",
        "pizza", "sushi", "ramen", "noodle", "dim sum", "bakery",
        "deliveroo", "foodpanda", "uber eats", "openrice",
    ],
    "Transport": [
        "mtr", "octopus", "bus", "taxi", "uber", "grab", "didi",
        "parking", "petrol", "shell", "caltex", "esso", "sinopec",
        "tunnel", "toll", "ferry",
    ],
    "Rent": [
        "rent", "landlord", "property", "management fee", "lease",
    ],
    "Utilities": [
        "clp", "hk electric", "towngas", "water supplies", "broadband",
        "hkt", "smartone", "3hk", "cmhk", "china mobile",
        "netflix", "spotify", "disney", "youtube", "apple.com/bill",
    ],
    "Shopping": [
        "amazon", "taobao", "tmall", "jd.com", "shopee", "lazada",
        "h&m", "zara", "uniqlo", "ikea", "decathlon", "apple store",
        "fortress", "broadway", "suning",
    ],
    "Healthcare": [
        "hospital", "clinic", "doctor", "dental", "pharmacy", "watsons",
        "mannings", "medical", "health", "insurance premium",
    ],
    "Entertainment": [
        "cinema", "movie", "concert", "ticket", "gym", "fitness",
        "sports", "golf", "travel", "hotel", "airbnb", "booking.com",
        "klook", "agoda",
    ],
    "Education": [
        "school", "tuition", "university", "course", "training",
        "udemy", "coursera", "book",
    ],
    "Transfer": [
        "transfer", "fps", "payme", "wechat pay", "alipay",
        "own account", "self transfer",
    ],
}


def categorize_by_keywords(description: str, merchant: str) -> str | None:
    """Return category if any keyword matches, else None."""
    text = f"{description} {merchant}".lower()
    for category, keywords in KEYWORD_RULES.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return None


async def categorize_batch_with_llm(
    transactions: list[dict],
    llm_router: LLMRouter | None = None,
) -> list[str]:
    """Categorize a batch of transactions using cloud LLM (PII-scrubbed).

    Returns list of category strings, same order as input.
    """
    router = llm_router or LLMRouter()
    model, base_url, api_key = router.resolve(TaskType.CATEGORIZE)

    scrubbed = scrub_transactions_batch(transactions)

    categories_list = ", ".join(sorted(KEYWORD_RULES.keys())) + ", Other"

    prompt_lines = []
    for i, tx in enumerate(scrubbed):
        prompt_lines.append(f"{i+1}. {tx['date']} | {tx['merchant']} | {tx['description']} | {tx['amount']} {tx['currency']}")

    prompt = f"""Categorize each transaction into exactly one category.
Categories: {categories_list}

Transactions:
{chr(10).join(prompt_lines)}

Reply with ONLY the category for each line, numbered. Example:
1. Groceries
2. Dining
3. Transport"""

    try:
        response = await acompletion(
            model=model,
            api_base=base_url or None,
            api_key=api_key,
            messages=[
                {"role": "system", "content": "You categorize financial transactions. Reply with only numbered categories, nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""

        # Parse "1. Groceries\n2. Dining\n..."
        results: list[str] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove number prefix
            parts = line.split(".", 1)
            cat = parts[-1].strip() if len(parts) > 1 else line
            # Validate against known categories
            if cat in KEYWORD_RULES or cat == "Other":
                results.append(cat)
            else:
                results.append("Other")

        # Pad if LLM returned fewer results
        while len(results) < len(transactions):
            results.append("Other")

        return results[:len(transactions)]

    except Exception:
        logger.exception("LLM categorization failed, defaulting to 'Other'")
        return ["Other"] * len(transactions)


async def categorize_transactions(transactions: list[dict]) -> list[str]:
    """Categorize transactions: keywords first, LLM for unknowns.

    Returns list of category strings, same length as input.
    """
    categories: list[str] = []
    unknown_indices: list[int] = []
    unknown_txs: list[dict] = []

    for i, tx in enumerate(transactions):
        cat = categorize_by_keywords(
            tx.get("description", ""),
            tx.get("merchant", ""),
        )
        if cat:
            categories.append(cat)
        else:
            categories.append("")  # placeholder
            unknown_indices.append(i)
            unknown_txs.append(tx)

    if unknown_txs:
        logger.info("Categorizing %d/%d transactions via LLM", len(unknown_txs), len(transactions))
        llm_cats = await categorize_batch_with_llm(unknown_txs)
        for idx, cat in zip(unknown_indices, llm_cats):
            categories[idx] = cat

    # Replace any remaining empty with "Other"
    return [c if c else "Other" for c in categories]
