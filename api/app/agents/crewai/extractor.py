import instructor
from litellm import acompletion
from pydantic import BaseModel
from app.agents.base import Transaction


class ExtractionResult(BaseModel):
    transactions: list[Transaction]


async def extract_transactions(text: str, llm_model: str, base_url: str, api_key: str) -> list[Transaction]:
    client = instructor.from_litellm(acompletion)
    resp = await client.chat.completions.create(
        model=llm_model,
        api_base=base_url or None,
        api_key=api_key,
        response_model=ExtractionResult,
        messages=[{
            "role": "system",
            "content": "Extract all financial transactions. Fields: date, description, credit, debit, balance, currency (default HKD). Balance-change rule: higher balance = Credit, lower = Debit.",
        }, {
            "role": "user",
            "content": text,
        }],
        temperature=0,
    )
    return resp.transactions
