import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_create_rejects_empty_document_ids():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/agent-runs",
            json={
                "document_ids": [],
                "agent": "crewai",
                "question": "q",
                "llm_provider_id": "openai",
                "llm_model_id": "openai/gpt-4o-mini",
            },
        )
        assert resp.status_code in (401, 422)


@pytest.mark.asyncio
async def test_create_rejects_unknown_agent():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/agent-runs",
            json={
                "document_ids": ["11111111-1111-1111-1111-111111111111"],
                "agent": "made-up",
                "question": "q",
                "llm_provider_id": "openai",
                "llm_model_id": "openai/gpt-4o-mini",
            },
        )
        assert resp.status_code in (401, 422)
