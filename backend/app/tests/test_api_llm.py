import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_llm_models_returns_providers():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/llm-models")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert len(data["providers"]) >= 1
