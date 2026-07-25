import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_upload_rejects_non_pdf():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/documents", files={"file": ("test.txt", b"hello", "text/plain")})
        assert resp.status_code in (401, 422)

@pytest.mark.asyncio
async def test_list_requires_auth():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/documents")
        assert resp.status_code == 401
