import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_login_invalid_credentials_returns_401():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/auth/login", json={"email":"x@x.com","password":"x"})
        assert resp.status_code == 401
