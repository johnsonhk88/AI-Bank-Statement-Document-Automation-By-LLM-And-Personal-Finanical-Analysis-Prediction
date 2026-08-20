from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text as sa_text
from app.config import settings
from app.db.session import AsyncSessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    assert settings.ADMIN_EMAIL and settings.ADMIN_PASSWORD_HASH, "ADMIN_EMAIL and ADMIN_PASSWORD_HASH must be set"
    yield

app = FastAPI(title="BankAI API", version="0.1.0", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"] if settings.APP_ENV == "dev" else [], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def error_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error":{"code":"INTERNAL","message":"Internal server error"}})

@app.get("/api/health")
async def health():
    try:
        async with AsyncSessionLocal() as s: await s.execute(sa_text("SELECT 1"))
        db_status = "ok"
    except Exception: db_status = "down"
    try:
        import redis; r = redis.Redis.from_url(settings.REDIS_URL); r.ping(); r.close()
        redis_status = "ok"
    except Exception: redis_status = "down"
    try:
        import qdrant_client; qc = qdrant_client.QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        qc.get_collections(); qdrant_status = "ok"
    except Exception: qdrant_status = "down"
    return {"status":"ok","db":db_status,"redis":redis_status,"qdrant":qdrant_status}

from app.api.auth import router as auth_router
from app.api.llm import router as llm_router
from app.api.documents import router as documents_router
from app.api.agent_runs import router as agent_runs_router
from app.api.cashflow import router as cashflow_router
app.include_router(auth_router)
app.include_router(llm_router)
app.include_router(documents_router)
app.include_router(agent_runs_router)
app.include_router(cashflow_router)
