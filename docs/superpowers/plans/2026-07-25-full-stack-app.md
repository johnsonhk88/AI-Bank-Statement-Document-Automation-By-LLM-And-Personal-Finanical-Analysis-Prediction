# Full-Stack App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a full-stack bank-statement agent wrapper (FastAPI + React + Postgres + Qdrant + Celery) while keeping all experiment code untouched.

**Architecture:** New top-level `api/`, `web/`, `infra/` dirs; docker-compose-first with dev + prod profiles; CrewAI adapter port from `backend/app/core/ai_agent_skills_dev.ipynb`; Nginx serves SPA on :80 in prod.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.0 async, Alembic, AsyncPG, Celery 5 (prefork), Qdrant 1.13+, Pydantic v2, python-jose, passlib, CrewAI 1.14.7, LiteLLM 1.79.2, langchain-qdrant, instructor 1.15+, React 18, TypeScript 5, Vite 5, TanStack Query 5, zustand, Tailwind 3, Docker Compose v2.

**Spec:** `docs/superpowers/specs/2026-07-25-full-stack-app-design.md`

## Global Constraints

- Python 3.11 floor; CrewAI 1.14.7; LiteLLM 1.79.2; instructor 1.15.1
- All API errors: `{"error":{"code":"...","message":"...","details":{}}}`
- Soft delete via `deleted_at` column; no hard delete for v1
- Single env-seeded admin (`ADMIN_EMAIL`, `ADMIN_PASSWORD_HASH` required at startup)
- `PYTHONPATH=/app:/app/backend` in all containers
- One Docker image for API + Worker (different entrypoints)
- Prefork Celery pool; `asyncio.run()` per task
- JWT 15-min TTL, HS256, bcrypt via passlib
- Document dedup via SHA-256
- Qdrant collections: `doc_<uuid>`; embeddings `intfloat/multilingual-e5-small` dim 768
- Frontend polls `/api/agent-runs/{id}` every ~2s; stops on terminal status
- Nginx SPA fallback + `/api` reverse proxy in prod

## Tasks

### Task 1: API project skeleton + config + db session + models + health

**Files:**
- Create: `api/pyproject.toml`, `api/requirements.txt`, `api/app/__init__.py`, `api/app/config.py`
- Create: `api/app/db/__init__.py`, `api/app/db/base.py`, `api/app/db/session.py`
- Create: `api/app/models/__init__.py`, `api/app/models/mixins.py`, `api/app/models/user.py`, `api/app/models/document.py`, `api/app/models/agent_run.py`
- Create: `api/app/main.py`, `api/app/deps.py`, `api/app/schemas/__init__.py`
- Create: `api/app/tests/__init__.py`, `api/app/tests/conftest.py`
- Create: `api/README.md`

**Interfaces:**
- Produces: `api.app.config.Settings` (pydantic-settings with POSTGRES_URL, REDIS_URL, QDRANT_URL, JWT_SECRET, etc.), `api.app.db.session.get_session` (async generator), `api.app.db.base.Base` (DeclarativeBase), `api.app.models.User`, `api.app.models.Document`, `api.app.models.AgentRun`, `api.app.models.AgentRunItem`, FastAPI `app` with `GET /api/health`, `api.app.deps.get_db`, `api.app.deps.get_current_user` (placeholder)

**Steps:**

- [ ] Step 1: Copy `api/pyproject.toml` and `api/requirements.txt` with deps pinned from root requirements: fastapi, uvicorn, sqlalchemy[asyncio], asyncpg, alembic, celery[redis], redis, qdrant-client, langchain-qdrant, langchain-huggingface, langchain-text-splitters, langchain-community, sentence-transformers, litellm==1.79.2, crewai==1.14.7, crewai[tools]==1.14.7, instructor==1.15.1, pymupdf, python-jose[cryptography], passlib[bcrypt], pydantic-settings, httpx, pytest, pytest-asyncio

- [ ] Step 2: Write `api/app/config.py`:
```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_ENV: str = "dev"
    POSTGRES_USER: str = "bankai"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "bankai"
    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    JWT_SECRET: str = ""
    JWT_ALG: str = "HS256"
    JWT_TTL_MINUTES: int = 15
    ADMIN_EMAIL: str = ""
    ADMIN_PASSWORD_HASH: str = ""
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    UPLOAD_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "uploads"
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```
Verify: `cd api && python -c "from app.config import settings; print(settings.POSTGRES_URL)"`

- [ ] Step 3: Write `api/app/db/base.py`:
```python
import uuid
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

convention = {"ix":"ix_%(column_0_label)s","uq":"uq_%(table_name)s_%(column_0_name)s","ck":"ck_%(table_name)s_%(constraint_name)s","fk":"fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s","pk":"pk_%(table_name)s"}
metadata = MetaData(naming_convention=convention)

class Base(DeclarativeBase):
    metadata = metadata
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
```
And `api/app/db/session.py`:
```python
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from api.app.config import settings

engine = create_async_engine(settings.POSTGRES_URL, echo=settings.APP_ENV == "dev")
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
```

- [ ] Step 4: Write `api/app/models/mixins.py`:
```python
import datetime
from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

class TimestampMixin:
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SoftDeleteMixin:
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
```
And `api/app/models/user.py`:
```python
from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column
from api.app.db.base import Base
from api.app.models.mixins import TimestampMixin

class User(Base, TimestampMixin):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=True)
```
And `api/app/models/document.py`:
```python
from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from api.app.db.base import Base
from api.app.models.mixins import SoftDeleteMixin, TimestampMixin

class Document(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "documents"
    owner_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(127), default="application/pdf")
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    qdrant_collection: Mapped[str | None] = mapped_column(String(64), nullable=True)
```
And `api/app/models/agent_run.py`:
```python
from __future__ import annotations
import uuid
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from api.app.db.base import Base
from api.app.models.mixins import SoftDeleteMixin, TimestampMixin

class AgentRun(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "agent_runs"
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    agent: Mapped[str] = mapped_column(String(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(255), nullable=False)
    started_at = mapped_column(TimestampMixin.created_at.type, nullable=True)
    finished_at = mapped_column(TimestampMixin.created_at.type, nullable=True)
    items: Mapped[list["AgentRunItem"]] = relationship("AgentRunItem", back_populates="run", cascade="all, delete-orphan")

class AgentRunItem(Base, TimestampMixin):
    __tablename__ = "agent_run_items"
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    markdown_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    transactions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    started_at = mapped_column(TimestampMixin.created_at.type, nullable=True)
    finished_at = mapped_column(TimestampMixin.created_at.type, nullable=True)
    run: Mapped[AgentRun] = relationship("AgentRun", back_populates="items")
```
And `api/app/models/__init__.py`:
```python
from api.app.models.user import User
from api.app.models.document import Document
from api.app.models.agent_run import AgentRun, AgentRunItem
__all__ = ["User", "Document", "AgentRun", "AgentRunItem"]
```

- [ ] Step 5: Write `api/app/main.py`:
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text as sa_text
from api.app.config import settings
from api.app.db.session import AsyncSessionLocal

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
        return JSONResponse(500, content={"error":{"code":"INTERNAL","message":"Internal server error"}})

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
```
And `api/app/deps.py`:
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.app.db.session import get_session as _get_session
from api.app.core.security import decode_access_token
from api.app.models import User

async def get_db(session: AsyncSession = Depends(_get_session)) -> AsyncSession:
    return session

bearer = HTTPBearer(auto_error=False)

async def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: AsyncSession = Depends(get_db)) -> User:
    if not creds: raise HTTPException(401, detail={"error":{"code":"UNAUTHORIZED","message":"Missing token"}})
    payload = decode_access_token(creds.credentials)
    if not payload: raise HTTPException(401, detail={"error":{"code":"UNAUTHORIZED","message":"Invalid token"}})
    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user = result.scalar_one_or_none()
    if not user: raise HTTPException(401, detail={"error":{"code":"UNAUTHORIZED","message":"User not found"}})
    return user
```

- [ ] Step 6: Run `cd api && python -c "from app.main import app; print(app.title)"` → prints "BankAI API"

- [ ] Step 7: Commit
```bash
git add api/pyproject.toml api/requirements.txt api/app/ api/README.md
git commit -m "feat(api): project skeleton, config, models, app factory, health endpoint"
```

### Task 2: Security + auth endpoints

**Files:**
- Create: `api/app/core/__init__.py`, `api/app/core/security.py`
- Create: `api/app/schemas/auth.py`
- Create: `api/app/api/__init__.py`, `api/app/api/auth.py`
- Modify: `api/app/deps.py` (already has get_current_user from Task 1; verify)
- Modify: `api/app/main.py` (include auth router)
- Test: `api/app/tests/test_api_auth.py`

**Interfaces:**
- Consumes: `api.app.models.User`, `api.app.config.settings`
- Produces: `api.app.core.security.create_access_token`, `decode_access_token`, `verify_password`, `hash_password`, `POST /api/auth/login`, `GET /api/auth/me`

**Steps:**

- [ ] Step 1: Write `api/app/core/security.py`:
```python
import datetime
from jose import JWTError, jwt
from passlib.context import CryptContext
from api.app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str: return pwd_context.hash(plain)
def verify_password(plain: str, hashed: str) -> bool: return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=settings.JWT_TTL_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALG)

def decode_access_token(token: str) -> dict | None:
    try: return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
    except JWTError: return None
```

- [ ] Step 2: Write `api/app/schemas/auth.py`:
```python
from pydantic import BaseModel, EmailStr

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"

class UserOut(BaseModel):
    id: str
    email: str
    is_admin: bool
    model_config = {"from_attributes": True}
```

- [ ] Step 3: Write `api/app/api/auth.py`:
```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.app.core.security import create_access_token, decode_access_token, verify_password
from api.app.models.user import User
from api.app.schemas.auth import LoginRequest, TokenResponse, UserOut
from api.app.deps import get_db, get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer()

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, detail={"error":{"code":"UNAUTHORIZED","message":"Invalid credentials"}})
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))

@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)
```

- [ ] Step 4: Update `api/app/main.py` to include the router: add `from api.app.api.auth import router as auth_router` and `app.include_router(auth_router)` at the bottom.

- [ ] Step 5: Write test `api/app/tests/test_api_auth.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_login_invalid_credentials_returns_401():
    from api.app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/auth/login", json={"email":"x@x.com","password":"x"})
        assert resp.status_code == 401
```
Run: `cd api && python -m pytest app/tests/test_api_auth.py -v` (may fail if DB not available; EXPECTS to fail gracefully or skip — this is structural)

- [ ] Step 6: Commit:
```bash
git add api/app/core/ api/app/schemas/auth.py api/app/api/__init__.py api/app/api/auth.py api/app/main.py api/app/tests/test_api_auth.py
git commit -m "feat(api): JWT security module + auth/login + auth/me endpoints"
```

### Task 3: Alembic init + migration + seed admin

**Files:**
- Create: `api/alembic.ini`, `api/alembic/env.py`, `api/alembic/script.py.mako`, `api/alembic/versions/0001_initial_schema.py`

**Interfaces:**
- Consumes: all models from Task 1
- Produces: runnable migration that creates users, documents, agent_runs, agent_run_items tables + seeds admin row from env

**Steps:**

- [ ] Step 1: Write `api/alembic.ini` (standard alembic ini with `script_location = alembic`, blank `sqlalchemy.url`).

- [ ] Step 2: Write `api/alembic/env.py` — async env using `create_async_engine(settings.POSTGRES_URL)`, `target_metadata = Base.metadata`, `asyncio.run(run_migrations_online())`.

- [ ] Step 3: Write `api/alembic/script.py.mako` (standard mako template).

- [ ] Step 4: Write `api/alembic/versions/0001_initial_schema.py` — `op.create_table("users", ...)` with id UUID, email string unique, password_hash, is_admin, timestamps; `op.create_table("documents", ...)` with owner_id FK, storage_path, content_sha256 unique, soft delete; `op.create_table("agent_runs", ...)` with owner_id FK, agent, question, status, llm_provider, llm_model, timestamps, soft delete; `op.create_table("agent_run_items", ...)` with run_id FK cascade, document_id FK, celery_task_id, status, error, markdown_report, transactions JSONB, timestamps. Add indexes. In the `import os` block at the end of upgrade(), read `ADMIN_EMAIL` and `ADMIN_PASSWORD_HASH` from os.environ and execute INSERT if both are present.

- [ ] Step 5: Start postgres (in docker-compose later, but for testing can run: `docker run -d -p 5432:5432 -e POSTGRES_USER=bankai -e POSTGRES_PASSWORD=bankai -e POSTGRES_DB=bankai postgres:16-alpine`). Then run:
```bash
cd api && JWT_SECRET=test ADMIN_EMAIL=admin@test.com ADMIN_PASSWORD_HASH='$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW' POSTGRES_HOST=localhost POSTGRES_PASSWORD=bankai pip install -e . 2>/dev/null; alembic -c alembic.ini upgrade head
```
Verify: `docker exec <pg> psql -U bankai -d bankai -c "SELECT * FROM users;"` shows one admin row.

- [ ] Step 6: Commit:
```bash
git add api/alembic.ini api/alembic/
git commit -m "feat(api): alembic init + 0001 initial schema migration + admin seed"
```

### Task 4: Storage + hashing + documents API

**Files:**
- Create: `api/app/core/hashing.py`, `api/app/core/storage.py`
- Create: `api/app/schemas/document.py`, `api/app/api/documents.py`
- Modify: `api/app/main.py` (include documents router)
- Test: `api/app/tests/test_api_documents.py`

**Interfaces:**
- Consumes: `api.app.models.Document`, `api.app.deps.get_db`, `api.app.deps.get_current_user`
- Produces: `api.app.core.hashing.sha256_file(path) -> str`, `api.app.core.storage.save_upload(file, owner_id) -> str`, `POST /api/documents`, `GET /api/documents`, `GET /api/documents/{id}`, `GET /api/documents/{id}/content`, `DELETE /api/documents/{id}`

**Steps:**

- [ ] Step 1: Write `api/app/core/hashing.py`:
```python
import hashlib

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] Step 2: Write `api/app/core/storage.py`:
```python
import datetime, uuid
from pathlib import Path
from fastapi import UploadFile
from api.app.config import settings

def save_upload(file: UploadFile, owner_id: uuid.UUID) -> str:
    now = datetime.datetime.now(datetime.UTC)
    subdir = settings.UPLOAD_ROOT / str(now.year) / f"{now.month:02d}"
    subdir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    dest = subdir / f"{doc_id}.pdf"
    with open(dest, "wb") as f: f.write(file.file.read())
    return str(dest.relative_to(settings.UPLOAD_ROOT.parent))
```

- [ ] Step 3: Write `api/app/schemas/document.py`:
```python
import datetime
from pydantic import BaseModel

class DocumentOut(BaseModel):
    id: str; original_filename: str; mime_type: str; size_bytes: int | None = None; page_count: int | None = None; deduplicated: bool = False; created_at: datetime.datetime
    model_config = {"from_attributes": True}

class DocumentListResponse(BaseModel):
    items: list[DocumentOut]; total: int
```

- [ ] Step 4: Write `api/app/api/documents.py` with endpoints:
  - `POST /api/documents` — accepts multipart `file`; validates .pdf extension; saves via storage.save_upload; computes sha256 via hashing.sha256_file; checks `Document.content_sha256` for dedup (returns existing row with deduplicated=true if found); creates new Document row; returns 201 DocumentOut
  - `GET /api/documents` — paginated list (`limit` default 50, `offset` default 0); where clause `owner_id==current_user.id AND deleted_at IS NULL`; order by created_at desc; returns DocumentListResponse
  - `GET /api/documents/{id}` — fetch by id with deleted_at NULL check; 404 if not found or not owner
  - `GET /api/documents/{id}/content` — `FileResponse` to the stored PDF path
  - `DELETE /api/documents/{id}` — sets `deleted_at`; refuses (409) if document has AgentRunItem with status pending/running
  All endpoints use `Depends(get_db)` and `Depends(get_current_user)`.

- [ ] Step 5: Update `api/app/main.py` to `from api.app.api.documents import router as documents_router` and `app.include_router(documents_router)`.

- [ ] Step 6: Write test `api/app/tests/test_api_documents.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_upload_rejects_non_pdf():
    from api.app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/documents", files={"file":("test.txt",b"hello","text/plain")})
        assert resp.status_code == 422
```
Run: `cd api && python -m pytest app/tests/test_api_documents.py -v`

- [ ] Step 7: Commit:
```bash
git add api/app/core/hashing.py api/app/core/storage.py api/app/schemas/document.py api/app/api/documents.py api/app/tests/test_api_documents.py api/app/main.py
git commit -m "feat(api): document upload with sha256 dedup, list, detail, download, soft delete"
```

### Task 5: LLM provider registry + /llm-models endpoint

**Files:**
- Create: `api/app/agents/__init__.py`, `api/app/agents/llm_provider.py`
- Create: `api/app/schemas/llm.py`, `api/app/api/llm.py`
- Create: `api/config/llm_providers.yaml`
- Modify: `api/app/main.py` (include llm router)
- Test: `api/app/tests/test_api_llm.py`

**Interfaces:**
- Produces: `api.app.agents.llm_provider.LLMProviderRegistry` (load YAML, `resolve(provider_id, model_id) -> (litellm_model, base_url, api_key)`, `list_providers() -> list[ProviderOut]` with availability probe), `GET /api/llm-models`

**Steps:**

- [ ] Step 1: Write `api/config/llm_providers.yaml` with providers: lm-studio (local, base_url=http://host.docker.internal:1234/v1, api_key_env=LM_STUDIO_API_KEY, models: openai/qwen2.5-14b-instruct, openai/google/gemma-3-12b-qat); openrouter (cloud, base_url=https://openrouter.ai/api/v1, api_key_env=OPENROUTER_API_KEY, models: openrouter/google/gemini-2.5-flash, openrouter/anthropic/claude-3.5-sonnet); gemini (cloud, api_key_env=GOOGLE_API_KEY, models: gemini/gemini-2.5-flash); ollama (local, base_url=http://host.docker.internal:11434, models: ollama/llama3.2); openai (cloud, api_key_env=OPENAI_API_KEY, models: openai/gpt-4o-mini, openai/gpt-4o).

- [ ] Step 2: Write `api/app/agents/llm_provider.py`:
```python
import httpx, os, yaml
from pathlib import Path
from pydantic import BaseModel

class ModelConfig(BaseModel): id: str; display_name: str
class ProviderConfig(BaseModel): id: str; display_name: str; kind: str; base_url: str = ""; api_key_env: str = ""; models: list[ModelConfig]
class LLMCatalog(BaseModel): providers: list[ProviderConfig]

class LLMProviderRegistry:
    def __init__(self, catalog_path: Path | None = None):
        if catalog_path is None:
            catalog_path = Path(__file__).resolve().parent.parent.parent / "config" / "llm_providers.yaml"
        self._catalog = LLMCatalog(**yaml.safe_load(catalog_path.read_text()))

    def resolve(self, provider_id: str, model_id: str) -> tuple[str, str, str]:
        # Returns (litellm_model, base_url, api_key)
        for p in self._catalog.providers:
            if p.id == provider_id:
                for m in p.models:
                    if m.id == model_id:
                        key = os.environ.get(p.api_key_env, "lm-studio" if p.kind == "local" else "")
                        if p.api_key_env and not key and p.kind != "local":
                            raise ValueError(f"API key not set: {p.api_key_env}")
                        return m.id, p.base_url or "", key
                raise ValueError(f"Model {model_id} not found in provider {provider_id}")
        raise ValueError(f"Provider {provider_id} not found")

    def list_providers(self) -> list[dict]:
        result = []
        for p in self._catalog.providers:
            avail, reason = True, None
            if p.kind == "local":
                try: r = httpx.get(p.base_url + "/models" if p.base_url else p.base_url, timeout=3); avail = r.is_success; reason = None if avail else f"HTTP {r.status_code}"
                except Exception as e: avail = False; reason = str(e)
            else:
                if p.api_key_env and not os.environ.get(p.api_key_env): avail = False; reason = f"{p.api_key_env} not set"
            result.append({"id":p.id,"display_name":p.display_name,"kind":p.kind,"available":avail,"unavailable_reason":reason,"models":[{"id":m.id,"display_name":m.display_name} for m in p.models]})
        return result
```

- [ ] Step 3: Write `api/app/schemas/llm.py`:
```python
from pydantic import BaseModel
class LLMModelOut(BaseModel): id: str; display_name: str
class LLMProviderOut(BaseModel): id: str; display_name: str; kind: str; available: bool; unavailable_reason: str | None = None; models: list[LLMModelOut]
class LLMCatalogResponse(BaseModel): providers: list[LLMProviderOut]
```

- [ ] Step 4: Write `api/app/api/llm.py`:
```python
from fastapi import APIRouter
from api.app.agents.llm_provider import LLMProviderRegistry
from api.app.schemas.llm import LLMCatalogResponse, LLMModelOut, LLMProviderOut

router = APIRouter(prefix="/api/llm-models", tags=["llm"])
_registry: LLMProviderRegistry | None = None

def get_llm_registry() -> LLMProviderRegistry:
    global _registry
    if _registry is None: _registry = LLMProviderRegistry()
    return _registry

@router.get("", response_model=LLMCatalogResponse)
async def list_llm_models():
    providers = get_llm_registry().list_providers()
    return LLMCatalogResponse(providers=[
        LLMProviderOut(id=p["id"], display_name=p["display_name"], kind=p["kind"],
                       available=p["available"], unavailable_reason=p["unavailable_reason"],
                       models=[LLMModelOut(id=m["id"], display_name=m["display_name"]) for m in p["models"]])
        for p in providers
    ])
```
Update `api/app/main.py` to include the llm router.

- [ ] Step 5: Write test `api/app/tests/test_api_llm.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_llm_models_returns_providers():
    from api.app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/llm-models")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert len(data["providers"]) >= 1
```
Run: `cd api && python -m pytest app/tests/test_api_llm.py -v`

- [ ] Step 6: Commit:
```bash
git add api/config/ api/app/agents/__init__.py api/app/agents/llm_provider.py api/app/schemas/llm.py api/app/api/llm.py api/app/tests/test_api_llm.py api/app/main.py
git commit -m "feat(api): LLM provider registry (YAML config) + /llm-models endpoint with availability probes"
```

### Task 6: AgentRegistry + BaseAgentAdapter + stubs + CrewAI adapter

**Files:**
- Create: `api/app/agents/base.py`, `api/app/agents/registry.py`
- Create: `api/app/agents/crewai/__init__.py`, `api/app/schemas/agent_run.py`
- Create: `api/app/agents/crewai/rag.py`, `api/app/agents/crewai/extractor.py`, `api/app/agents/crewai/adapter.py`
- Create: `api/app/agents/stubs.py`
- Create: `api/app/workers/__init__.py`
- Modify: `api/app/agents/__init__.py` (register all adapters)
- Test: `api/app/tests/test_crewai_adapter.py`

**Interfaces:**
- Consumes: LLMProviderRegistry, backend.app.skills.crewai_skills_loader, Qdrant, LiteLLM, CrewAI
- Produces: `AgentRegistry`, `BaseAgentAdapter.run(pdf_path, question, llm_provider_id, llm_model_id, agent_run_item_id) -> AgentResult`, `CrewAIAdapter`, `DeepAgentsAdapter`, `HermesAdapter`

**Steps:**

- [ ] Step 1: Write `api/app/agents/base.py`:
```python
from abc import ABC, abstractmethod
from uuid import UUID
from pydantic import BaseModel, Field

class Transaction(BaseModel):
    date: str = ""; description: str = ""; credit: float | None = None; debit: float | None = None; balance: float | None = None; currency: str = "HKD"

class AgentResult(BaseModel):
    markdown_report: str; transactions: list[Transaction] = Field(default_factory=list); raw: dict | None = None

class AgentInfo(BaseModel):
    name: str; display_name: str; enabled: bool; description: str

class BaseAgentAdapter(ABC):
    name: str; display_name: str; enabled: bool = True; description: str = ""

    @abstractmethod
    async def run(self, *, pdf_path: str, question: str, llm_provider_id: str, llm_model_id: str, agent_run_item_id: UUID) -> AgentResult: ...
```

- [ ] Step 2: Write `api/app/agents/registry.py`:
```python
from api.app.agents.base import AgentInfo, BaseAgentAdapter

class AgentRegistry:
    def __init__(self): self._adapters: dict[str, BaseAgentAdapter] = {}
    def register(self, adapter: BaseAgentAdapter): self._adapters[adapter.name] = adapter
    def get(self, name: str) -> BaseAgentAdapter:
        if name not in self._adapters: raise ValueError(f"Unknown agent: {name}")
        return self._adapters[name]
    def list(self) -> list[AgentInfo]:
        return [AgentInfo(name=a.name, display_name=a.display_name, enabled=a.enabled, description=a.description) for a in self._adapters.values()]
    def is_valid(self, name: str) -> bool: return name in self._adapters and self._adapters[name].enabled

agent_registry = AgentRegistry()
```

- [ ] Step 3: Write stubs `api/app/agents/stubs.py`:
```python
from api.app.agents.base import BaseAgentAdapter, AgentResult

class DeepAgentsAdapter(BaseAgentAdapter):
    name = "deep-agents"; display_name = "Deep Agents (LangChain)"; enabled = False; description = "Coming soon"
    async def run(self, **kw): raise NotImplementedError

class HermesAdapter(BaseAgentAdapter):
    name = "hermes"; display_name = "Hermes (Docker)"; enabled = False; description = "Coming soon"
    async def run(self, **kw): raise NotImplementedError
```

- [ ] Step 4: Write `api/app/agents/crewai/rag.py`:
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from api.app.config import settings

_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None: _embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL, model_kwargs={"device":"cpu"})
    return _embeddings

def store_in_qdrant(text: str, collection_name: str) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = splitter.create_documents([text])
    QdrantVectorStore.from_documents(docs, embedding=_get_embeddings(), url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None, collection_name=collection_name)

def query_qdrant(question: str, collection_name: str, k: int = 4) -> str:
    store = QdrantVectorStore.from_existing_collection(embedding=_get_embeddings(), collection_name=collection_name, url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
    results = store.similarity_search(question, k=k)
    return "\n\n".join([r.page_content for r in results])
```

- [ ] Step 5: Write `api/app/agents/crewai/extractor.py`:
```python
import instructor
from litellm import acompletion
from pydantic import BaseModel
from api.app.agents.base import Transaction

class ExtractionResult(BaseModel):
    transactions: list[Transaction]

async def extract_transactions(text: str, llm_model: str, base_url: str, api_key: str) -> list[Transaction]:
    client = instructor.from_litellm(acompletion)
    resp = await client.chat.completions.create(
        model=llm_model, api_base=base_url or None, api_key=api_key,
        response_model=ExtractionResult,
        messages=[{"role":"system","content":"Extract all financial transactions. Fields: date, description, credit, debit, balance, currency (default HKD). Balance-change rule: higher balance = Credit, lower = Debit."},
                  {"role":"user","content": text}],
        temperature=0)
    return resp.transactions
```

- [ ] Step 6: Write `api/app/agents/crewai/adapter.py`:
```python
import sys
from pathlib import Path
from uuid import UUID
from crewai import Agent, Crew, LLM, Process, Task
from api.app.agents.base import AgentResult, BaseAgentAdapter
from api.app.agents.crewai.rag import query_qdrant
from api.app.agents.crewai.extractor import extract_transactions
from api.app.agents.llm_provider import LLMProviderRegistry

BACKEND = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND))
from app.skills.crewai_skills_loader import load_skills

class CrewAIAdapter(BaseAgentAdapter):
    name = "crewai"; display_name = "CrewAI (Multi-Agent)"; enabled = True; description = "CrewAI-based multi-agent bank statement analysis"

    def __init__(self): self._llm_registry = LLMProviderRegistry()

    async def run(self, *, pdf_path: str, question: str, llm_provider_id: str, llm_model_id: str, agent_run_item_id: UUID) -> AgentResult:
        litellm_model, base_url, api_key = self._llm_registry.resolve(llm_provider_id, llm_model_id)
        llm = LLM(model=litellm_model, base_url=base_url, api_key=api_key)
        skills = load_skills()

        agent1 = Agent(role="Bank Statement Parser", goal="Extract transactions", backstory="You parse bank statements.", llm=llm, skills=skills)
        agent2 = Agent(role="PII Redactor", goal="Redact PII before storage", backstory="You protect privacy.", llm=llm, skills=skills)
        agent3 = Agent(role="Vector Store Manager", goal="Index content", backstory="You manage knowledge bases.", llm=llm, skills=skills)
        agent4 = Agent(role="Financial Analyst", goal="Analyze and answer questions", backstory="You are a CFA.", llm=llm, skills=skills)
        agent5 = Agent(role="Output Formatter", goal="Format final report", backstory="You format reports.", llm=llm, skills=skills)

        task1 = Task(description=f"Extract text from PDF: {pdf_path}. Apply bank-statement-parsing skill.", expected_output="Structured transaction list", agent=agent1)
        task2 = Task(description="Redact all PII from extracted text. Apply pii-handling skill.", expected_output="Redacted text", agent=agent2)
        task3 = Task(description="Store redacted text in Qdrant. Use vector_store tool.", expected_output="Storage confirmation", agent=agent3)
        task4 = Task(description=f"Answer: {question}. Use rag query and financial-analysis skill. Cross-check: Opening + Credits - Debits = Closing.", expected_output="Markdown financial report", agent=agent4, output_file=f"/tmp/report_{agent_run_item_id}.md")
        task5 = Task(description="Apply output-format skill.", expected_output="Final formatted markdown", agent=agent5)

        crew = Crew(tasks=[task1, task2, task3, task4, task5], process=Process.sequential, verbose=False)
        result = await crew.kickoff_async(inputs={"pdf_path": pdf_path, "query": question})

        raw_text = getattr(result, "raw", str(result))
        try: transactions = await extract_transactions(raw_text, litellm_model, base_url, api_key)
        except Exception: transactions = []

        return AgentResult(markdown_report=str(result), transactions=transactions, raw={"crewai_raw": str(result)})
```

- [ ] Step 7: Write `api/app/agents/__init__.py`:
```python
from api.app.agents.registry import agent_registry
from api.app.agents.crewai.adapter import CrewAIAdapter
from api.app.agents.stubs import DeepAgentsAdapter, HermesAdapter

agent_registry.register(CrewAIAdapter())
agent_registry.register(DeepAgentsAdapter())
agent_registry.register(HermesAdapter())
```

- [ ] Step 8: Write test `api/app/tests/test_crewai_adapter.py` with `@patch("crewai.Crew")` mocking `kickoff_async` to return a mock with `raw` attribute; verify adapter returns `AgentResult` with markdown and empty transactions list.

- [ ] Step 9: Commit:
```bash
git add api/app/agents/ api/app/schemas/agent_run.py api/app/tests/test_crewai_adapter.py
git commit -m "feat(api): agent registry, BaseAgentAdapter, CrewAI adapter port, Deep Agents + Hermes stubs"
```

### Task 7: Celery worker

**Files:**
- Create: `api/app/worker/__init__.py`, `api/app/worker/celery_app.py`, `api/app/worker/runner.py`, `api/app/worker/tasks.py`
- Test: `api/app/tests/test_worker_task.py`

**Interfaces:**
- Consumes: `agent_registry`, settings, models
- Produces: `celery_app`, `run_agent_item.delay(agent_run_item_id)` task, `asyncio.run(runner.run_item_async(item_id))`

**Steps:**

- [ ] Step 1: Write `api/app/worker/celery_app.py`:
```python
from celery import Celery
from api.app.config import settings

celery_app = Celery("api", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    task_acks_late=True, task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1, task_time_limit=600, task_soft_time_limit=480,
    timezone="UTC", imports=["api.app.worker.tasks"],
)
```

- [ ] Step 2: Write `api/app/worker/runner.py` — async function `run_item_async(item_id: UUID)`:
  - Opens async DB session
  - Loads `AgentRunItem` + parent `AgentRun` (for question, agent name, llm_config) + `Document` (for pdf path)
  - Sets item.status="running", item.started_at=now
  - Calls `adapter = agent_registry.get(run.agent); result = await adapter.run(pdf_path=full_path, question=run.question, llm_provider_id=run.llm_provider, llm_model_id=run.llm_model, agent_run_item_id=item.id)`
  - On success: sets item.status="succeeded", markdown_report, transactions (list of dicts via model_dump), finished_at
  - On failure: sets item.status="failed", error=str(exc)[:4000], finished_at
  - Commits, then calls `_refresh_run_status(db, run.id)` which recalculates parent AgentRun.status from children (all pending→pending, any running→running, all succeed→succeeded, mix→partial, all failed→failed); updates started_at/finished_at on the run.

- [ ] Step 3: Write `api/app/worker/tasks.py`:
```python
import asyncio, uuid
from api.app.worker.celery_app import celery_app
from api.app.worker.runner import run_item_async

@celery_app.task(name="agent.run_item", bind=True, max_retries=2)
def run_agent_item(self, agent_run_item_id: str) -> None:
    try: asyncio.run(run_item_async(uuid.UUID(agent_run_item_id)))
    except Exception as exc: raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

- [ ] Step 4: Write test `api/app/tests/test_worker_task.py` — use `task_always_eager=True` on celery_app in test; mock `agent_registry.get()` to return an adapter stub; verify task completes and runner sets status=succeeded; test runner failure path sets status=failed.

- [ ] Step 5: Commit:
```bash
git add api/app/worker/ api/app/tests/test_worker_task.py
git commit -m "feat(api): Celery worker (prefork, asyncio.run per task, status machine, retry)"
```

### Task 8: Agent runs API + wire everything

**Files:**
- Create: `api/app/schemas/agent_run.py` (full version)
- Create: `api/app/api/agent_runs.py`
- Modify: `api/app/main.py` (include agent_runs router; verify all 4 routers are included)
- Modify: `api/app/deps.py` (ensure get_current_user uses correct import path)
- Test: `api/app/tests/test_api_agent_runs.py`

**Interfaces:**
- Produces: `POST /api/agent-runs`, `GET /api/agent-runs`, `GET /api/agent-runs/{id}`, `POST /api/agent-runs/{id}/retry`, `DELETE /api/agent-runs/{id}`

**Steps:**

- [ ] Step 1: Write `api/app/schemas/agent_run.py` with AgentRunCreate (document_ids: list[UUID], agent: str="crewai", question: str, llm_provider_id: str, llm_model_id: str), AgentRunItemOut, AgentRunOut, AgentRunListResponse, RetryResponse classes.

- [ ] Step 2: Write `api/app/api/agent_runs.py`:
  - `POST /api/agent-runs` — validates body.agent via agent_registry.is_valid(), body.llm_provider_id+model_id via llm_registry.resolve(); enforces len(document_ids) >= 1; validates each document exists; creates AgentRun + N AgentRunItems; flushes; for each item: `task = run_agent_item.delay(str(item.id))` and stores celery_task_id; commits; returns 201 AgentRunOut (eager-loaded with items via selectinload)
  - `GET /api/agent-runs` — paginated; filters by owner_id, deleted_at IS NULL, optional status filter; order by created_at desc; returns AgentRunListResponse
  - `GET /api/agent-runs/{id}` — eagerly loads items; 404 if not found or not owner or soft-deleted; returns AgentRunOut
  - `POST /api/agent-runs/{id}/retry?item_ids[]=...` — finds items; validates each has status="failed" (409 otherwise); resets each item (status=pending, error=None, markdown_report=None, transactions=None, started_at/finished_at=None); enqueues new celery tasks; commits; returns RetryResponse
  - `DELETE /api/agent-runs/{id}` — revokes pending/running celery tasks (celery_app.control.revoke); sets AgentRun.deleted_at = now; commits; returns 204

- [ ] Step 3: Update `api/app/main.py` to include agent_runs_router. Ensure all 4 routers (auth, llm, documents, agent_runs) are registered via `app.include_router()`.

- [ ] Step 4: Write test `api/app/tests/test_api_agent_runs.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient

@pytest.mark.asyncio
async def test_create_rejects_empty_document_ids():
    from api.app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent-runs", json={"document_ids":[],"agent":"crewai","question":"q","llm_provider_id":"openai","llm_model_id":"openai/gpt-4o-mini"})
        assert resp.status_code == 422

@pytest.mark.asyncio
async def test_create_rejects_unknown_agent():
    from api.app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/agent-runs", json={"document_ids":["11111111-1111-1111-1111-111111111111"],"agent":"made-up","question":"q","llm_provider_id":"openai","llm_model_id":"openai/gpt-4o-mini"})
        assert resp.status_code == 422
```
Run: `cd api && python -m pytest app/tests/test_api_agent_runs.py -v`

- [ ] Step 5: Commit:
```bash
git add api/app/schemas/agent_run.py api/app/api/agent_runs.py api/app/tests/test_api_agent_runs.py api/app/main.py
git commit -m "feat(api): agent runs API (create batch + enqueue N tasks, list, detail, retry, soft delete)"
```

### Task 9: Dockerfiles + docker-compose dev profile

**Files:**
- Create: `infra/api.Dockerfile`, `infra/docker-compose.yml`, `infra/docker-compose.override.yml`, `infra/postgres-init.sql`, `infra/.env.example`

**Steps:**

- [ ] Step 1: Write `infra/api.Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt
COPY api/ /app/api/
COPY backend/ /app/backend/
COPY data/ /app/data/
ENV PYTHONPATH=/app:/app/backend
```

- [ ] Step 2: Write `infra/postgres-init.sql`: `CREATE EXTENSION IF NOT EXISTS pgcrypto;`

- [ ] Step 3: Write `infra/.env.example` with all env vars documented (POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, REDIS_URL, QDRANT_URL, QDRANT_API_KEY, JWT_SECRET, ADMIN_EMAIL, ADMIN_PASSWORD_HASH, LM_STUDIO_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, EMBEDDING_MODEL). Include comment documenting how to generate ADMIN_PASSWORD_HASH: `python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('your-password'))"`.

- [ ] Step 4: Write `infra/docker-compose.yml` with services: postgres (16-alpine, pgdata volume, healthcheck pg_isready), redis (7-alpine, healthcheck redis-cli ping), qdrant (v1.13.4, qdrant_data volume, healthcheck curl), api (build context=.., dockerfile=infra/api.Dockerfile, depends_on all 3 healthy, env vars, port 8000:8000, command runs alembic upgrade head then uvicorn api.app.main:app --host 0.0.0.0 --port 8000 [--reload in dev]), worker (same image, command celery -A api.app.worker.celery_app worker --loglevel=info --concurrency=2). All env vars use ${VAR:-default} syntax.

- [ ] Step 5: Write `infra/docker-compose.override.yml` adding source volume mounts for api and worker services: `../api:/app/api`, `../backend:/app/backend`, `../data:/app/data`.

- [ ] Step 6: Test docker-compose:
```bash
cd infra && cp .env.example .env && vim .env  # fill in JWT_SECRET + ADMIN_PASSWORD_HASH
docker compose up -d
sleep 10
curl http://localhost:8000/api/health | python -m json.tool
curl http://localhost:8000/api/llm-models | python -m json.tool | head -30
docker compose down
```

- [ ] Step 7: Commit:
```bash
git add infra/
git commit -m "feat(infra): api.Dockerfile, docker-compose dev profile (PG, Redis, Qdrant, API, Worker), env.example"

### Task 10: Web scaffold (Vite + React + TS + Tailwind + Router)

**Files:**
- Create: `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/tsconfig.node.json`, `web/postcss.config.js`, `web/tailwind.config.ts`, `web/index.html`, `web/eslint.config.js`
- Create: `web/src/vite-env.d.ts`, `web/src/main.tsx`, `web/src/App.tsx`, `web/src/router.tsx`
- Create: `web/src/index.css` (Tailwind directives), `web/src/types/index.ts`
- Create: `web/src/__tests__/` (empty + vitest setup)

**Steps:**

- [ ] Step 1: Write `web/package.json` with scripts (dev, build, test, lint), dependencies (react 18, react-dom, react-router-dom 6, @tanstack/react-query 5, zustand 5, react-markdown 9, remark-gfm 4), devDependencies (typescript 5, vite 5, @vitejs/plugin-react 4, tailwindcss 3, postcss, autoprefixer, vitest 2, @testing-library/react 16, jsdom 25).

- [ ] Step 2: Write `web/vite.config.ts` — react plugin, dev server port 5173, proxy `/api` to `http://api:8000`.

- [ ] Step 3: Write `web/tailwind.config.ts` — content scanning `./index.html` + `./src/**/*.{ts,tsx}`, theme extend empty.

- [ ] Step 4: Write `web/index.html` — standard Vite HTML5 template with `<div id="root"></div>`, body classes `bg-gray-50 text-gray-900`.

- [ ] Step 5: Write `web/src/index.css` — `@tailwind base; @tailwind components; @tailwind utilities;`.

- [ ] Step 6: Write `web/src/types/index.ts` — all TypeScript interfaces mirroring the API schemas: UserOut, AgentInfo, LLMModel, LLMProvider, DocumentOut, Transaction, AgentRunItemOut, AgentRunOut, AgentStatus type.

- [ ] Step 7: Write `web/src/main.tsx` — creates QueryClientProvider wrapping BrowserRouter wrapping App, renders to root.

- [ ] Step 8: Write `web/src/router.tsx` — Routes: /login -> LoginPage, /batches -> BatchListPage (protected), /batches/new -> NewBatchPage (protected), /batches/:id -> BatchDetailPage (protected), * -> Navigate to /batches.

- [ ] Step 9: Write `web/src/App.tsx` — just returns `<Router />`.

- [ ] Step 10: Run `cd web && npm install && npm run dev` — verify Vite starts on :5173.

- [ ] Step 11: Commit:
```bash
git add web/
git commit -m "feat(web): React+Vite+TS+Tailwind+Router+TanStack Query scaffold"
```

### Task 11: Auth store + API client + Login + ProtectedRoute + Layout

**Files:**
- Create: `web/src/stores/authStore.ts`, `web/src/api/client.ts`, `web/src/api/auth.ts`
- Create: `web/src/hooks/useAuth.ts`
- Create: `web/src/components/ProtectedRoute.tsx`, `web/src/components/Layout.tsx`
- Create: `web/src/pages/LoginPage.tsx`

**Steps:**

- [ ] Step 1: Write `web/src/stores/authStore.ts` — zustand store with `token`, `user`, `login(token, user)`, `logout()`; persists token to localStorage.

- [ ] Step 2: Write `web/src/api/client.ts` — `api.get<T>(path)`, `api.post<T>(path, body?)`, `api.delete<T>(path)`; sets Authorization header from localStorage token; on 401 redirects to /login.

- [ ] Step 3: Write `web/src/api/auth.ts` — `login(email, password) -> {access_token, user}` and `me() -> UserOut`.

- [ ] Step 4: Write `web/src/hooks/useAuth.ts` — useEffect that calls `me()` if token exists but user is null; exposes `{isAuthenticated, user, login, logout}`.

- [ ] Step 5: Write `web/src/pages/LoginPage.tsx` — form with email/password fields; calls authApi.login; on success navigates to /batches; shows error on failure; if already authenticated, redirects to /batches.

- [ ] Step 6: Write `web/src/components/ProtectedRoute.tsx` — if !isAuthenticated, <Navigate to="/login">; else <Outlet />.

- [ ] Step 7: Write `web/src/components/Layout.tsx` — nav bar with "BankAI" logo link, "New Batch" link, user email, logout button; renders <Outlet /> inside a max-w-6xl container.

- [ ] Step 8: Commit:
```bash
git add web/src/stores/ web/src/api/client.ts web/src/api/auth.ts web/src/hooks/ web/src/components/Layout.tsx web/src/components/ProtectedRoute.tsx web/src/pages/LoginPage.tsx
git commit -m "feat(web): auth store, API client, LoginPage, ProtectedRoute, Layout"
```

### Task 12: Agent/Model dropdowns + PDF dropzone + NewBatchPage

**Files:**
- Create: `web/src/api/llm.ts`, `web/src/api/documents.ts`, `web/src/api/agentRuns.ts`
- Create: `web/src/hooks/useLLMModels.ts`
- Create: `web/src/components/AgentDropdown.tsx`, `web/src/components/ModelDropdown.tsx`, `web/src/components/PdfMultiDropzone.tsx`
- Create: `web/src/pages/NewBatchPage.tsx`
- Test: `web/src/__tests__/AgentDropdown.test.tsx`, `web/src/__tests__/ModelDropdown.test.tsx`

**Steps:**

- [ ] Step 1: Write `web/src/api/llm.ts` — `listLLMModels()` returns `{providers: LLMProvider[]}`.

- [ ] Step 2: Write `web/src/api/documents.ts` — `uploadDocument(file: File)` POST /api/documents with FormData; `listDocuments(limit, offset)` GET /api/documents.

- [ ] Step 3: Write `web/src/api/agentRuns.ts` — `createAgentRun(body)` POST /api/agent-runs; `listAgentRuns(limit, offset, status?)` GET; `getAgentRun(id)` GET; `retryAgentRunItems(id, item_ids?)` POST; `deleteAgentRun(id)` DELETE.

- [ ] Step 4: Write `web/src/hooks/useLLMModels.ts` — useQuery with key ["llm-models"], refetchInterval 60000.

- [ ] Step 5: Write `web/src/components/AgentDropdown.tsx` — select with hardcoded options: crewai (enabled), deep-agents (disabled, "coming soon"), hermes (disabled, "coming soon"); props: value, onChange.

- [ ] Step 6: Write `web/src/components/ModelDropdown.tsx` — two side-by-side selects (provider + model); fetches via useLLMModels; disabled providers show unavailable_reason; props: providerId, modelId, onProviderChange(providerId, modelId).

- [ ] Step 7: Write `web/src/components/PdfMultiDropzone.tsx` — drag-and-drop zone + hidden file input (accept=.pdf, multiple); on drop/select, calls uploadDocument per file; shows uploaded files list with "deduped" badge; remove button per file; loading state during upload; props: selectedDocs, onDocsChange.

- [ ] Step 8: Write `web/src/pages/NewBatchPage.tsx`:
  - State: selectedDocs (DocumentOut[]), question (string), agent ("crewai"), providerId (from localStorage default or first available), modelId (from localStorage default or first model of provider)
  - Renders: `<h1>New Batch</h1>`, PdfMultiDropzone, div with "Agent" label + AgentDropdown, div with "Model" label + ModelDropdown, textarea for question, Submit button
  - On submit: calls createAgentRun with document_ids, agent, question, llm_provider_id, llm_model_id; saves providerId+modelId to localStorage as default; on success navigates to /batches/{id}
  - Loading + error states

- [ ] Step 9: Write vitest tests — `AgentDropdown.test.tsx` verifies it renders 3 options with correct enabled/disabled states; `ModelDropdown.test.tsx` mocks useLLMModels to return fake providers and verifies select options render.

- [ ] Step 10: Commit:
```bash
git add web/src/api/ web/src/hooks/web/src/components/ web/src/pages/ web/src/__tests__/
git commit -m "feat(web): agent/model dropdowns, PDF dropzone, NewBatchPage with localStorage defaults"
```

### Task 13: BatchListPage + BatchDetailPage + polling + MarkdownViewer + TransactionsTable

**Files:**
- Create: `web/src/hooks/useBatchPolling.ts`
- Create: `web/src/components/BatchItemCard.tsx`, `web/src/components/MarkdownViewer.tsx`, `web/src/components/TransactionsTable.tsx`
- Create: `web/src/pages/BatchListPage.tsx`, `web/src/pages/BatchDetailPage.tsx`
- Test: `web/src/__tests__/BatchItemCard.test.tsx`, `web/src/__tests__/useBatchPolling.test.ts`

**Steps:**

- [ ] Step 1: Write `web/src/pages/BatchListPage.tsx` — useQuery ["agent-runs"] fetching `listAgentRuns()`; renders table with columns: Created, Agent, Model, Status, Items (count), Actions (View link); pagination; link to /batches/new.

- [ ] Step 2: Write `web/src/hooks/useBatchPolling.ts` — useQuery with key ["agent-run", id], enabled=true, refetchInterval=2000 while status is pending/running; exposes `{data, isLoading, error}`.

- [ ] Step 3: Write `web/src/components/BatchItemCard.tsx` — card per AgentRunItemOut: shows document filename (fetched from documents API or passed via parent), status badge (color-coded: pending=gray, running=blue+spinner, succeeded=green, failed=red), error text if failed, expandable section with MarkdownViewer + TransactionsTable, retry button if failed.

- [ ] Step 4: Write `web/src/components/MarkdownViewer.tsx` — uses `react-markdown` + `remark-gfm` to render markdown_report string; prosified div.

- [ ] Step 5: Write `web/src/components/TransactionsTable.tsx` — renders transactions JSONB array as a sortable HTML table with columns Date, Description, Credit, Debit, Balance, Currency.

- [ ] Step 6: Write `web/src/pages/BatchDetailPage.tsx` — gets batch id from useParams; uses useBatchPolling to fetch batch; displays batch info header (agent, model, question, status); maps over items -> BatchItemCard; retry button per failed item (calls retryAgentRunItems).

- [ ] Step 7: Write vitest tests: `useBatchPolling.test.ts` verifies polling stops on succeeded status; `BatchItemCard.test.tsx` renders pending, running, succeeded, failed variants correctly.

- [ ] Step 8: Commit:
```bash
git add web/src/hooks/useBatchPolling.ts web/src/components/BatchItemCard.tsx web/src/components/MarkdownViewer.tsx web/src/components/TransactionsTable.tsx web/src/pages/BatchListPage.tsx web/src/pages/BatchDetailPage.tsx web/src/__tests__/
git commit -m "feat(web): BatchListPage, BatchDetailPage with polling, MarkdownViewer, TransactionsTable"
```

### Task 14: Nginx prod config + web Dockerfile + compose prod profile

**Files:**
- Create: `web/nginx.conf`, `web/Dockerfile`
- Modify: `infra/docker-compose.yml` (add web service under prod profile)
- Create: `infra/k8s/README.md` (stub)

**Steps:**

- [ ] Step 1: Write `web/nginx.conf`:
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://api:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

- [ ] Step 2: Write `web/Dockerfile`:
```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

- [ ] Step 3: Update `infra/docker-compose.yml` — add `web` service under `profiles: [prod]` that builds from `web/Dockerfile` (context: ../web), depends_on api, port 80:80.

- [ ] Step 4: Write `infra/k8s/README.md` — "Kubernetes / MicroK8s manifests deferred to post-v1 implementation. See spec at docs/superpowers/specs/2026-07-25-full-stack-app-design.md."

- [ ] Step 5: Test prod profile:
```bash
cd infra && docker compose --profile prod build && docker compose --profile prod up -d
sleep 15
curl http://localhost:80/  # should return index.html
docker compose --profile prod down
```

- [ ] Step 6: Commit:
```bash
git add web/nginx.conf web/Dockerfile infra/docker-compose.yml infra/k8s/
git commit -m "feat(infra): nginx prod config, web Dockerfile multi-stage build, compose prod profile"
```

### Task 15: README updates + final verification smoke

**Files:**
- Modify: `README.md` (add v1 full-stack quickstart section)
- Modify: `.gitignore` (add missing entries from Task 1 scaffolding if not already present)

**Steps:**

- [ ] Step 1: Update `.gitignore` with entries: `api/__pycache__/`, `api/.venv/`, `api/app/core/*.db`, `web/node_modules/`, `web/dist/`, `infra/.env`, `data/uploads/`.

- [ ] Step 2: Update root `README.md` — add a "## Full-Stack App (v1)" section between "Quick Start" and "Roadmap", documenting: `cp infra/.env.example infra/.env` + fill secrets, `cd infra && docker compose up -d` (dev profile), `cd web && npm install && npm run dev` (frontend dev), link to the spec and plan docs.

- [ ] Step 3: Full end-to-end smoke test:
  - `docker compose up -d` (from infra, with .env populated)
  - Wait for healthy
  - `curl http://localhost:8000/api/health` → all ok
  - `curl http://localhost:8000/api/auth/login -H "Content-Type: application/json" -d '{"email":"admin@bankai.local","password":"your-pass"}'` → returns token
  - `curl -X POST http://localhost:8000/api/documents -F "file=@../data/bank-statement-document/Dummy-Bank-Statement.pdf" -H "Authorization: Bearer <token>"` → returns DocumentOut
  - `curl -X POST http://localhost:8000/api/agent-runs -H "Content-Type: application/json" -H "Authorization: Bearer <token>" -d '{"document_ids":["<doc_id>"],"agent":"crewai","question":"What are total debits?","llm_provider_id":"lm-studio","llm_model_id":"openai/qwen2.5-14b-instruct"}'` → returns 201 with batch
  - `curl http://localhost:8000/api/agent-runs/<batch_id> -H "Authorization: Bearer <token>"` → status updates
  - Navigate to `http://localhost:5173` in browser → login → batches list → new batch → upload PDF → submit → view results
  - Soft delete batch: `curl -X DELETE http://localhost:8000/api/agent-runs/<batch_id> -H "Authorization: Bearer <token>"` → 204
  - Soft delete document: same pattern → 204 (unless referenced by running items)

- [ ] Step 4: Commit:
```bash
git add README.md .gitignore
git commit -m "docs: full-stack v1 quickstart in README, final .gitignore updates"
```
```
