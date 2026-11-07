from fastapi import APIRouter, Depends

from app.config import settings
from app.deps import get_current_user
from app.models import User

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/grafana")
async def grafana_link(_: User = Depends(get_current_user)):
    return {
        "url": settings.GRAFANA_URL,
        "dashboard": "/d/bankai-local-llm",
        "title": "BankAI Local LLM (GPU / CPU / RAM)",
    }
