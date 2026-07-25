from fastapi import APIRouter
from app.agents.llm_provider import LLMProviderRegistry
from app.schemas.llm import LLMCatalogResponse, LLMModelOut, LLMProviderOut

router = APIRouter(prefix="/api/llm-models", tags=["llm"])
_registry: LLMProviderRegistry | None = None


def get_llm_registry() -> LLMProviderRegistry:
    global _registry
    if _registry is None:
        _registry = LLMProviderRegistry()
    return _registry


@router.get("", response_model=LLMCatalogResponse)
async def list_llm_models():
    providers = get_llm_registry().list_providers()
    return LLMCatalogResponse(providers=[
        LLMProviderOut(
            id=p["id"],
            display_name=p["display_name"],
            kind=p["kind"],
            available=p["available"],
            unavailable_reason=p["unavailable_reason"],
            models=[LLMModelOut(id=m["id"], display_name=m["display_name"]) for m in p["models"]],
        )
        for p in providers
    ])
