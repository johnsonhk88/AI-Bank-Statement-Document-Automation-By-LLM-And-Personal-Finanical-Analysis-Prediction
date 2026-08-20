import httpx
import os
import yaml
from pathlib import Path
from pydantic import BaseModel


class ModelConfig(BaseModel):
    id: str
    display_name: str


class ProviderConfig(BaseModel):
    id: str
    display_name: str
    kind: str
    base_url: str = ""
    api_key_env: str = ""
    models: list[ModelConfig]


class LLMCatalog(BaseModel):
    providers: list[ProviderConfig]


class LLMProviderRegistry:
    def __init__(self, catalog_path: Path | None = None):
        if catalog_path is None:
            catalog_path = Path(__file__).resolve().parent.parent.parent / "config" / "llm_providers.yaml"
        self._catalog = LLMCatalog(**yaml.safe_load(catalog_path.read_text()))

    def resolve(self, provider_id: str, model_id: str) -> tuple[str, str, str]:
        for p in self._catalog.providers:
            if p.id == provider_id:
                for m in p.models:
                    if m.id == model_id:
                        default_key = "no-key" if p.kind == "local" else ""
                        key = os.environ.get(p.api_key_env, default_key) if p.api_key_env else default_key
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
                try:
                    r = httpx.get(p.base_url + "/models" if p.base_url else p.base_url, timeout=3)
                    avail = r.is_success
                    reason = None if avail else f"HTTP {r.status_code}"
                except Exception as e:
                    avail = False
                    reason = str(e)
            else:
                if p.api_key_env and not os.environ.get(p.api_key_env):
                    avail = False
                    reason = f"{p.api_key_env} not set"
            result.append({
                "id": p.id,
                "display_name": p.display_name,
                "kind": p.kind,
                "available": avail,
                "unavailable_reason": reason,
                "models": [{"id": m.id, "display_name": m.display_name} for m in p.models],
            })
        return result
