import os
from pathlib import Path

import httpx
import yaml
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
    litellm_prefix: str = ""
    models: list[ModelConfig]


class LLMCatalog(BaseModel):
    providers: list[ProviderConfig]


def _models_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/models"


def _litellm_id(raw_id: str, prefix: str) -> str:
    if not prefix or "/" in raw_id:
        return raw_id
    return f"{prefix}/{raw_id}"


class LLMProviderRegistry:
    def __init__(self, catalog_path: Path | None = None):
        if catalog_path is None:
            catalog_path = Path(__file__).resolve().parent.parent.parent / "config" / "llm_providers.yaml"
        self._catalog = LLMCatalog(**yaml.safe_load(catalog_path.read_text()))

    def _api_key(self, p: ProviderConfig) -> str:
        if not p.api_key_env:
            return "none" if p.kind == "local" else ""
        fallback = "none" if p.kind == "local" else ""
        return os.environ.get(p.api_key_env, fallback)

    def resolve(self, provider_id: str, model_id: str) -> tuple[str, str, str]:
        for p in self._catalog.providers:
            if p.id != provider_id:
                continue
            key = self._api_key(p)
            if p.api_key_env and not key and p.kind != "local":
                raise ValueError(f"API key not set: {p.api_key_env}")
            for m in p.models:
                if m.id == model_id:
                    return m.id, p.base_url or "", key
            if p.kind == "local":
                return _litellm_id(model_id, p.litellm_prefix), p.base_url or "", key
            raise ValueError(f"Model {model_id} not found in provider {provider_id}")
        raise ValueError(f"Provider {provider_id} not found")

    def _live_models(self, p: ProviderConfig) -> list[dict] | None:
        url = _models_url(p.base_url)
        if not url:
            return None
        headers = {}
        key = self._api_key(p)
        if key and key not in ("none", "lm-studio"):
            headers["Authorization"] = f"Bearer {key}"
        try:
            r = httpx.get(url, headers=headers, timeout=3)
            if not r.is_success:
                return None
            payload = r.json()
        except Exception:
            return None
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return None
        models = []
        for row in rows:
            if isinstance(row, str):
                raw = row
            elif isinstance(row, dict):
                raw = row.get("id") or row.get("name")
            else:
                continue
            if not raw:
                continue
            litellm_id = _litellm_id(str(raw), p.litellm_prefix)
            models.append({"id": litellm_id, "display_name": str(raw)})
        return models or None

    def list_providers(self) -> list[dict]:
        result = []
        for p in self._catalog.providers:
            avail, reason = True, None
            live = None
            if p.kind == "local":
                url = _models_url(p.base_url)
                try:
                    headers = {}
                    key = self._api_key(p)
                    if key and key not in ("none", "lm-studio"):
                        headers["Authorization"] = f"Bearer {key}"
                    r = httpx.get(url, headers=headers, timeout=3) if url else None
                    if r is None:
                        avail, reason = False, "no base_url"
                    else:
                        avail = r.is_success
                        reason = None if avail else f"HTTP {r.status_code}"
                        if avail:
                            live = self._live_models(p)
                except Exception as e:
                    avail = False
                    reason = str(e)
            else:
                if p.api_key_env and not os.environ.get(p.api_key_env):
                    avail = False
                    reason = f"{p.api_key_env} not set"
            catalog_models = [{"id": m.id, "display_name": m.display_name} for m in p.models]
            result.append({
                "id": p.id,
                "display_name": p.display_name,
                "kind": p.kind,
                "available": avail,
                "unavailable_reason": reason,
                "models": live or catalog_models,
            })
        return result
