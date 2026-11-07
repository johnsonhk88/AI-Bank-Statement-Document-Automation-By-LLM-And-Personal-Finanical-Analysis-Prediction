from pydantic import BaseModel


class LLMModelOut(BaseModel):
    id: str
    display_name: str


class LLMProviderOut(BaseModel):
    id: str
    display_name: str
    kind: str
    available: bool
    unavailable_reason: str | None = None
    models: list[LLMModelOut]


class LLMCatalogResponse(BaseModel):
    providers: list[LLMProviderOut]
