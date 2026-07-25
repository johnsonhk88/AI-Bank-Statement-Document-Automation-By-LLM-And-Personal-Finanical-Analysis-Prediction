import datetime
from pydantic import BaseModel

class DocumentOut(BaseModel):
    id: str
    original_filename: str
    mime_type: str
    size_bytes: int | None = None
    page_count: int | None = None
    deduplicated: bool = False
    created_at: datetime.datetime
    model_config = {"from_attributes": True}

class DocumentListResponse(BaseModel):
    items: list[DocumentOut]
    total: int
