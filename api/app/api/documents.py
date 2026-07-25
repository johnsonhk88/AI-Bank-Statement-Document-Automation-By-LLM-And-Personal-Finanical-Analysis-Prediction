import datetime, os
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.hashing import sha256_file
from app.core.storage import save_upload
from app.config import settings
from app.deps import get_db, get_current_user
from app.models import Document, AgentRunItem, User
from app.schemas.document import DocumentOut, DocumentListResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(422, detail={"error": {"code": "VALIDATION", "message": "Only PDF files are accepted"}})

    storage_path = save_upload(file, current_user.id)
    full_path = settings.UPLOAD_ROOT.parent / storage_path
    sha256 = sha256_file(str(full_path))
    size_bytes = full_path.stat().st_size

    existing = await db.execute(
        select(Document).where(Document.content_sha256 == sha256, Document.deleted_at.is_(None))
    )
    dup = existing.scalar_one_or_none()
    if dup:
        os.remove(str(full_path))
        return DocumentOut(
            id=str(dup.id),
            original_filename=dup.original_filename,
            mime_type=dup.mime_type,
            size_bytes=dup.size_bytes,
            page_count=dup.page_count,
            deduplicated=True,
            created_at=dup.created_at,
        )

    doc = Document(
        owner_id=current_user.id,
        original_filename=file.filename,
        storage_path=storage_path,
        content_sha256=sha256,
        mime_type=file.content_type or "application/pdf",
        size_bytes=size_bytes,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return DocumentOut(
        id=str(doc.id),
        original_filename=doc.original_filename,
        mime_type=doc.mime_type,
        size_bytes=doc.size_bytes,
        page_count=doc.page_count,
        deduplicated=False,
        created_at=doc.created_at,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    where = [Document.owner_id == current_user.id, Document.deleted_at.is_(None)]
    count_query = select(func.count(Document.id)).where(*where)
    total = (await db.execute(count_query)).scalar() or 0

    result = await db.execute(
        select(Document).where(*where).order_by(Document.created_at.desc()).offset(offset).limit(limit)
    )
    rows = result.scalars().all()
    items = [
        DocumentOut(
            id=str(r.id),
            original_filename=r.original_filename,
            mime_type=r.mime_type,
            size_bytes=r.size_bytes,
            page_count=r.page_count,
            deduplicated=False,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return DocumentListResponse(items=items, total=total)


@router.get("/{id}", response_model=DocumentOut)
async def get_document(
    id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == id, Document.deleted_at.is_(None), Document.owner_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Document not found"}})
    return DocumentOut(
        id=str(doc.id),
        original_filename=doc.original_filename,
        mime_type=doc.mime_type,
        size_bytes=doc.size_bytes,
        page_count=doc.page_count,
        deduplicated=False,
        created_at=doc.created_at,
    )


@router.get("/{id}/content")
async def get_document_content(
    id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == id, Document.deleted_at.is_(None), Document.owner_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Document not found"}})
    full_path = settings.UPLOAD_ROOT.parent / doc.storage_path
    if not full_path.exists():
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "File not found on disk"}})
    return FileResponse(str(full_path), media_type=doc.mime_type, filename=doc.original_filename)


@router.delete("/{id}", status_code=204)
async def delete_document(
    id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Document).where(Document.id == id, Document.deleted_at.is_(None), Document.owner_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, detail={"error": {"code": "NOT_FOUND", "message": "Document not found"}})

    ref = await db.execute(
        select(AgentRunItem).where(
            AgentRunItem.document_id == doc.id,
            AgentRunItem.status.in_(["pending", "running"]),
        )
    )
    if ref.scalars().first():
        raise HTTPException(
            409,
            detail={
                "error": {
                    "code": "CONFLICT",
                    "message": "Document is referenced by a pending or running agent run",
                }
            },
        )

    doc.deleted_at = datetime.datetime.now(datetime.UTC)
    await db.commit()
