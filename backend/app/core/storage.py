import datetime, uuid
from pathlib import Path
from fastapi import UploadFile
from app.config import settings

def save_upload(file: UploadFile, owner_id: uuid.UUID) -> str:
    now = datetime.datetime.now(datetime.UTC)
    subdir = settings.UPLOAD_ROOT / str(now.year) / f"{now.month:02d}"
    subdir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    dest = subdir / f"{doc_id}.pdf"
    with open(dest, "wb") as f:
        f.write(file.file.read())
    return str(dest.relative_to(settings.UPLOAD_ROOT.parent))
