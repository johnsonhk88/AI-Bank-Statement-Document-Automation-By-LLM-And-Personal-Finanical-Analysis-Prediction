import datetime, shutil, uuid
from pathlib import Path
from fastapi import UploadFile
from app.config import settings

CHUNK_SIZE = 1024 * 256  # 256 KB chunks
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file


def save_upload(file: UploadFile, owner_id: uuid.UUID) -> str:
    now = datetime.datetime.now(datetime.UTC)
    subdir = settings.UPLOAD_ROOT / str(now.year) / f"{now.month:02d}"
    subdir.mkdir(parents=True, exist_ok=True)
    doc_id = uuid.uuid4()
    dest = subdir / f"{doc_id}.pdf"
    written = 0
    with open(dest, "wb") as f:
        while True:
            chunk = file.file.read(CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise ValueError(f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
            f.write(chunk)
    return str(dest.relative_to(settings.UPLOAD_ROOT.parent))
