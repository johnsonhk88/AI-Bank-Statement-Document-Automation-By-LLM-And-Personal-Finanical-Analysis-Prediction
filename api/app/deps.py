from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_session as _get_session
from app.core.security import decode_access_token
from app.models import User

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
