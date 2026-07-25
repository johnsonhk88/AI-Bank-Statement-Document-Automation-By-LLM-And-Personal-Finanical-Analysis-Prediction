from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.security import create_access_token, decode_access_token, verify_password
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse, UserOut
from app.deps import get_db, get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])
bearer = HTTPBearer()

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, detail={"error":{"code":"UNAUTHORIZED","message":"Invalid credentials"}})
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))

@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)
