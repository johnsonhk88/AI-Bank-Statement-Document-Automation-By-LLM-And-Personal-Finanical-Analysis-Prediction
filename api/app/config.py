from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_ENV: str = "dev"
    POSTGRES_USER: str = "bankai"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "bankai"
    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    JWT_SECRET: str = ""
    JWT_ALG: str = "HS256"
    JWT_TTL_MINUTES: int = 15
    ADMIN_EMAIL: str = ""
    ADMIN_PASSWORD_HASH: str = ""
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    UPLOAD_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "uploads"
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
