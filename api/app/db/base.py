import uuid
from sqlalchemy import MetaData
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

convention = {"ix":"ix_%(column_0_label)s","uq":"uq_%(table_name)s_%(column_0_name)s","ck":"ck_%(table_name)s_%(constraint_name)s","fk":"fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s","pk":"pk_%(table_name)s"}
metadata = MetaData(naming_convention=convention)

class Base(DeclarativeBase):
    metadata = metadata
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
