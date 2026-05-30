from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    slice_count: Mapped[int | None] = mapped_column(Integer)
    voxel_spacing: Mapped[str | None] = mapped_column(String)   # JSON "[sx, sy, sz]"
    shape: Mapped[str | None] = mapped_column(String)           # JSON "[x, y, z]"
    notes: Mapped[str | None] = mapped_column(Text)
    upload_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    status: Mapped[str] = mapped_column(String, default="ready")
