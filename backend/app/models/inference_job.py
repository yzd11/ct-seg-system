from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)              # UUID = Celery task_id
    case_id: Mapped[str] = mapped_column(String, ForeignKey("cases.id"))
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="queued")          # queued|running|done|failed|cancelled
    progress: Mapped[int] = mapped_column(Integer, default=0)              # 0-100
    current_slice: Mapped[int] = mapped_column(Integer, default=0)
    total_slices: Mapped[int] = mapped_column(Integer, default=0)
    liver_volume_ml: Mapped[float | None] = mapped_column(Float)
    tumor_volume_ml: Mapped[float | None] = mapped_column(Float)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
