from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SliceResult(Base):
    __tablename__ = "slice_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String, ForeignKey("inference_jobs.id"))
    slice_index: Mapped[int] = mapped_column(Integer, nullable=False)
    liver_area_px: Mapped[int] = mapped_column(Integer, default=0)
    tumor_area_px: Mapped[int] = mapped_column(Integer, default=0)
    liver_perimeter_px: Mapped[int] = mapped_column(Integer, default=0, nullable=True)
    tumor_perimeter_px: Mapped[int] = mapped_column(Integer, default=0, nullable=True)

    __table_args__ = (
        Index("idx_slice_results", "job_id", "slice_index"),
    )
