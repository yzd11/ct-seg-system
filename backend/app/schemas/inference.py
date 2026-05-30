from datetime import datetime

from pydantic import BaseModel


class JobCreate(BaseModel):
    model_config = {"protected_namespaces": ()}

    case_id: str
    model_name: str


class JobResponse(BaseModel):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    id: str
    case_id: str
    model_name: str
    status: str
    progress: int
    current_slice: int
    total_slices: int
    liver_volume_ml: float | None
    tumor_volume_ml: float | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

class SliceResultResponse(BaseModel):
    slice_index: int
    liver_area_px: int
    tumor_area_px: int
    liver_perimeter_px: int | None = None
    tumor_perimeter_px: int | None = None

    model_config = {"from_attributes": True}
