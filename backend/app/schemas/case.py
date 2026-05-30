from datetime import datetime

from pydantic import BaseModel


class CaseCreate(BaseModel):
    patient_id: str
    notes: str | None = None


class CaseResponse(BaseModel):
    id: str
    patient_id: str
    filename: str
    slice_count: int | None
    voxel_spacing: str | None
    shape: str | None
    notes: str | None
    upload_time: datetime
    status: str

    model_config = {"from_attributes": True}
