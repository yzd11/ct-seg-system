import json
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.case import Case
from app.services.nifti_service import get_slice_png, get_volume_metadata

router = APIRouter(prefix="/nifti", tags=["nifti"])


@router.get("/{case_id}/metadata")
async def nifti_metadata(case_id: str, db: AsyncSession = Depends(get_db)):
    case = await db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return {
        "slice_count": case.slice_count,
        "voxel_spacing": json.loads(case.voxel_spacing) if case.voxel_spacing else None,
        "shape": json.loads(case.shape) if case.shape else None,
    }


@router.get("/{case_id}/slice/{idx}")
async def nifti_slice(
    case_id: str,
    idx: int,
    center: int = 50,
    width: int = 400,
    db: AsyncSession = Depends(get_db),
):
    case = await db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    if case.slice_count and (idx < 0 or idx >= case.slice_count):
        raise HTTPException(400, f"Slice index out of range [0, {case.slice_count})")

    png_bytes = get_slice_png(case.filepath, idx, center, width)
    return Response(content=png_bytes, media_type="image/png")
