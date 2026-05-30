import json
import re
import shutil
import uuid
from pathlib import Path

import nibabel as nib
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.case import Case
from app.schemas.case import CaseResponse

router = APIRouter(prefix="/cases", tags=["cases"])

_MAX_UPLOAD_BYTES = 600 * 1024 * 1024          # 600 MB
_PATIENT_ID_RE   = re.compile(r"^[\w\-]{1,64}$")

# NIfTI-1 magic: bytes 344-347 == b"ni1\x00" or b"n+1\x00"
# NIfTI-2 magic: bytes 4-7    == b"ni2\x00" or b"n+2\x00"
def _is_nifti(header_bytes: bytes) -> bool:
    if len(header_bytes) < 348:
        return False
    magic_nii1 = header_bytes[344:348]
    magic_nii2 = header_bytes[4:8]
    return magic_nii1 in (b"ni1\x00", b"n+1\x00") or magic_nii2 in (b"ni2\x00", b"n+2\x00")


@router.post("/", response_model=CaseResponse, status_code=201)
async def upload_case(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    notes: str = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    # ── 患者 ID 格式校验 ──────────────────────────────────────────────────────
    if not _PATIENT_ID_RE.match(patient_id):
        raise HTTPException(400, "patient_id 只能包含字母、数字、下划线或连字符，长度 1-64")

    # ── 扩展名校验 ────────────────────────────────────────────────────────────
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Only .nii or .nii.gz files are supported")

    # ── 读取文件内容（含大小限制）────────────────────────────────────────────
    raw = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件超过 {_MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限")

    # ── NIfTI magic bytes 校验 ────────────────────────────────────────────────
    if not _is_nifti(raw):
        raise HTTPException(400, "文件内容不符合 NIfTI 格式（magic bytes 校验失败）")

    case_id = str(uuid.uuid4())
    case_dir = settings.upload_dir / case_id
    case_dir.mkdir(parents=True)

    # Preserve original extension so nibabel can detect format correctly
    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    dest = case_dir / f"original{suffix}"
    with dest.open("wb") as f:
        f.write(raw)

    # Parse NIfTI header (memmap, no data loading)
    try:
        nii = nib.load(str(dest))
        shape = list(nii.shape[:3])
        spacing = [float(s) for s in nii.header.get_zooms()[:3]]
        slice_count = shape[2] if len(shape) >= 3 else None
    except Exception:
        shutil.rmtree(case_dir)
        raise HTTPException(422, "Failed to parse NIfTI header")

    case = Case(
        id=case_id,
        patient_id=patient_id,
        filename=file.filename,
        filepath=str(dest),
        slice_count=slice_count,
        voxel_spacing=json.dumps(spacing),
        shape=json.dumps(shape),
        notes=notes or None,
        status="ready",
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)
    return case


@router.get("/", response_model=list[CaseResponse])
async def list_cases(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Case).order_by(Case.upload_time.desc()))
    return result.scalars().all()


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str, db: AsyncSession = Depends(get_db)):
    case = await db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@router.delete("/{case_id}", status_code=204)
async def delete_case(case_id: str, db: AsyncSession = Depends(get_db)):
    case = await db.get(Case, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    case_dir = settings.upload_dir / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    await db.delete(case)
    await db.commit()
