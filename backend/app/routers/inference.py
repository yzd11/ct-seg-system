import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.inference_job import InferenceJob
from app.models.slice_result import SliceResult
from app.schemas.inference import JobCreate, JobResponse, SliceResultResponse

router = APIRouter(prefix="/inference", tags=["inference"])

VALID_MODELS = {"unet", "resunet", "unet_pp", "att_unet_pp"}


@router.post("/jobs", response_model=JobResponse, status_code=201)
async def submit_job(body: JobCreate, db: AsyncSession = Depends(get_db)):
    if body.model_name not in VALID_MODELS:
        raise HTTPException(400, f"Unknown model '{body.model_name}'. Valid: {sorted(VALID_MODELS)}")

    # Verify case exists
    from app.models.case import Case
    case = await db.get(Case, body.case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    # Import Celery task lazily to avoid circular import
    from app.tasks.inference_task import run_inference

    job_id = run_inference.delay(body.case_id, body.model_name).id

    job = InferenceJob(
        id=job_id,
        case_id=body.case_id,
        model_name=body.model_name,
        status="queued",
        total_slices=case.slice_count or 0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(InferenceJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/results", response_model=list[SliceResultResponse])
async def job_results(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SliceResult)
        .where(SliceResult.job_id == job_id)
        .order_by(SliceResult.slice_index)
    )
    return result.scalars().all()


@router.get("/jobs/{job_id}/mask/{idx}")
async def job_mask(job_id: str, idx: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(InferenceJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    mask_path = settings.upload_dir / job.case_id / "results" / job_id / "masks" / f"{idx:04d}.png"
    if not mask_path.exists():
        raise HTTPException(404, "Mask not ready yet")

    return Response(content=mask_path.read_bytes(), media_type="image/png")


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(InferenceJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    # Cancel if still running
    if job.status in ("queued", "running"):
        from app.tasks.celery_app import celery_app
        celery_app.control.revoke(job_id, terminate=True)

    # Delete mask files from disk
    mask_dir = settings.upload_dir / job.case_id / "results" / job_id
    if mask_dir.exists():
        shutil.rmtree(mask_dir)

    # Delete SliceResults then InferenceJob
    await db.execute(delete(SliceResult).where(SliceResult.job_id == job_id))
    await db.delete(job)
    await db.commit()


@router.get("/cases/{case_id}/jobs", response_model=list[JobResponse])
async def case_jobs(case_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(InferenceJob)
        .where(InferenceJob.case_id == case_id)
        .order_by(InferenceJob.created_at.desc())
    )
    return result.scalars().all()
