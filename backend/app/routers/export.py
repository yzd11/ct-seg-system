from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.inference_job import InferenceJob
from app.models.slice_result import SliceResult
from app.services.export_service import generate_pdf

router = APIRouter(prefix="/export", tags=["export"])


@router.post("/pdf")
async def export_pdf(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(InferenceJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(400, "Inference not completed yet")

    from app.models.case import Case
    case = await db.get(Case, job.case_id)

    # Fetch all slice results for statistics
    result = await db.execute(
        select(SliceResult)
        .where(SliceResult.job_id == job_id)
        .order_by(SliceResult.slice_index)
    )
    slice_results = result.scalars().all()

    pdf_bytes = await generate_pdf(job, case, slice_results)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{job_id[:8]}.pdf"'},
    )
