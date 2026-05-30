"""Celery task: run full-volume inference for one case with one model."""
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.services.inference_service import run_slice
from app.services.model_registry import get_model
from app.services.overlay_service import save_mask_png
from app.tasks.celery_app import celery_app
from app.utils.metrics import compute_volume_ml

# Use synchronous SQLAlchemy inside Celery (no async event loop)
_sync_engine = create_engine(
    str(settings.database_url).replace("sqlite+aiosqlite", "sqlite"),
    connect_args={"check_same_thread": False},
)
SyncSession = sessionmaker(bind=_sync_engine)


def _perimeter_px(binary_mask: np.ndarray) -> int:
    """Count 4-connectivity boundary edges using NumPy (no cv2/skimage required)."""
    if not binary_mask.any():
        return 0
    m = binary_mask.astype(np.int16)
    h      = int(np.abs(np.diff(m, axis=0)).sum())          # row transitions
    v      = int(np.abs(np.diff(m, axis=1)).sum())          # col transitions
    border = int(m[0, :].sum() + m[-1, :].sum()             # image-edge pixels
                 + m[:, 0].sum() + m[:, -1].sum())
    return h + v + border


def _get_job(session: Session, job_id: str):
    from app.models.inference_job import InferenceJob
    return session.get(InferenceJob, job_id)


@celery_app.task(bind=True, name="run_inference")
def run_inference(self, case_id: str, model_name: str):
    job_id = self.request.id
    session = SyncSession()

    try:
        # ── Mark running ──────────────────────────────────────────────────────
        import time
        job = _get_job(session, job_id)
        for _ in range(5):          # 最多重试 5 次，总等待 2.5s
            if job is not None:
                break
            time.sleep(0.5)
            job = _get_job(session, job_id)
        if job is None:
            raise RuntimeError(f"InferenceJob {job_id} not found in DB after retries")
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        session.commit()

        # ── Load case ─────────────────────────────────────────────────────────
        from app.models.case import Case
        case = session.get(Case, case_id)
        if case is None:
            raise RuntimeError(f"Case {case_id} not found in DB")
        filepath = case.filepath
        spacing = json.loads(case.voxel_spacing) if case.voxel_spacing else [1.0, 1.0, 1.0]
        sx, sy, sz = spacing

        nii = nib.load(filepath)
        vol = nii.get_fdata(dtype=np.float32)   # (X, Y, Z)
        total_slices = vol.shape[2]

        job.total_slices = total_slices
        session.commit()

        # ── Load model (LRU cache) ─────────────────────────────────────────────
        model = get_model(model_name)

        # ── Per-slice inference ────────────────────────────────────────────────
        masks_dir = settings.upload_dir / case_id / "results" / job_id / "masks"

        from app.models.slice_result import SliceResult
        liver_areas, tumor_areas = [], []

        for z in range(total_slices):
            gray = vol[:, :, z].T    # (H, W)
            mask = run_slice(model, gray)

            liver_px = int((mask == 1).sum())
            tumor_px = int((mask == 2).sum())
            liver_perim = _perimeter_px(mask == 1)
            tumor_perim = _perimeter_px(mask == 2)
            liver_areas.append(liver_px)
            tumor_areas.append(tumor_px)

            save_mask_png(mask, masks_dir / f"{z:04d}.png")

            sr = SliceResult(
                job_id=job_id, slice_index=z,
                liver_area_px=liver_px, tumor_area_px=tumor_px,
                liver_perimeter_px=liver_perim, tumor_perimeter_px=tumor_perim,
            )
            session.add(sr)

            # Update progress every 10 slices
            if z % 10 == 0 or z == total_slices - 1:
                job.current_slice = z + 1
                job.progress = int((z + 1) / total_slices * 100)
                session.commit()

        # ── Compute volumes ────────────────────────────────────────────────────
        pixel_area_mm2 = sx * sy
        job.liver_volume_ml = compute_volume_ml(liver_areas, pixel_area_mm2, sz)
        job.tumor_volume_ml = compute_volume_ml(tumor_areas, pixel_area_mm2, sz)
        job.status = "done"
        job.progress = 100
        job.finished_at = datetime.now(timezone.utc)
        session.commit()

    except Exception as exc:
        job = _get_job(session, job_id)
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            session.commit()
        raise
    finally:
        session.close()
