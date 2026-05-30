"""NIfTI loading with LRU cache for volume arrays (max 3 cases)."""
from functools import lru_cache
from io import BytesIO

import nibabel as nib
import numpy as np
from PIL import Image

from app.utils.ct_window import apply_window


@lru_cache(maxsize=3)
def _load_volume(filepath: str) -> np.ndarray:
    """Load NIfTI volume as float32 array, shape (X, Y, Z). Cached."""
    nii = nib.load(filepath)
    return nii.get_fdata(dtype=np.float32)


def get_slice_png(
    filepath: str,
    idx: int,
    center: int = 50,
    width: int = 400,
) -> bytes:
    """Return CT slice at z=idx as grayscale PNG bytes."""
    vol = _load_volume(filepath)          # (X, Y, Z)
    arr = vol[:, :, idx].T                # (H, W) – transpose for display
    windowed = apply_window(arr, center, width)
    img = Image.fromarray(windowed, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_volume_metadata(filepath: str) -> dict:
    nii = nib.load(filepath)
    shape = list(nii.shape[:3])
    spacing = [float(s) for s in nii.header.get_zooms()[:3]]
    return {"shape": shape, "spacing": spacing, "slice_count": shape[2]}


def invalidate_cache(filepath: str) -> None:
    """Remove a specific filepath from the LRU cache."""
    _load_volume.cache_clear()
