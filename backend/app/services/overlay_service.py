"""Render segmentation mask as RGBA overlay PNG."""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


# Liver: green semi-transparent; Tumor: red high-opacity
_COLORS = {
    1: (0, 200, 0, 160),    # liver
    2: (220, 30, 30, 200),  # tumor
}


def mask_to_rgba(mask: np.ndarray) -> bytes:
    """Convert 2D mask (H, W) with values 0/1/2 to RGBA PNG bytes."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, color in _COLORS.items():
        rgba[mask == cls] = color

    img = Image.fromarray(rgba, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_mask_png(mask: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(mask_to_rgba(mask))
