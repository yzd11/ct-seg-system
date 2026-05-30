import numpy as np


def apply_window(arr: np.ndarray, center: int = 50, width: int = 400) -> np.ndarray:
    """Clip HU values and rescale to [0, 255] uint8."""
    lo = center - width / 2
    hi = center + width / 2
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / width * 255).astype(np.uint8)
