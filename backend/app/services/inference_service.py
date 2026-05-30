"""Single-slice inference: CT array → segmentation mask (256×256)."""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from app.config import settings
from app.utils.ct_window import apply_window

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_slice(model: torch.nn.Module, gray_arr: np.ndarray) -> np.ndarray:
    """
    Args:
        model:    loaded, eval-mode PyTorch model
        gray_arr: 2D float32 HU array (H, W) for a single slice

    Returns:
        mask: 2D uint8 array (256, 256) with values 0/1/2
    """
    windowed = apply_window(gray_arr)              # uint8 (H, W)
    img = Image.fromarray(windowed, mode="L").convert("RGB")
    img = img.resize((settings.img_size, settings.img_size), Image.BILINEAR)

    tensor = to_tensor(img).unsqueeze(0).to(_DEVICE)   # (1, 3, 256, 256)

    out = model(tensor)
    if isinstance(out, (list, tuple)):
        out = out[0]                                    # deep supervision → take first output

    mask = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask
