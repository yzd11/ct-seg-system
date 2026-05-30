"""LRU model cache: keep at most MAX_MODELS loaded models in memory."""
from collections import OrderedDict
from pathlib import Path

import torch

from app.config import settings

_cache: OrderedDict[str, torch.nn.Module] = OrderedDict()
MAX_MODELS = settings.max_cached_models

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name: str) -> torch.nn.Module:
    if model_name in _cache:
        _cache.move_to_end(model_name)
        return _cache[model_name]

    # Evict LRU entry if at capacity
    if len(_cache) >= MAX_MODELS:
        evicted, _ = _cache.popitem(last=False)
        print(f"[ModelRegistry] Evicted '{evicted}' from cache")

    model = _build_model(model_name)
    _cache[model_name] = model
    return model


def _build_model(model_name: str) -> torch.nn.Module:
    from app.ml_models import MODEL_REGISTRY

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_channels=3, num_classes=3)

    weight_path = settings.weights_dir / model_name / "best.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights not found: {weight_path}")

    state = torch.load(str(weight_path), map_location=_DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(_DEVICE)
    model.eval()
    print(f"[ModelRegistry] Loaded '{model_name}' on {_DEVICE}")
    return model
