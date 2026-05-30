def compute_volume_ml(area_px_list: list[int], pixel_area_mm2: float, slice_thickness_mm: float) -> float:
    """Estimate organ volume in milliliters from per-slice pixel counts."""
    total_px = sum(area_px_list)
    volume_mm3 = total_px * pixel_area_mm2 * slice_thickness_mm
    return volume_mm3 / 1000.0
