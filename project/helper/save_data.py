"""Persistencia de resultados de separación CROP en formato JSON."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def resolve_unmix_save_path(data_save_path=None, *, data_save_dir=None,
                            notebook_name=None, model_name=None, crop_name=None,
                            initial_params=None):
    """Generate a timestamped filename with parameter values at three decimals."""
    if data_save_dir is not None:
        if data_save_path is not None:
            raise ValueError("Use data_save_path or data_save_dir, not both.")
        if not notebook_name:
            raise ValueError("notebook_name is required with data_save_dir.")
        def filename_part(value):
            return re.sub(r"[^\w.\-]+", "-", str(value))
        def parameter_part(value):
            if isinstance(value, (int, float, np.integer, np.floating)):
                return filename_part(f"{value:.3f}")
            return filename_part(value)
        filename = (
            f"{filename_part(Path(notebook_name).stem)}_"
            f"{filename_part(crop_name)}_"
            f"{filename_part(Path(model_name).stem)}_"
            f"gamma_{parameter_part(initial_params.get('gamma', 'NA'))}_"
            f"delta_{parameter_part(initial_params.get('delta', 'NA'))}_"
            f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
        )
        data_save_path = Path(data_save_dir) / filename
    return data_save_path


def save_unmix_data(data_save_path, *, model, iterations, initial_params,
                    final_params, curves, metrics, images, notebook_name=None,
                    run_metadata=None):
    """Save schema v2 results, converting arrays, tensors and non-finite values.

    Returns the written Path. The payload remains compatible with
    load_unmix_curves in helper.graphics.graphics.
    """
    payload = {
        "model": model,
        "parameters": {
            "iterations": iterations,
            "model_params": initial_params,
            "final_model_params": final_params.copy(),
        },
        "last_results": {
            "acc_both": curves["acc_both"][-1] if curves and curves["acc_both"] else metrics["acc_both"],
            "psnr": curves["recon_bpsnr"][-1] if curves and curves["recon_bpsnr"] else metrics["recon_bpsnr"][0],
            "ssim": curves["ssim"][-1] if curves and curves["ssim"] else metrics["ssim"][0],
        },
        "data": {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notebook_name": notebook_name,
            "metadata": run_metadata or {},
            "curves": curves,
            "metrics": metrics,
            "images": images,
        },
    }
    path = Path(data_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_data(payload), file, ensure_ascii=False, allow_nan=False, indent=2)
    return path


def _json_data(value):
    """Convert NumPy arrays, eager tensors and scalars to portable JSON data."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: _json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__float__": str(value)}
    return value

