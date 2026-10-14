from __future__ import annotations

from pathlib import Path

import numpy as np

from .inference import FitResult


def save_fit_result(path: str | Path, result: FitResult) -> None:
    """Save compact numerical outputs as a compressed NPZ file."""
    path = Path(path)
    np.savez_compressed(
        path,
        t=result.t,
        y=result.y,
        n_flares=np.asarray(result.n_flares),
        flare_indices=result.flare_indices,
        posterior_samples=result.posterior_samples,
        map_params=result.map_params,
        map_log_likelihood=np.asarray(result.map_log_likelihood),
        map_model=result.map_model,
        map_trend=result.map_trend,
        map_flares=result.map_flares,
        sigma_map=np.asarray(result.sigma_map),
        ppd_lower=result.ppd_lower,
        ppd_upper=result.ppd_upper,
        confidence_level=np.asarray(result.confidence_level),
        map_source=np.asarray(result.map_source),
        rhat_max=np.asarray(np.nan if result.diagnostics is None else result.diagnostics["rhat_max"]),
        ess_bulk_min=np.asarray(
            np.nan if result.diagnostics is None else result.diagnostics["ess_bulk_min"]
        ),
    )
