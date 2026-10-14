from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PriorConfig, SamplerConfig
from .inference import FitResult, fit_curve

Curve = tuple[np.ndarray, np.ndarray, np.ndarray | list[int] | None]


def _fit_worker(args):
    t, y, flare_indices, config, priors, seed_offset, on_error = args
    worker_config = replace(
        config, random_seed=config.random_seed + seed_offset, cores=1, progressbar=False
    )
    try:
        return fit_curve(
            t,
            y,
            flare_indices,
            config=worker_config,
            priors=priors,
            keep_idata=False,
        )
    except Exception as exc:  # one bad curve must not kill a 2000-curve run
        if on_error == "raise":
            raise
        return exc


def fit_many_curves(
    curves: Iterable[Curve],
    *,
    max_workers: int | None = None,
    config: SamplerConfig | None = None,
    priors: PriorConfig | None = None,
    on_error: str = "return",
) -> list[FitResult | Exception]:
    """Fit independent curves in separate processes.

    PyMC is forced to one core inside each worker to avoid nested parallelism.
    With ``on_error="return"`` (default) a failed curve yields its exception in
    the result list instead of aborting the whole batch; use ``"raise"`` to
    stop at the first failure.
    """
    config = config or SamplerConfig(cores=1)
    priors = priors or PriorConfig()
    payload = [(*curve, config, priors, i, on_error) for i, curve in enumerate(curves)]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_fit_worker, payload))


def fit_one(curve_id, t, y, flare_idx, config):
    """Fit one curve; module-level so ProcessPoolExecutor workers can import it."""
    result = fit_curve(t, y, flare_idx, config=config)
    return curve_id, result.map_params.tolist()


def save(results, path):
    """Write ``results`` as JSON atomically: the file is never left half-written."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results))
    os.replace(tmp, path)
