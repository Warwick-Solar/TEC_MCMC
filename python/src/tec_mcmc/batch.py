"""Fit many TEC curves in parallel, saving best-fit parameters to one JSON file."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PriorConfig, SamplerConfig
from .inference import fit_curve
from .tec import load_tec_csv, prepare_curve


def _fit_one(curve_id, t, y, flare_idx, config, priors):
    """Fit one curve in a worker process (module-level so it can be pickled)."""
    result = fit_curve(t, y, flare_idx, config=config, priors=priors)
    return curve_id, result.map_params.tolist()


def save_json_atomic(obj, path: str | Path) -> None:
    """Write ``obj`` as JSON atomically: the file is never left half-written."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def fit_tec_csv(
    csv_path: str | Path,
    out_json: str | Path,
    t_peaks_solar,
    *,
    config: SamplerConfig | None = None,
    priors: PriorConfig | None = None,
    curves: int | Iterable[int] | None = None,
    max_workers: int | None = None,
    save_every: int = 10,
) -> dict:
    """Fit TEC curves from a CSV in parallel and collect best-fit parameters in JSON.

    Each curve gets an entry ``"flare_<id>"`` (``<id>`` = 0-based curve number)::

        {"t_0": absolute start time [min],
         "t": fitted time grid relative to t_0 [min],
         "y_raw": raw TEC values (before plateau interpolation),
         "flare_indices": fitted flare sample indices (last may be a dummy),
         "true_indices": indices of the known solar flares,
         "has_dummy": whether a dummy flare was added,
         "map_params": [offset, A1, P1, phi1, ..., A4, P4, phi4,
                        (flare amp, peak, rise, decay) x n_flares, sigma]}

    Flare peak times in ``map_params`` are relative to ``t_0``.

    ``curves``: ``None`` = all curves, an int ``n`` = the first ``n`` curves,
    or an iterable of curve ids.  The JSON is rewritten every ``save_every``
    new results, and curves already in it are skipped, so an interrupted run
    can be resumed by calling this again.  A failing curve is reported and
    skipped instead of stopping the run.

    Each worker runs PyMC on one core.  Set ``PYTENSOR_FLAGS`` and the
    ``*_NUM_THREADS`` environment variables *before* calling this; worker
    processes inherit them.  On macOS (``spawn``), call it from a notebook or
    from under ``if __name__ == "__main__":`` in a script.
    """
    config = config or SamplerConfig(cores=1, progressbar=False)
    priors = priors or PriorConfig()
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_json.read_text()) if out_json.exists() else {}

    all_curves = load_tec_csv(csv_path)
    if curves is None:
        ids = range(len(all_curves))
    elif isinstance(curves, int):
        ids = range(curves)
    else:
        ids = curves

    jobs, meta = [], {}
    for i in ids:
        if f"flare_{i}" in results:
            continue
        t_abs, y_raw = all_curves[i]
        rng = np.random.default_rng(i)  # dummy-flare placement, reproducible per curve
        t, y, flare_idx, true_idx, t0 = prepare_curve(t_abs, y_raw, t_peaks_solar, rng=rng)
        jobs.append((i, t, y, flare_idx))
        meta[i] = {
            "t_0": t0,
            "t": t.tolist(),
            "y_raw": np.asarray(y_raw, float).tolist(),
            "flare_indices": np.asarray(flare_idx, int).tolist(),
            "true_indices": np.asarray(true_idx, int).tolist(),
            "has_dummy": len(flare_idx) > len(true_idx),
        }
    print(f"{len(jobs)} curves to fit, {len(results)} already done")

    start, unsaved = time.time(), 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fit_one, *job, config, priors): job[0] for job in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                _, params = fut.result()
            except Exception as exc:  # one bad curve must not stop the run
                print(f"curve {i} failed: {exc!r}")
                continue
            results[f"flare_{i}"] = {**meta[i], "map_params": params}
            unsaved += 1
            if unsaved >= save_every:
                save_json_atomic(results, out_json)
                unsaved = 0
            elapsed = time.time() - start
            print(
                f"[{n}/{len(jobs)}] curve {i} done, "
                f"~{elapsed / n * (len(jobs) - n) / 60:.0f} min left"
            )
    save_json_atomic(results, out_json)
    print(f"saved {len(results)} curves to {out_json} ({(time.time() - start) / 60:.1f} min)")
    return results
