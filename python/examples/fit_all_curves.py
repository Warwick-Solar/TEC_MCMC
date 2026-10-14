"""Fit every TEC curve in parallel and collect the best-fit parameters in one JSON file.

Run from the repository root:

    caffeinate -i python examples/fit_all_curves.py

Output: one JSON object with an entry per curve,

    "flare_<id>": {
        "t_0": absolute start time [min],
        "t": fitted time grid, relative to t_0 [min],
        "y_raw": raw TEC values (before plateau interpolation),
        "flare_indices": sample indices of the fitted flares (last may be a dummy),
        "map_params": [offset, A1, P1, phi1, ..., A4, P4, phi4,
                       (flare amp, peak, rise, decay) x n_flares, sigma]
    }

<id> is the 0-based curve number, the same as CURVE in the notebooks.
Flare peak times in map_params are relative to t_0.
The file is rewritten every few curves; rerunning the script skips curves
already in it, so an interrupted run can be resumed.
"""

import os

# Must be set before pytensor is imported; worker processes inherit them.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")  # same as pytensor.config.cxx = "" in the notebooks
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tec_mcmc import SamplerConfig, fit_curve
from tec_mcmc.tec import load_tec_csv, prepare_curve

CSV_PATH = Path("Data/tec_vert_no_SZA_20170907_El_10_90_Lat_15_60_Lon_90_90_SSA_30_90min.csv")
OUT_JSON = Path("tec_mcmc_results/map_params.json")
T_PEAKS_SOLAR = np.array([302., 454., 594., 615., 873., 876., 880., 891., 993., 1123., 1308.])

MAX_WORKERS = 8      # number of performance cores
CURVES = None        # None = all curves; e.g. range(8) for a timing test
SAVE_EVERY = 10      # rewrite the JSON after this many new results

CONFIG = SamplerConfig(
    draws=20_000, tune=20_000, chains=4, cores=1,
    sampler="demetropolisz", random_seed=42, n_starts=30,
    progressbar=False,
)


def _fit_one(curve_id, t, y, flare_idx):
    """Runs in a worker process."""
    result = fit_curve(t, y, flare_idx, config=CONFIG)
    return curve_id, result.map_params.tolist()


def _save(results):
    tmp = OUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(results))
    os.replace(tmp, OUT_JSON)   # atomic: the file is never left half-written


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}

    curves = load_tec_csv(CSV_PATH)
    ids = range(len(curves)) if CURVES is None else CURVES

    jobs, meta = [], {}
    for i in ids:
        if f"flare_{i}" in results:
            continue
        t_abs, y_raw = curves[i]
        rng = np.random.default_rng(i)   # dummy-flare placement, as in the notebooks
        t, y, flare_idx, _true_idx, t0 = prepare_curve(t_abs, y_raw, T_PEAKS_SOLAR, rng=rng)
        jobs.append((i, t, y, flare_idx))
        meta[i] = {
            "t_0": t0,
            "t": t.tolist(),
            "y_raw": np.asarray(y_raw, float).tolist(),
            "flare_indices": np.asarray(flare_idx, int).tolist(),
        }
    print(f"{len(jobs)} curves to fit, {len(results)} already done")

    start, unsaved = time.time(), 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fit_one, *job): job[0] for job in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            try:
                _, params = fut.result()
            except Exception as exc:   # one bad curve must not stop the run
                print(f"curve {i} failed: {exc!r}")
                continue
            results[f"flare_{i}"] = {**meta[i], "map_params": params}
            unsaved += 1
            if unsaved >= SAVE_EVERY:
                _save(results)
                unsaved = 0
            elapsed = time.time() - start
            print(f"[{n}/{len(jobs)}] curve {i} done, "
                  f"~{elapsed / n * (len(jobs) - n) / 60:.0f} min left")
    _save(results)
    print(f"saved {len(results)} curves to {OUT_JSON}")


if __name__ == "__main__":   # required: macOS starts workers with "spawn"
    main()
