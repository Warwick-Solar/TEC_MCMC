"""Template replacing the high-level loop in TEC_all-day_noise.pro.

Edit CSV_PATH and T_PEAKS_SOLAR before running (from the repository root).
"""

from pathlib import Path

import numpy as np

from tec_mcmc import SamplerConfig
from tec_mcmc.batch import fit_many_curves
from tec_mcmc.io import save_fit_result
from tec_mcmc.tec import load_tec_csv, prepare_curve

CSV_PATH = Path("Data/tec_vert_no_SZA_20170907_El_10_90_Lat_15_60_Lon_90_90_SSA_30_90min.csv")
T_PEAKS_SOLAR = np.array([302., 454., 594., 615., 873., 876., 880., 891., 993., 1123., 1308.])


def main() -> None:
    rng = np.random.default_rng(42)
    raw_curves = load_tec_csv(CSV_PATH)  # header-less CSV, seconds -> minutes

    jobs, curve_ids = [], []
    for i, (t_abs, y_raw) in enumerate(raw_curves):
        if len(t_abs) < 20:  # nothing sensible to fit
            continue
        t_rel, y, flare_idx, _true_idx, _t0 = prepare_curve(
            t_abs, y_raw, T_PEAKS_SOLAR, rng=rng
        )
        jobs.append((t_rel, y, flare_idx))
        curve_ids.append(i)

    config = SamplerConfig(chains=4, cores=1, sampler="demetropolisz", random_seed=42)

    results = fit_many_curves(jobs, max_workers=8, config=config)
    outdir = Path("tec_mcmc_results")
    outdir.mkdir(exist_ok=True)
    for i, result in zip(curve_ids, results):
        if isinstance(result, Exception):
            print(f"curve {i} failed: {result!r}")
            continue
        save_fit_result(outdir / f"curve_{i:04d}.npz", result)


if __name__ == "__main__":
    main()
