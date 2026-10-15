# TEC Bayesian background–flare decomposition (Python)

Python version of the IDL routine `tec_bayes_decomp.pro` (see `../IDL`). It detrends GNSS total electron content (TEC) curves and extracts the ionospheric response to solar flares (ΔTEC). Each TEC curve (one satellite pass) is fitted over its whole duration as a smooth background plus a set of flare-induced peaks:

```
y(t)   = B(t) + Σ_k F_k(t)
B(t)   = c0 + Σ_{j=1..4} A_j sin(2π t / P_j + φ_j)
F_k(t) = I_k exp[ (t − t_k)/τ_r,k]   for t <  t_k
         I_k exp[−(t − t_k)/τ_d,k]   for t ≥ t_k
```

The model, priors, preprocessing and dummy-peak logic follow the IDL routine; the posterior is sampled with [PyMC](https://www.pymc.io) instead of SoBAT, so no IDL licence is needed. The number of peaks and their initial times come from user-supplied reference times (e.g. solar EUV peaks). Curves with fewer than 3 reference peaks get one extra "dummy" peak at a random time, which is used to estimate the level of spurious amplitudes. The method is described in Bekker, Kolotkov & Belov (submitted to GRL).

## Requirements

- Python 3.12 or 3.13.
- NumPy, SciPy, pandas, Matplotlib, **PyMC 6** and ArviZ 1 (versions in `pyproject.toml`).

With Conda (an isolated environment, recommended):

```bash
cd TEC_MCMC/python
conda env create -f environment.yml
conda activate tec-mcmc
```

or with pip, in an existing Python 3.12 environment:

```bash
cd TEC_MCMC/python
python -m pip install -e '.[test]'
```

Check:

```bash
python -c "import tec_mcmc, pymc; print(tec_mcmc.__file__, pymc.__version__)"
pytest            # fast deterministic tests
pytest -m mcmc    # short end-to-end MCMC run on a synthetic curve (~1 min)
```

No C compiler is needed: set `pytensor.config.cxx = ""` (or `PYTENSOR_FLAGS="cxx="`) as in the example notebook.

## Input data

The same CSV as for the IDL version, without a header:

| column | content |
|---|---|
| 1 | time, UT seconds of day |
| 2 … N+1 | TEC [TECU] for N satellite passes; `NaN` where a pass has no data |

`load_tec_csv` converts time to **minutes**. `../IDL/example/` contains a short file in this format (176 passes, 07:00–09:00 UT on 8 March 2011); the notebook uses the full-day file for 7 September 2017 in `../Data/`.

Reference peak times are given separately, in **UT minutes of day**.

## Usage

`notebooks/Example.ipynb` is the worked example: it fits one curve (plots, diagnostics, chain agreement, posterior summary) and then fits many curves in parallel.

### One curve

```python
import numpy as np
from tec_mcmc import SamplerConfig, fit_curve, load_tec_csv, prepare_curve

t_peaks = np.array([544., 549., 555., 561., 578.])        # UT minutes
curves = load_tec_csv("tec_20170906.csv")                 # list of (t [min], TEC) per pass

i = 0                                                     # 0-based curve index, as CURVES in IDL
t_abs, y_raw = curves[i]
t, y, flare_idx, true_idx, t0 = prepare_curve(t_abs, y_raw, t_peaks,
                                              rng=np.random.default_rng(i))

result = fit_curve(t, y, flare_idx, config=SamplerConfig(draws=20_000, tune=20_000))
dtec = y - result.map_trend                               # ΔTEC
```

`prepare_curve` performs all steps of the IDL loop before `mcmc_fit`: plateau collapse and interpolation, time shift to 0, matching of the reference times to samples, and the dummy peak.

### All curves

```python
from tec_mcmc import fit_tec_csv

if __name__ == "__main__":
    fit_tec_csv("tec_20170906.csv", "results/map_params.json", t_peaks,
                config=SamplerConfig(progressbar=False), curves=None, max_workers=8)
```

Curves are fitted in parallel worker processes, one PyMC core each. The JSON file is rewritten every `save_every` curves and curves already in it are skipped, so an interrupted run resumes when called again. Set `PYTENSOR_FLAGS` and `OMP/MKL/OPENBLAS_NUM_THREADS=1` before the call (see the notebook). On macOS, call it from a notebook or under `if __name__ == "__main__":`.

### Settings

`SamplerConfig` (see `config.py` for all fields):

| field | default | meaning |
|---|---|---|
| `draws`, `tune` | 20 000, 20 000 | posterior samples and tuning steps per chain |
| `chains`, `cores` | 4, 1 | number of chains; cores per curve (keep 1 with `fit_tec_csv`) |
| `sampler` | `"demetropolisz"` | PyMC step method (`"nuts"` also available) |
| `confidence_level` | 0.95 | credible level |
| `ppd_draws` | 5000 | posterior-predictive draws for the credible band |
| `n_starts` | 30 | multi-start least-squares runs that seed the chains (0 = IDL initial guess) |
| `random_seed` | 42 | sampler seed |

`fit_tec_csv` keywords: `curves` (`None` = all, `n` = first n, or a list of 0-based indices), `max_workers`, `save_every` (10). The dummy-peak position is seeded with the curve index.

Prior ranges are in `PriorConfig` and match the IDL script.

MCMC dominates the run time. For a quick test, use a few curves and short chains, e.g. `curves=[0, 1]`, `draws=tune=5_000`. For production, use 50 000–100 000 draws (a few minutes per curve).

## Output

`fit_curve` returns a `FitResult`:

- `t`, `y`: time [min, from the first point] and preprocessed TEC
- `map_model`, `map_trend`, `map_flares`: best-fit model, background B(t), sum of peaks Σ F_k(t). ΔTEC = `y − map_trend`
- `ppd_lower`, `ppd_upper`: credible band of the posterior predictive distribution
- `map_params`, `sigma_map`, `map_log_likelihood`: best-fit parameters, noise level and log-likelihood
- `posterior_samples`: all samples (rows chain by chain, same column layout as `map_params`)
- `flare_indices`, `n_flares`: indices of the initial peak times and number of peaks
- `diagnostics`: R-hat, ESS and chain agreement (see below); `map_source`: `"mcmc"` or `"optimizer"`

`fit_tec_csv` writes one JSON file with an entry `"flare_<i>"` per curve: `t_0` (start time, UT min), `t`, `y_raw`, `flare_indices`, `true_indices`, `has_dummy` and `map_params`.

Parameter layout: `[c0, (A_j, P_j, φ_j)×4, (I_k, t_k, τ_r,k, τ_d,k)×K, σ_noise]`. Peak times `t_k` are counted from the first point of the curve, so UT minute = `t_k + t_0`. Harmonics are sorted by period, shortest first. If a dummy peak was added (fewer than 3 reference peaks), it is the last peak block; otherwise there is no dummy block. `unpack_parameter_matrix` splits samples into named arrays.

No figures are written; the notebook shows how to plot the fit, ΔTEC and the chains.

## Differences from the IDL version

- **Sampler.** PyMC's `DEMetropolisZ` (gradient-free, adaptive) instead of SoBAT. Chains start from the best solutions of a multi-start least-squares search, and the best fit is the highest-likelihood state among all samples and that solution.
- **Parametrisation.** Harmonics are sampled as `b_sin`, `b_cos` with ordered periods. The prior is unchanged (uniform in amplitude and phase) and so are the fitted curves; this only removes label switching and the phase wrap.
- **Output.** `FitResult` / JSON instead of the `.sav` file; no NaN dummy block for curves with ≥ 3 reference peaks.
- **Preprocessing details.** Beyond the first/last plateau the edge value is held constant (IDL `INTERPOL` extrapolates linearly); the last plateau is closed; double precision throughout.


## Notes on convergence

Over a ~2-hour window many sets of four sinusoids give the same smooth trend, and two nearby-period sinusoids with large opposite amplitudes can cancel. Single background parameters are therefore not identifiable and their R-hat stays above 1; this is a property of the model and its wide priors, not of the sampler. The quantities that matter are checked instead:

- `rhat_curves_max`, `ess_curves_min`: R-hat and ESS of the trend, the summed peaks, the full model and σ. `converged` is `True` when R-hat < 1.05 and ESS > 100.
- `*_chain_spread_over_sigma`: largest difference between the chains' mean curves, in units of σ. If the trend and flare spreads exceed the model spread, the background/flare split is uncertain.

The best-fit trend and peaks are reliable. When chains settle in different modes, the credible band mixes them and should be read as indicative.

## Files

| file | content |
|---|---|
| `src/tec_mcmc/config.py` | `SamplerConfig` (sampling) and `PriorConfig` (prior ranges) |
| `src/tec_mcmc/models.py` | model functions and parameter layout |
| `src/tec_mcmc/preprocess.py` | plateau handling, noise estimate, reference-peak matching, dummy peak |
| `src/tec_mcmc/tec.py` | `load_tec_csv`, `prepare_curve` |
| `src/tec_mcmc/optimize.py` | multi-start least-squares search that seeds the chains |
| `src/tec_mcmc/inference.py` | PyMC model, sampling, best fit, credible band, diagnostics |
| `src/tec_mcmc/batch.py` | `fit_tec_csv`: parallel, resumable fitting of a whole CSV |
| `tests/` | unit tests and an end-to-end MCMC test (`pytest -m mcmc`) |
| `notebooks/Example.ipynb` | worked example |

## Citation

If you use this code, please cite:

- Bekker, S. Z., Kolotkov, D. Y., & Belov, S. A., *A Bayesian approach to GNSS TEC detrending during solar flares: substructure and relaxation of the ionospheric response*, Geophys. Res. Lett. (submitted).
- Abril-Pla, O., et al. (2023), *PyMC: a modern, and comprehensive probabilistic programming framework in Python*, PeerJ Comput. Sci., 9, e1516, [doi:10.7717/peerj-cs.1516](https://doi.org/10.7717/peerj-cs.1516).

## Author

Sergey Belov, University of Warwick (sergey.belov@warwick.ac.uk)
