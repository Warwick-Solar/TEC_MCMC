# TEC MCMC detrending — Python prototype

This package reproduces the **scientific model and outputs used by `TEC_all-day_noise.pro`** without requiring IDL, GDL, or SoBAT.

It is deliberately **not a line-by-line port of SoBAT**. PyMC supplies the MCMC machinery. The code preserves the parts of the SoBAT workflow that matter for the colleague's script:

- four-sinusoid background model;
- optional asymmetric exponential flare components;
- the same bounded uniform priors used in the IDL script;
- unknown Gaussian observational noise `sigma`, with the SoBAT default prior `Uniform(0, max(y)-min(y))`;
- a returned fit defined by the **highest-posterior sampled state** (for these uniform priors this is the highest-likelihood sampled state), matching `mcmc_fit`'s practical behaviour;
- posterior-predictive draws including observational noise;
- pointwise 95% posterior-predictive intervals;
- batch-level multiprocessing across independent light curves.

The MCMC kernel is intentionally *not* identical to SoBAT. The default is PyMC's `DEMetropolisZ`, which is a gradient-free adaptive Metropolis-family sampler. `NUTS` is also available. This should be validated against representative SoBAT fits before publication.

## Safe installation on a Mac

The supplied Conda environment is isolated from your existing Python/MPI/PETSc setup:

```bash
cd tec_mcmc_python
conda env create -f environment.yml
conda activate tec-mcmc
```

Alternatively, in an already isolated Python 3.12 environment:

```bash
python -m pip install -e '.[test]'
```

## Run deterministic tests

```bash
pytest
```

These tests do not need to run a long MCMC chain.

## Run the synthetic end-to-end smoke test

```bash
python examples/synthetic_smoke_test.py
```

The smoke test intentionally uses a small number of draws so it finishes quickly. It is a functionality test, not a convergence-quality run.

## Minimal use

```python
from tec_mcmc import SamplerConfig, fit_curve

result = fit_curve(
    t,
    y,
    flare_indices=[120],
    config=SamplerConfig(
        draws=4000,
        tune=4000,
        chains=4,
        cores=1,                 # safe inside outer batch workers
        sampler="demetropolisz",
        random_seed=42,
    ),
)

trend = result.map_trend
flares = result.map_flares
lower = result.ppd_lower
upper = result.ppd_upper
```

## Parallel batch fitting

Parallelise **curves**, not iterations of a single Markov chain:

```python
from tec_mcmc.batch import fit_many_curves

results = fit_many_curves(curves, max_workers=8, config=config)
```

Each entry in `curves` is `(t, y, flare_indices)`.

## Important methodological note

The four harmonic components have exchangeable/overlapping priors in the original script. Their labels can therefore switch. The *sum* of the four harmonics (the detrending curve) can still be well determined, while individual harmonic periods may not be uniquely labelled. This prototype preserves that behaviour for compatibility. If individual harmonic parameters are to be interpreted scientifically, impose an ordering or non-overlapping period ranges in a later model revision.

## Files

- `src/tec_mcmc/models.py` — NumPy model functions and parameter-vector layout.
- `src/tec_mcmc/preprocess.py` — plateau handling, noise estimate, and flare-candidate helpers.
- `src/tec_mcmc/inference.py` — PyMC model, sampling, MAP-sample extraction, PPD intervals.
- `src/tec_mcmc/batch.py` — process-level parallel fitting across curves.
- `src/tec_mcmc/io.py` — save/load compact fit results.
- `tests/test_core.py` — deterministic unit tests.
- `examples/synthetic_smoke_test.py` — small end-to-end MCMC test.
- `examples/run_tec_csv_template.py` — template for adapting the colleague's CSV workflow.


## Review notes / changes (v0.1.1)

**Bug fixes**

- `load_tec_csv` used `header="infer"`, but the TEC CSV files have no header, so the first time sample was silently dropped from every curve. It now defaults to `header=None` (as IDL `READ_CSV` behaves).
- The example notebook skipped the IDL preprocessing (plateau collapse + interpolation, time shift to 0). Use `tec.prepare_curve(...)`, which performs every pre-`mcmc_fit` step of the IDL loop (including the dummy-flare logic).
- `estimate_noise_savgol` now follows IDL exactly: window rule `n/5` (2 if `< degree`), `EDGE_TRUNCATE` boundary (`mode="nearest"`) and sample standard deviation (ddof=1).
- `fit_many_curves` no longer loses a whole batch when one curve fails (`on_error="return"`).

**Sampling quality**

The posterior of four free-period sinusoids plus flares is strongly multimodal. Started from the generic IDL guess, `DEMetropolisZ` stayed far from the best mode (e.g. curve 0: sigma 0.15 vs 0.08 attainable, R-hat 2-4). The fit now:

1. runs a multi-start bounded least-squares search (`optimize.py`, `SamplerConfig.n_starts`);
2. starts each chain from one of the best optima (pushed inside the prior bounds, jittered);
3. returns the highest-likelihood state among all MCMC samples *and* the best optimiser solution (`FitResult.map_source` says which);
4. reports `FitResult.diagnostics` (`rhat_max`, `ess_bulk_min`, `converged`).

Caveat: chains still typically settle in different modes (R-hat >> 1). The best-fit trend and flares are reliable; the pooled posterior-predictive band is a mixture over modes and should be read as indicative only. Default draws were raised to 20 000 + 20 000 (roughly the IDL 100 000 + 100 000 budget; DEMetropolisZ is cheap).

**Other**

- `_log_likelihoods` is vectorised (the per-row Python loop dominated run time for >1e5 samples).
- New tests for the loader, the likelihood, the optimiser and the noise estimate.
- `notebooks/fit_one_curve.ipynb` replaces the ad-hoc `src/tec_mcmc/Untitled.ipynb` workflow.

**Known differences from the IDL routine** (deliberate): IDL's `INTERPOL` extrapolates linearly beyond the first/last plateau midpoint, whereas we hold the edge value constant; the final plateau is closed (the IDL `q EQ n` test is unreachable); IDL stores plateau arrays in single precision.


## Convergence work (v0.1.2)

Why R-hat stays high for single parameters: over a ~2-hour window many different sets of four sinusoids produce the same smooth trend, and two nearby-period sinusoids with large opposite amplitudes can cancel almost exactly. Single trend parameters are therefore not identifiable. That is a property of the model and its wide IDL priors (amplitude up to 500, period down to 50 min), not of the sampler; SoBAT faces the same posterior but only reports its MAP.

Changes:

- **Cartesian harmonics** (`b_sin`, `b_cos` instead of amplitude/phase). This removes the phase wrap at +-pi and the singularity at A = 0. With `PriorConfig.amplitude_prior="idl"` (the default), a `-log A` potential keeps the IDL prior (uniform in A and in phase) exactly.
- **Ordered periods** remove the 24 label-switching copies of every mode. The fitted curves are unchanged; harmonic 1 is now the shortest period.
- **Chain starts fixed.** v0.1.1 jittered starts by a fraction of the *prior range*, e.g. +-2.5 in amplitude for a 0.3 signal, so chains began outside the mode and wandered. Chains now start at the best optimum, or at any optimum within `mode_tolerance` log-likelihood units of it, jittered by the local Laplace standard deviation.
- **Curve-level diagnostics.** These cover the trend, the summed flares, the full model and sigma: `rhat_curves_max`, `ess_curves_min`, and `*_chain_spread_over_sigma` (the largest gap between chain-mean curves divided by sigma). `converged` is judged on the curves. Parameter-level values are still reported.

Tried and not adopted: NUTS and nutpie (slower, and worse on these ridges); SMC; tighter amplitude bounds (they made mixing worse, and they change the science). Analytic marginalisation of the linear parameters made the flare parameters mix better. It also showed that the posterior puts real mass on large cancelling sinusoid pairs, so the trend/flare split is inherently uncertain.

Curve 789 (4 known flares, 4 x (100k + 100k) draws, ~3.5 min): full-model chains agree to 0.26 sigma, curve R-hat is 1.25 with ESS of about 550, and the trend/flare split differs by about 1.5 sigma between chains. With 20k draws, curve R-hat is about 1.9 and ESS about 55. For production, use 50k-100k draws.
