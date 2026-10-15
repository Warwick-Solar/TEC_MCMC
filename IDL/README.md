# TEC Bayesian background–flare decomposition

IDL code for detrending GNSS total electron content (TEC) curves and extracting the ionospheric response to solar flares (ΔTEC). Each TEC curve (one satellite pass) is fitted over its whole duration as a smooth background plus a set of flare-induced peaks:

```
y(t)   = B(t) + Σ_k F_k(t)
B(t)   = c0 + Σ_{j=1..4} A_j sin(2π t / P_j + φ_j)
F_k(t) = I_k exp[ (t − t_k)/τ_r,k]   for t <  t_k
         I_k exp[−(t − t_k)/τ_d,k]   for t ≥ t_k
```

The posterior is sampled by MCMC with [SoBAT](https://github.com/Sergey-Anfinogentov/SoBAT). The number of peaks and their initial times come from user-supplied reference times (e.g. solar EUV peaks). Curves with fewer than 3 reference peaks get one extra "dummy" peak at a random time, which is used to estimate the level of spurious amplitudes. The method is described in Bekker, Kolotkov & Belov (submitted to GRL).

## Requirements

- IDL 8.0 or later.
- **SoBAT** (Solar Bayesian Analysis Toolkit; [Anfinogentov et al. 2021, ApJS 252, 11](https://doi.org/10.3847/1538-4365/abc5c1)), which does the MCMC sampling:
  ```
  git clone https://github.com/Sergey-Anfinogentov/SoBAT.git
  ```
  (or *Code → Download ZIP* on the [repository page](https://github.com/Sergey-Anfinogentov/SoBAT)). The IDL routines are in `SoBAT/idl`.
- **Coyote IDL library**, needed by SoBAT for `cgPercentiles`:
  ```
  git clone https://github.com/idl-coyote/coyote.git
  ```
  Skip this if Coyote is already on your path (e.g. via SolarSoft).

Add both libraries and this repository to the IDL path, e.g. in your IDL startup file:

```idl
!PATH = EXPAND_PATH('+/path/to/SoBAT/idl') + PATH_SEP(/SEARCH_PATH) + $
        EXPAND_PATH('+/path/to/coyote')    + PATH_SEP(/SEARCH_PATH) + $
        EXPAND_PATH('+/path/to/this/repo') + PATH_SEP(/SEARCH_PATH) + !PATH
```

or via *IDL Workbench → Preferences → IDL → Paths*. Check:

```idl
IDL> print, file_which('mcmc_fit.pro'), file_which('cgpercentiles.pro')
```

## Input data

A comma-separated file without a header:

| column | content |
|---|---|
| 1 | time, UT seconds of day |
| 2 … N+1 | TEC [TECU] for N satellite passes; `NaN` where a pass has no data |

`example/` contains a short file in this format (176 passes, 07:00–09:00 UT on 8 March 2011).

Reference peak times are given separately, in **UT minutes of day**.

## Usage

```idl
IDL> tec_bayes_decomp, 'tec_20170906.csv', [544., 549., 555., 561., 578.], OUT_DIR='results/'
```

Main keywords (see the header of `tec_bayes_decomp.pro` for all of them):

| keyword | default | meaning |
|---|---|---|
| `OUT_DIR` | folder of the CSV | output folder |
| `CURVES` | all | 0-based indices of the TEC columns to process |
| `BURN_IN`, `N_SAMPLES` | 1e5, 1e5 | MCMC burn-in and posterior samples per curve |
| `CONF_LEVEL` | 0.95 | credible level |
| `SEED` | clock | seed for the dummy-peak position |
| `/NOPLOT` | – | no graphics or PNG output |

MCMC dominates the run time. For a quick test, use a few curves and short chains, e.g. `CURVES=[0,1], BURN_IN=1e4, N_SAMPLES=1e4`. Prior ranges are set in one block near the top of the main procedure.

For runs without a display, use `/NOPLOT`, or `SET_PLOT, 'Z'` with `DEVICE, SET_PIXEL_DEPTH=24` before running so that PNGs are still written.

## Output

`<OUT_DIR>/<name>_bayes.sav`, holding IDL lists with one element per curve:

- `time_list`, `obs_list`: time [UT min] and observed TEC
- `fits_list`, `trend_list`, `flares_list`: best-fit model, background B(t), sum of peaks Σ F_k(t). ΔTEC = `obs_list[i] − trend_list[i]`
- `cred_up_list`, `cred_down_list`: credible band of the posterior predictive distribution
- `params_list`, `param_ci_list`: best-fit (maximum a posteriori) parameters and their credible intervals
- `maxpos_list`, `n_ref_list`, `column_list`: indices of the initial peak times, number of reference peaks, CSV column index

Parameter layout: `[c0, (A_j, P_j, φ_j)×4, (I_k, t_k, τ_r,k, τ_d,k)×K, σ_noise]`. Peak times `t_k` are counted from the first point of the curve, so UT minute = `t_k + time_list[i][0]`. The last peak block is always the dummy peak; it is `NaN` for curves with ≥ 3 reference peaks.

PNG figures are written for each curve (initial guess, fit) and for all curves together (fits, ΔTEC).

## Citation

If you use this code, please cite:

- Bekker, S. Z., Kolotkov, D. Y., & Belov, S. A., *A Bayesian approach to GNSS TEC detrending during solar flares: substructure and relaxation of the ionospheric response*, Geophys. Res. Lett. (submitted).
- Anfinogentov, S. A., Nakariakov, V. M., Pascoe, D. J., & Goddard, C. R. (2021), *Solar Bayesian Analysis Toolkit — a new Markov chain Monte Carlo IDL code for Bayesian parameter inference*, ApJS, 252, 11, [doi:10.3847/1538-4365/abc5c1](https://doi.org/10.3847/1538-4365/abc5c1).

## Author

Dmitrii Kolotkov, University of Warwick (d.kolotkov.1@warwick.ac.uk)
