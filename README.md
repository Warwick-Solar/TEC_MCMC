# TEC_MCMC: Bayesian background–flare decomposition of GNSS TEC

Code for detrending GNSS total electron content (TEC) curves and extracting the ionospheric response to solar flares (ΔTEC). Each TEC curve (one satellite pass) is fitted over its whole duration as a smooth background plus a set of flare-induced peaks:

```
y(t)   = B(t) + Σ_k F_k(t)
B(t)   = c0 + Σ_{j=1..4} A_j sin(2π t / P_j + φ_j)
F_k(t) = I_k exp[ (t − t_k)/τ_r,k]   for t <  t_k
         I_k exp[−(t − t_k)/τ_d,k]   for t ≥ t_k
```

The posterior is sampled by MCMC. The number of peaks and their initial times come from user-supplied reference times (e.g. solar EUV peaks). Curves with fewer than 3 reference peaks get one extra "dummy" peak at a random time, which is used to estimate the level of spurious amplitudes. The method is described in Bekker, Kolotkov & Belov (submitted to GRL).

## Implementations

| folder | language | MCMC | entry point | output |
|---|---|---|---|---|
| [`IDL/`](IDL/README.md) | IDL 8+ | [SoBAT](https://github.com/Sergey-Anfinogentov/SoBAT) | `tec_bayes_decomp` | `.sav` file + PNG figures |
| [`python/`](python/README.md) | Python 3.12+ | [PyMC](https://www.pymc.io) | `fit_curve`, `fit_tec_csv` | `FitResult` object / JSON file |

The IDL code is the reference implementation used in the paper. The Python package uses the same model, priors, preprocessing and dummy-peak logic but a different sampler, and does not need an IDL licence; see [`python/README.md`](python/README.md#differences-from-the-idl-version) for the differences.

Both read the same input: a header-less CSV with time (UT seconds of day) in the first column and TEC [TECU] for one satellite pass in each further column (`NaN` where a pass has no data), plus reference peak times in UT minutes of day. `IDL/example/` contains a short file in this format (8 March 2011); `Data/` holds the full-day file for 7 September 2017 used in the Python example notebook.

## Quick start

IDL (with SoBAT and Coyote on the path):

```idl
IDL> tec_bayes_decomp, 'tec_20170906.csv', [544., 549., 555., 561., 578.], OUT_DIR='results/'
```

Python:

```bash
cd python
conda env create -f environment.yml && conda activate tec-mcmc
```

then run `python/notebooks/Example.ipynb` (Jupyter or VS Code).

Installation, options and output formats are described in each folder's README.

## Citation

If you use this code, please cite:

- Bekker, S. Z., Kolotkov, D. Y., & Belov, S. A., *A Bayesian approach to GNSS TEC detrending during solar flares: substructure and relaxation of the ionospheric response*, Geophys. Res. Lett. (submitted).

and, for the IDL version, SoBAT ([Anfinogentov et al. 2021, ApJS 252, 11](https://doi.org/10.3847/1538-4365/abc5c1)); for the Python version, PyMC ([Abril-Pla et al. 2023, PeerJ Comput. Sci. 9, e1516](https://doi.org/10.7717/peerj-cs.1516)).

## License

Apache License 2.0, see [`LICENSE`](LICENSE).

## Authors

- Dmitrii Kolotkov, University of Warwick (d.kolotkov.1@warwick.ac.uk): IDL code
- Sergey Belov, University of Warwick (sergey.belov@warwick.ac.uk): Python code, developed with AI coding assistants (OpenAI ChatGPT, Anthropic Claude); see [`python/README.md`](python/README.md#ai-assistance)
