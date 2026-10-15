from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PriorConfig, SamplerConfig
from .models import (
    AMP_IDX,
    N_BACKGROUND_PARAMS,
    N_FLARE_PARAMS,
    N_HARMONICS,
    PERIOD_IDX,
    PHASE_IDX,
    full_model,
    split_model_components,
)
from .optimize import LocalOptimum, interior, multistart_least_squares, parameter_bounds
from .preprocess import estimate_noise_savgol


@dataclass
class FitResult:
    t: np.ndarray
    y: np.ndarray
    n_flares: int
    flare_indices: np.ndarray
    posterior_samples: np.ndarray
    map_params: np.ndarray
    map_log_likelihood: float
    map_model: np.ndarray
    map_trend: np.ndarray
    map_flares: np.ndarray
    sigma_map: float
    ppd_lower: np.ndarray
    ppd_upper: np.ndarray
    confidence_level: float
    idata: Any | None = None
    diagnostics: dict[str, Any] | None = None
    map_source: str = "mcmc"


def initial_values(
    t: np.ndarray,
    y: np.ndarray,
    flare_indices: np.ndarray,
    noise_estimate: float,
    priors: PriorConfig,
) -> tuple[np.ndarray, float]:
    """Generic IDL-style starting state.

    Returns ``(theta, sigma)`` with ``theta`` in the flat IDL/SoBAT parameter
    order (without sigma), pushed just inside the prior bounds.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    flare_indices = np.asarray(flare_indices, dtype=int)
    k = int(flare_indices.size)

    span = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    a1 = max(0.5 * float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    p1 = 4.0 * span
    a2 = 0.1 * a1

    amps = np.array([a1, a2, 0.5 * a2, 0.25 * a2], dtype=float)
    if k:
        periods = np.array([p1, p1 / 2.0, p1 / 3.0, max(p1 / 4.0, 50.0)])
    else:
        p2 = 0.5 * p1
        periods = np.array([p1, p2, 0.5 * p2, 0.25 * p2])

    theta = np.zeros(N_BACKGROUND_PARAMS + N_FLARE_PARAMS * k)
    theta[0] = np.nanmean(y)
    theta[AMP_IDX] = amps
    theta[PERIOD_IDX] = periods
    theta[PHASE_IDX] = 0.0
    if k:
        f = theta[N_BACKGROUND_PARAMS:].reshape(k, N_FLARE_PARAMS)
        f[:, 1] = t[flare_indices]  # peak time
        f[:, 2] = 1.0  # rise time
        f[:, 3] = 10.0  # decay time

    lo, hi = parameter_bounds(t, flare_indices, noise_estimate, priors)
    eps = 1e-6 * np.maximum(1.0, hi - lo)
    theta = np.clip(theta, lo + eps, hi - eps)
    if k:  # small positive flare amplitudes (set after clipping, not clipped)
        theta[N_BACKGROUND_PARAMS::N_FLARE_PARAMS] = max(0.1 * noise_estimate, 1e-8)

    sigma_high = max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    sigma_eps = 1e-6 * max(1.0, sigma_high)
    sigma = float(np.clip(noise_estimate, sigma_eps, sigma_high - sigma_eps))
    return theta, sigma


def _ordered_periods(periods: np.ndarray, low: float, high: float) -> np.ndarray:
    """Sort periods ascending and make them strictly increasing inside (low, high)."""
    p = np.sort(np.asarray(periods, dtype=float))
    eps = 1e-6 * (high - low)
    p = np.clip(p, low + eps, high - 5 * eps)
    for j in range(1, p.size):
        if p[j] <= p[j - 1] + eps:
            p[j] = p[j - 1] + eps
    return p


def model_start(
    theta: np.ndarray,
    sigma: float,
    n_flares: int,
    *,
    period_low: float,
    period_high: float,
    config: SamplerConfig,
) -> dict[str, np.ndarray | float]:
    """Convert a flat parameter vector into PyMC `initvals`.

    Handles the cartesian harmonic parametrisation and the period ordering.
    """
    th = np.asarray(theta, dtype=float)
    amp = th[AMP_IDX]
    per = th[PERIOD_IDX]
    pha = th[PHASE_IDX]
    d: dict[str, np.ndarray | float] = {"offset": float(th[0]), "sigma": float(sigma)}
    if config.order_periods:
        order = np.argsort(per)
        amp, pha = amp[order], pha[order]
        d["harmonic_period"] = _ordered_periods(per, period_low, period_high)
    else:
        d["harmonic_period"] = per.copy()
    if config.cartesian_harmonics:
        d["b_sin"] = amp * np.cos(pha)
        d["b_cos"] = amp * np.sin(pha)
    else:
        d["harmonic_amplitude"] = amp.copy()
        d["harmonic_phase"] = pha.copy()
    if n_flares:
        f = th[N_BACKGROUND_PARAMS:].reshape(n_flares, N_FLARE_PARAMS)
        d["flare_amplitude"] = f[:, 0].copy()
        d["flare_peak_time"] = f[:, 1].copy()
        d["flare_rise_time"] = f[:, 2].copy()
        d["flare_decay_time"] = f[:, 3].copy()
    return d


def build_pymc_model(
    t: np.ndarray,
    y: np.ndarray,
    flare_indices: np.ndarray | list[int] | None = None,
    *,
    noise_estimate: float | None = None,
    priors: PriorConfig | None = None,
    config: SamplerConfig | None = None,
):
    """Build the PyMC model with the priors and Gaussian likelihood.

    Sampling-friendly reparametrisations (switchable in `SamplerConfig`):

    * ``cartesian_harmonics``: ``A sin(wt + phi) = b_sin sin(wt) + b_cos cos(wt)``.
      Removes the phase wrap-around at +-pi and the polar singularity at A = 0,
      both of which trap random-walk samplers.  With
      ``priors.amplitude_prior == "idl"`` a ``-log A`` potential restores the
      IDL prior (uniform in A in [0, A_max] and in phi) exactly.
    * ``order_periods``: periods sorted ascending.  The four harmonics are
      exchangeable, so without this the posterior has 4! = 24 identical copies
      of every mode (label switching).  It does not change the fitted curve.
    """
    import pymc as pm
    import pytensor.tensor as pt

    priors = priors or PriorConfig()
    config = config or SamplerConfig()
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    flare_indices = np.asarray(flare_indices if flare_indices is not None else [], dtype=int)
    n_flares = int(flare_indices.size)

    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("t and y must be one-dimensional arrays of equal length.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("t and y must contain only finite values before fitting.")
    if n_flares and (flare_indices.min() < 0 or flare_indices.max() >= t.size):
        raise IndexError("flare_indices contains an out-of-range index.")

    if noise_estimate is None:
        noise_estimate, _ = estimate_noise_savgol(y)

    theta0, sigma0 = initial_values(t, y, flare_indices, noise_estimate, priors)
    period_low = (
        priors.harmonic_period_low_with_flares
        if n_flares
        else priors.harmonic_period_low_without_flares
    )
    period_high = priors.harmonic_period_high
    a_max = priors.harmonic_amplitude_high
    sigma_high = max(float(np.ptp(y)), 1e-6)
    start = model_start(
        theta0, sigma0, n_flares,
        period_low=period_low, period_high=period_high, config=config,
    )

    with pm.Model() as model:
        t_data = pm.Data("t", t)

        offset = pm.Uniform(
            "offset", priors.offset_low, priors.offset_high, initval=start["offset"]
        )
        period_kw = (
            {"transform": pm.distributions.transforms.ordered} if config.order_periods else {}
        )
        period = pm.Uniform(
            "harmonic_period", period_low, period_high, shape=N_HARMONICS,
            initval=start["harmonic_period"], **period_kw,
        )
        omega_t = 2.0 * np.pi * t_data[None, :] / period[:, None]

        if config.cartesian_harmonics:
            b_sin = pm.Uniform("b_sin", -a_max, a_max, shape=N_HARMONICS, initval=start["b_sin"])
            b_cos = pm.Uniform("b_cos", -a_max, a_max, shape=N_HARMONICS, initval=start["b_cos"])
            amp = pm.Deterministic("harmonic_amplitude", pt.sqrt(b_sin**2 + b_cos**2))
            pm.Deterministic("harmonic_phase", pt.arctan2(b_cos, b_sin))
            # restrict to the disc A <= A_max (the IDL amplitude range)
            pm.Potential("amplitude_bound", pt.switch(pt.all(amp <= a_max), 0.0, -np.inf))
            if priors.amplitude_prior == "idl":
                pm.Potential("uniform_amplitude_prior", -pt.sum(pt.log(amp + 1e-300)))
            elif priors.amplitude_prior != "flat_cartesian":
                raise ValueError(f"Unknown amplitude_prior {priors.amplitude_prior!r}")
            harmonic_terms = b_sin[:, None] * pt.sin(omega_t) + b_cos[:, None] * pt.cos(omega_t)
        else:
            amp = pm.Uniform(
                "harmonic_amplitude", priors.harmonic_amplitude_low, a_max, shape=N_HARMONICS,
                initval=start["harmonic_amplitude"],
            )
            phase = pm.Uniform(
                "harmonic_phase", priors.phase_low, priors.phase_high, shape=N_HARMONICS,
                initval=start["harmonic_phase"],
            )
            harmonic_terms = amp[:, None] * pt.sin(omega_t + phase[:, None])

        mu = offset + pt.sum(harmonic_terms, axis=0)

        if n_flares:
            peak_guess = t[flare_indices]
            flare_amp = pm.Uniform(
                "flare_amplitude",
                0.0,
                max(priors.flare_amplitude_sigma_factor * noise_estimate, 1e-8),
                shape=n_flares,
                initval=start["flare_amplitude"],
            )
            flare_peak = pm.Uniform(
                "flare_peak_time",
                peak_guess - priors.flare_peak_before,
                peak_guess + priors.flare_peak_after,
                shape=n_flares,
                initval=start["flare_peak_time"],
            )
            flare_rise = pm.Uniform(
                "flare_rise_time",
                priors.flare_rise_low,
                priors.flare_rise_high,
                shape=n_flares,
                initval=start["flare_rise_time"],
            )
            flare_decay = pm.Uniform(
                "flare_decay_time",
                priors.flare_decay_low,
                priors.flare_decay_high,
                shape=n_flares,
                initval=start["flare_decay_time"],
            )

            dt = t_data[None, :] - flare_peak[:, None]
            # Both exponentials are made numerically safe even if the backend
            # evaluates both branches of the switch.
            rise_exp = pt.exp(pt.minimum(dt, 0.0) / (flare_rise[:, None] + 1e-20))
            decay_exp = pt.exp(-pt.maximum(dt, 0.0) / flare_decay[:, None])
            flare_terms = flare_amp[:, None] * pt.switch(dt < 0.0, rise_exp, decay_exp)
            mu = mu + pt.sum(flare_terms, axis=0)

        sigma = pm.Uniform("sigma", 0.0, sigma_high, initval=start["sigma"])
        pm.Normal("obs", mu=mu, sigma=sigma + 1e-20, observed=y)

    return model


def _posterior_parameter_matrix(posterior: Any, n_flares: int) -> np.ndarray:
    """Flatten a PyMC posterior (``idata.posterior``) into SoBAT parameter order."""
    offset = np.asarray(posterior["offset"].values).reshape(-1, 1)
    amp = np.asarray(posterior["harmonic_amplitude"].values).reshape(-1, N_HARMONICS)
    period = np.asarray(posterior["harmonic_period"].values).reshape(-1, N_HARMONICS)
    phase = np.asarray(posterior["harmonic_phase"].values).reshape(-1, N_HARMONICS)

    cols = [offset]
    for j in range(N_HARMONICS):
        cols.extend([amp[:, j : j + 1], period[:, j : j + 1], phase[:, j : j + 1]])

    if n_flares:
        flare_amp = np.asarray(posterior["flare_amplitude"].values).reshape(-1, n_flares)
        flare_peak = np.asarray(posterior["flare_peak_time"].values).reshape(-1, n_flares)
        flare_rise = np.asarray(posterior["flare_rise_time"].values).reshape(-1, n_flares)
        flare_decay = np.asarray(posterior["flare_decay_time"].values).reshape(-1, n_flares)
        for k in range(n_flares):
            cols.extend(
                [
                    flare_amp[:, k : k + 1],
                    flare_peak[:, k : k + 1],
                    flare_rise[:, k : k + 1],
                    flare_decay[:, k : k + 1],
                ]
            )

    sigma = np.asarray(posterior["sigma"].values).reshape(-1, 1)
    cols.append(sigma)
    return np.hstack(cols)


def _log_likelihoods(
    t: np.ndarray, y: np.ndarray, samples: np.ndarray, n_flares: int, chunk: int = 2000
) -> np.ndarray:
    """Gaussian log likelihood for every posterior row in physical coordinates.

    Vectorised over blocks of rows.
    """
    t = np.asarray(t, dtype=float)
    out = np.empty(samples.shape[0], dtype=float)
    n = t.size
    for a in range(0, samples.shape[0], chunk):
        s = samples[a : a + chunk]
        mu = np.broadcast_to(s[:, [0]], (s.shape[0], n)).copy()
        for j in range(N_HARMONICS):
            i = 1 + 3 * j
            mu += s[:, [i]] * np.sin(2.0 * np.pi * t[None, :] / s[:, [i + 1]] + s[:, [i + 2]])
        for k in range(n_flares):
            i = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * k
            dt = t[None, :] - s[:, [i + 1]]
            rise = np.exp(np.minimum(dt, 0.0) / (s[:, [i + 2]] + 1e-300))
            decay = np.exp(-np.maximum(dt, 0.0) / s[:, [i + 3]])
            mu += s[:, [i]] * np.where(dt < 0.0, rise, decay)
        sigma = s[:, [-1]] + 1e-20
        resid = y[None, :] - mu
        out[a : a + chunk] = -0.5 * np.sum(
            np.log(2.0 * np.pi * sigma**2) + resid**2 / sigma**2, axis=1
        )
    return out


def _chain_starts(
    optima: list[LocalOptimum],
    fallback: tuple[np.ndarray, float],
    t: np.ndarray,
    y: np.ndarray,
    flare_indices: np.ndarray,
    noise_estimate: float,
    priors: PriorConfig,
    config: SamplerConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Chain starts drawn around the best optimum.

    Only optima whose log-likelihood is within ``config.mode_tolerance`` of the
    best one are used (cycled over chains): a mode 10 log-units worse carries
    ~e^-10 of the posterior mass, and a chain started there just wastes the
    run and inflates R-hat.  Each start is jittered by
    ``config.start_jitter`` x the local (Laplace) standard deviation of every
    parameter, so chains start *inside* the mode rather than far outside it.
    Without optima, every chain starts at ``fallback = (theta, sigma)``.
    """
    n_flares = flare_indices.size
    n = y.size
    lo, hi = parameter_bounds(t, flare_indices, noise_estimate, priors)
    period_low, period_high = lo[2], hi[2]
    sigma_high = max(float(np.ptp(y)), 1e-6)
    if not optima:
        return [
            model_start(*fallback, n_flares,
                        period_low=period_low, period_high=period_high, config=config)
        ] * config.chains

    ll = np.array([-0.5 * n * (np.log(2 * np.pi * o.rms**2) + 1.0) for o in optima])
    good = [o for o, l in zip(optima, ll) if l >= ll[0] - config.mode_tolerance]
    starts = []
    for c in range(config.chains):
        opt = good[c % len(good)]
        x = opt.theta + config.start_jitter * opt.sd * rng.normal(size=opt.theta.size)
        x = interior(x, lo, hi, 1e-6)
        sigma = float(np.clip(opt.rms, 1e-6 * sigma_high, 0.999 * sigma_high))
        starts.append(
            model_start(x, sigma, n_flares, period_low=period_low,
                        period_high=period_high, config=config)
        )
    return starts


def _posterior_predictive_intervals(
    t: np.ndarray,
    samples: np.ndarray,
    n_flares: int,
    *,
    confidence_level: float,
    n_draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_model = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * n_flares
    n_available = samples.shape[0]
    if n_available == 0:
        raise ValueError("No posterior samples available.")

    n_draws = min(int(n_draws), n_available)
    idx = rng.choice(n_available, size=n_draws, replace=False)
    ppd = np.empty((n_draws, t.size), dtype=float)
    for j, i in enumerate(idx):
        row = samples[i]
        mu = full_model(t, row[:n_model], n_flares)
        ppd[j] = mu + row[-1] * rng.normal(size=t.size)

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(ppd, [alpha, 1.0 - alpha], axis=0)
    return lower, upper


def _split_rhat_ess(x: np.ndarray) -> tuple[float, float]:
    """Rank-normalised split R-hat (bulk + folded, Vehtari et al. 2021) and bulk ESS.

    `x` has shape (chains, draws).  Lightweight, so it can be applied to many
    derived quantities without arviz.
    """
    from scipy.special import ndtri
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    c, n = x.shape
    half = n // 2
    if c * half < 8 or not np.all(np.isfinite(x)):
        return np.nan, np.nan
    xs = np.concatenate([x[:, :half], x[:, n - half :]], axis=0)  # split chains

    def _rhat_ess(z):
        m, k = z.shape
        r = rankdata(z, method="average").reshape(z.shape)
        z = ndtri((r - 0.375) / (z.size + 0.25))  # rank-normalise
        means = z.mean(axis=1)
        w = z.var(axis=1, ddof=1).mean()
        b = k * means.var(ddof=1)
        if w <= 0:
            return np.nan, np.nan
        var_plus = (k - 1) / k * w + b / k
        rhat = np.sqrt(var_plus / w)
        # ESS via Geyer initial positive sequence on the averaged autocorrelation
        zc = z - means[:, None]
        f = np.fft.rfft(zc, n=2 * k, axis=1)
        acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :k] / k
        rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
        rho[0] = 1.0
        s = 0.0
        for i in range(0, k - 1, 2):
            pair = rho[i] + rho[i + 1]
            if pair < 0:
                break
            s += pair
        tau = max(-1.0 + 2.0 * s, 1.0 / np.log10(m * k))
        return float(rhat), float(m * k / tau)

    r_bulk, ess = _rhat_ess(xs)
    r_fold, _ = _rhat_ess(np.abs(xs - np.median(xs)))
    return float(np.nanmax([r_bulk, r_fold])), ess


def _convergence_diagnostics(
    idata: Any,
    t: np.ndarray,
    n_flares: int,
    chains: int,
    *,
    n_grid: int = 25,
    max_draws: int = 2000,
) -> dict[str, Any]:
    """Convergence diagnostics on parameters *and* on the fitted curves.

    Individual sinusoid parameters are often weakly identified (e.g. a long
    period trading off against amplitude and offset) while the *curve* they
    produce is pinned down by the data.  Since the outputs used downstream are
    the trend and flare curves, `converged` is judged on

    * ``rhat_curves_max`` / ``ess_curves_min``: trend, summed flares and full
      model evaluated on ``n_grid`` times, plus ``sigma``;

    while ``rhat_params_max`` / ``ess_params_min`` (all free parameters) are
    reported for information.
    """
    out: dict[str, Any] = {
        "rhat_curves_max": np.nan, "ess_curves_min": np.nan,
        "rhat_params_max": np.nan, "ess_params_min": np.nan,
        "trend_chain_spread_over_sigma": np.nan,
        "flares_chain_spread_over_sigma": np.nan,
        "model_chain_spread_over_sigma": np.nan,
        "converged": False,
    }
    if chains < 2:
        return out
    try:
        post = idata.posterior
        n_draw = post.sizes["draw"]
        step = max(1, n_draw // max_draws)
        sub = post.isel(draw=slice(None, None, step))
        samples = _posterior_parameter_matrix(sub, n_flares)  # (chain*draw, p)
        c, d = sub.sizes["chain"], sub.sizes["draw"]

        # parameter-level
        derived = ("harmonic_amplitude", "harmonic_phase") if "b_sin" in sub else ()
        free = [v for v in sub.data_vars if v not in derived]
        par_r, par_e = [], []
        for v in free:
            arr = np.asarray(sub[v].values).reshape(c, d, -1)
            for j in range(arr.shape[2]):
                r, e = _split_rhat_ess(arr[:, :, j])
                par_r.append(r)
                par_e.append(e)

        # curve-level
        tg = np.linspace(t.min(), t.max(), n_grid)
        n_model = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * n_flares
        trend = np.empty((samples.shape[0], n_grid))
        flares = np.empty_like(trend)
        for i, row in enumerate(samples):
            trend[i], flares[i] = split_model_components(tg, row[:n_model], n_flares)
        cur_r, cur_e = [], []
        blocks = [trend, trend + flares, samples[:, [-1]]]
        if n_flares:
            blocks.append(flares)
        for blk in blocks:
            blk = blk.reshape(c, d, -1)
            for j in range(blk.shape[2]):
                if np.ptp(blk[:, :, j]) == 0:
                    continue
                r, e = _split_rhat_ess(blk[:, :, j])
                cur_r.append(r)
                cur_e.append(e * step)
        # Practical check: how far apart are the chains' mean curves, in units
        # of the noise sigma?  (R-hat can be large even when the chains agree to
        # a small fraction of the noise, because within-chain spread is tiny.)
        sig = float(np.median(samples[:, -1]))
        for name, blk in (("trend", trend), ("flares", flares), ("model", trend + flares)):
            cm = blk.reshape(c, d, -1).mean(axis=1)
            out[f"{name}_chain_spread_over_sigma"] = float(np.max(np.ptp(cm, axis=0)) / sig)
        out["rhat_params_max"] = float(np.nanmax(par_r))
        out["ess_params_min"] = float(np.nanmin(par_e)) * step
        out["rhat_curves_max"] = float(np.nanmax(cur_r))
        out["ess_curves_min"] = float(np.nanmin(cur_e))
        out["converged"] = bool(out["rhat_curves_max"] < 1.05 and out["ess_curves_min"] > 100)
    except Exception:  # diagnostics must never break a fit
        pass
    return out


def fit_curve(
    t: np.ndarray,
    y: np.ndarray,
    flare_indices: np.ndarray | list[int] | None = None,
    *,
    config: SamplerConfig | None = None,
    priors: PriorConfig | None = None,
    noise_estimate: float | None = None,
    keep_idata: bool = False,
) -> FitResult:
    """Fit one curve with PyMC and return SoBAT-compatible high-level outputs."""
    import pymc as pm

    config = config or SamplerConfig()
    priors = priors or PriorConfig()
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    flare_indices = np.asarray(flare_indices if flare_indices is not None else [], dtype=int)
    n_flares = int(flare_indices.size)

    if noise_estimate is None:
        noise_estimate, _ = estimate_noise_savgol(y)

    theta0, sigma0 = initial_values(t, y, flare_indices, noise_estimate, priors)
    model = build_pymc_model(
        t,
        y,
        flare_indices,
        noise_estimate=noise_estimate,
        priors=priors,
        config=config,
    )

    rng = np.random.default_rng(config.random_seed + 1)
    optima: list[LocalOptimum] = []
    if config.n_starts > 0:
        optima = multistart_least_squares(
            t,
            y,
            flare_indices,
            noise_estimate,
            priors,
            first_guess=theta0,
            n_starts=config.n_starts,
            max_nfev=config.max_nfev,
            rng=rng,
        )
    chain_inits = _chain_starts(
        optima, (theta0, sigma0), t, y, flare_indices, noise_estimate, priors, config, rng
    )

    with model:
        step = None
        if config.sampler == "demetropolisz":
            step = pm.DEMetropolisZ()
        elif config.sampler != "nuts":
            raise ValueError(f"Unknown sampler {config.sampler!r}")

        sample_kwargs: dict[str, Any] = dict(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            random_seed=config.random_seed,
            progressbar=config.progressbar,
            initvals=chain_inits,
            return_inferencedata=True,
            compute_convergence_checks=False,
        )
        if step is not None:
            sample_kwargs["step"] = step
        else:
            sample_kwargs["nuts"] = {"target_accept": config.target_accept}

        idata = pm.sample(**sample_kwargs)

    samples = _posterior_parameter_matrix(idata.posterior, n_flares)
    loglike = _log_likelihoods(t, y, samples, n_flares)
    i_map = int(np.nanargmax(loglike))
    map_params = samples[i_map].copy()
    map_ll = float(loglike[i_map])
    map_source = "mcmc"

    if config.polish_map and optima:
        cand = np.r_[optima[0].theta, max(optima[0].rms, 1e-12)][None, :]
        ll_opt = float(_log_likelihoods(t, y, cand, n_flares)[0])
        if ll_opt > map_ll:
            map_params, map_ll, map_source = cand[0].copy(), ll_opt, "optimizer"

    n_model = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * n_flares
    map_model = full_model(t, map_params[:n_model], n_flares)
    map_trend, map_flares = split_model_components(t, map_params[:n_model], n_flares)
    diagnostics = _convergence_diagnostics(idata, t, n_flares, config.chains)

    rng = np.random.default_rng(config.random_seed + 137)
    ppd_lower, ppd_upper = _posterior_predictive_intervals(
        t,
        samples,
        n_flares,
        confidence_level=config.confidence_level,
        n_draws=config.ppd_draws,
        rng=rng,
    )

    return FitResult(
        t=t,
        y=y,
        n_flares=n_flares,
        flare_indices=flare_indices.copy(),
        posterior_samples=samples,
        map_params=map_params,
        map_log_likelihood=map_ll,
        map_model=map_model,
        map_trend=map_trend,
        map_flares=map_flares,
        sigma_map=float(map_params[-1]),
        ppd_lower=ppd_lower,
        ppd_upper=ppd_upper,
        confidence_level=config.confidence_level,
        idata=idata if keep_idata else None,
        diagnostics=diagnostics,
        map_source=map_source,
    )
