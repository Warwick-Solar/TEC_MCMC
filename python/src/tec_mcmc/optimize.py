"""Multi-start bounded least-squares search for good starting states.

The posterior of a four-sinusoid + flare model with free periods is strongly
multimodal.  A gradient-free Metropolis sampler started from the (generic) IDL
initial guess rarely finds the best mode in a reasonable number of iterations,
so we first locate good local optima with many cheap least-squares fits.  They
are used (i) to start the MCMC chains and (ii) as extra candidates for the
returned best-fit ("MAP") state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .config import PriorConfig
from .models import N_BACKGROUND_PARAMS, N_FLARE_PARAMS, full_model


@dataclass
class LocalOptimum:
    theta: np.ndarray  # model parameters (no sigma), IDL/SoBAT order
    rms: float  # root-mean-square residual == ML estimate of sigma
    sd: np.ndarray  # Laplace (Gauss-Newton) standard deviation of each parameter


def parameter_bounds(
    t: np.ndarray,
    flare_indices: np.ndarray,
    noise_estimate: float,
    priors: PriorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior bounds in the flat parameter order (identical to the PyMC priors)."""
    k = len(flare_indices)
    period_low = (
        priors.harmonic_period_low_with_flares
        if k
        else priors.harmonic_period_low_without_flares
    )
    lo = [priors.offset_low]
    hi = [priors.offset_high]
    for _ in range(4):
        lo += [priors.harmonic_amplitude_low, period_low, priors.phase_low]
        hi += [priors.harmonic_amplitude_high, priors.harmonic_period_high, priors.phase_high]
    for i in np.asarray(flare_indices, dtype=int):
        lo += [
            0.0,
            t[i] - priors.flare_peak_before,
            priors.flare_rise_low,
            priors.flare_decay_low,
        ]
        hi += [
            max(priors.flare_amplitude_sigma_factor * noise_estimate, 1e-8),
            t[i] + priors.flare_peak_after,
            priors.flare_rise_high,
            priors.flare_decay_high,
        ]
    return np.asarray(lo, float), np.asarray(hi, float)


def _random_start(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    n_flares: int,
    noise_estimate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.empty_like(lo)
    x[0] = np.mean(y)
    half_range = max(0.5 * float(np.ptp(y)), 1e-6)
    for j in range(4):
        i = 1 + 3 * j
        x[i] = rng.uniform(0.0, half_range)
        x[i + 1] = np.exp(rng.uniform(np.log(lo[i + 1]), np.log(hi[i + 1])))
        x[i + 2] = rng.uniform(lo[i + 2], hi[i + 2])
    for k in range(n_flares):
        i = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * k
        x[i] = rng.uniform(0.0, 0.5 * hi[i])
        x[i + 1] = rng.uniform(lo[i + 1], hi[i + 1])
        x[i + 2] = rng.uniform(0.1, hi[i + 2])
        x[i + 3] = np.exp(rng.uniform(np.log(lo[i + 3]), np.log(hi[i + 3])))
    return x


def _laplace_sd(jac: np.ndarray, rms: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """sqrt(diag(rms^2 (J^T J)^-1)), clipped to [1e-9, 1e-2] x prior range.

    Directions that the data do not constrain (singular J^T J, e.g. the period
    of a harmonic with ~zero amplitude) get the upper clip value.
    """
    rng_ = hi - lo
    try:
        cov = rms**2 * np.linalg.pinv(jac.T @ jac, rcond=1e-12)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sd = np.full(lo.size, np.inf)
    sd = np.where(np.isfinite(sd), sd, np.inf)
    return np.clip(sd, 1e-9 * rng_, 1e-2 * rng_)


def interior(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, frac: float) -> np.ndarray:
    """Clip `x` to lie at least `frac` of the range inside the bounds."""
    margin = frac * (hi - lo)
    return np.clip(x, lo + margin, hi - margin)


def multistart_least_squares(
    t: np.ndarray,
    y: np.ndarray,
    flare_indices: np.ndarray,
    noise_estimate: float,
    priors: PriorConfig,
    *,
    first_guess: np.ndarray | None = None,
    n_starts: int = 30,
    max_nfev: int = 300,
    rng: np.random.Generator | None = None,
) -> list[LocalOptimum]:
    """Return local optima sorted by residual RMS (best first)."""
    rng = rng or np.random.default_rng(0)
    flare_indices = np.asarray(flare_indices, dtype=int)
    k = flare_indices.size
    lo, hi = parameter_bounds(t, flare_indices, noise_estimate, priors)

    def resid(theta):
        return full_model(t, theta, k) - y

    starts = []
    if first_guess is not None:
        starts.append(np.asarray(first_guess, float))
    while len(starts) < max(n_starts, 1):
        starts.append(_random_start(y, lo, hi, k, noise_estimate, rng))

    out: list[LocalOptimum] = []
    for x0 in starts:
        try:
            sol = least_squares(
                resid,
                interior(x0, lo, hi, 1e-3),
                bounds=(lo, hi),
                method="trf",
                max_nfev=max_nfev,
            )
        except (ValueError, FloatingPointError):
            continue
        rms = float(np.sqrt(np.mean(sol.fun**2)))
        if np.isfinite(rms):
            out.append(LocalOptimum(sol.x.copy(), rms, _laplace_sd(sol.jac, rms, lo, hi)))
    out.sort(key=lambda o: o.rms)
    return out
