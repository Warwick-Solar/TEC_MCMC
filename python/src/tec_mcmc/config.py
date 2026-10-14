from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplerConfig:
    """Sampling and posterior-predictive settings.

    `cores=1` is deliberate: when many independent curves are fitted, use
    process-level parallelism across curves instead of nested multiprocessing.
    """

    draws: int = 20000
    tune: int = 20000
    chains: int = 4
    cores: int = 1
    sampler: Literal["demetropolisz", "nuts"] = "demetropolisz"
    target_accept: float = 0.9
    random_seed: int = 42
    progressbar: bool = True
    confidence_level: float = 0.95
    ppd_draws: int = 5000

    # Multi-start least-squares search used to initialise the chains (see
    # `optimize.py`).  Set `n_starts=0` to start from the IDL initial guess only.
    n_starts: int = 30
    max_nfev: int = 300
    # Chains start at the best optimum (or at any optimum whose log-likelihood
    # is within `mode_tolerance` of it), jittered by `start_jitter` x the local
    # Laplace standard deviation of each parameter.
    mode_tolerance: float = 3.0
    start_jitter: float = 0.5
    # Sampling-friendly reparametrisations (do not change the fitted curve):
    # b_sin/b_cos instead of amplitude/phase, and periods sorted ascending to
    # remove the 4! label-switching copies of every mode.
    cartesian_harmonics: bool = True
    order_periods: bool = True
    # Also offer the best optimiser solution as a candidate for the returned
    # best-fit state (it is kept only if it has a higher likelihood than every
    # MCMC sample).
    polish_map: bool = True


@dataclass(frozen=True)
class PriorConfig:
    """Prior bounds."""

    offset_low: float = -500.0
    offset_high: float = 500.0

    harmonic_amplitude_low: float = 0.0
    harmonic_amplitude_high: float = 500.0
    harmonic_period_high: float = 24.0 * 60.0
    harmonic_period_low_with_flares: float = 50.0
    harmonic_period_low_without_flares: float = 1.0
    phase_low: float = -3.141592653589793
    phase_high: float = 3.141592653589793
    # "idl": uniform in amplitude and phase exactly as in the IDL script;
    # "flat_cartesian": uniform in (b_sin, b_cos), i.e. p(A) proportional to A.
    # Only used with SamplerConfig.cartesian_harmonics=True.
    amplitude_prior: str = "idl"

    flare_amplitude_sigma_factor: float = 20.0
    flare_peak_before: float = 1.0
    flare_peak_after: float = 10.0
    flare_rise_low: float = 0.0
    flare_rise_high: float = 5.0
    flare_decay_low: float = 2.0
    flare_decay_high: float = 100.0
