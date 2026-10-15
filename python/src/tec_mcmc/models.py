from __future__ import annotations

import numpy as np

N_HARMONICS = 4
N_BACKGROUND_PARAMS = 13
N_FLARE_PARAMS = 4

# Column indices of the harmonic parameters in the flat parameter vector.
AMP_IDX = 1 + 3 * np.arange(N_HARMONICS)
PERIOD_IDX = AMP_IDX + 1
PHASE_IDX = AMP_IDX + 2


def harmonic_background(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Four-sinusoid background used by the IDL routine.

    Parameter layout:
        [c0,
         A1, P1, phi1,
         A2, P2, phi2,
         A3, P3, phi3,
         A4, P4, phi4]
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(params, dtype=float)
    if p.shape[-1] < N_BACKGROUND_PARAMS:
        raise ValueError("Need at least 13 background parameters.")

    out = np.full_like(t, p[0], dtype=float)
    for j in range(N_HARMONICS):
        i = 1 + 3 * j
        amplitude, period, phase = p[i : i + 3]
        if period <= 0:
            raise ValueError("Harmonic periods must be positive.")
        out += amplitude * np.sin(2.0 * np.pi * t / period + phase)
    return out


def flare_profile(
    t: np.ndarray,
    amplitude: float,
    peak_time: float,
    rise_time: float,
    decay_time: float,
) -> np.ndarray:
    """Asymmetric exponential flare profile from the IDL routine."""
    t = np.asarray(t, dtype=float)
    if rise_time <= 0 or decay_time <= 0:
        raise ValueError("Flare rise and decay times must be positive.")

    dt = t - peak_time
    out = np.empty_like(t, dtype=float)
    left = dt < 0
    out[left] = amplitude * np.exp(dt[left] / rise_time)
    out[~left] = amplitude * np.exp(-dt[~left] / decay_time)
    return out


def _add_flares(out: np.ndarray, t: np.ndarray, p: np.ndarray, n_flares: int) -> np.ndarray:
    """Add the `n_flares` flare profiles in `p` to `out` in place (fixed order)."""
    for k in range(n_flares):
        i = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * k
        out += flare_profile(t, *p[i : i + N_FLARE_PARAMS])
    return out


def full_model(t: np.ndarray, params: np.ndarray, n_flares: int) -> np.ndarray:
    """Background plus `n_flares` asymmetric exponential components.

    `params` excludes the observational-noise sigma. If a vector including
    sigma is passed, slice it first with `params[:-1]`.
    """
    p = np.asarray(params, dtype=float)
    expected = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * n_flares
    if p.size != expected:
        raise ValueError(f"Expected {expected} model parameters, got {p.size}.")

    return _add_flares(harmonic_background(t, p[:N_BACKGROUND_PARAMS]), t, p, n_flares)


def split_model_components(
    t: np.ndarray, params: np.ndarray, n_flares: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(background, summed_flares)` for a model parameter vector."""
    p = np.asarray(params, dtype=float)
    background = harmonic_background(t, p[:N_BACKGROUND_PARAMS])
    flares = _add_flares(np.zeros_like(np.asarray(t, dtype=float)), t, p, n_flares)
    return background, flares


def unpack_parameter_matrix(
    matrix: np.ndarray, n_flares: int
) -> dict[str, np.ndarray]:
    """Unpack rows in the SoBAT-compatible parameter order.

    The final column is observational-noise sigma.
    """
    m = np.asarray(matrix, dtype=float)
    expected = N_BACKGROUND_PARAMS + N_FLARE_PARAMS * n_flares + 1
    if m.ndim != 2 or m.shape[1] != expected:
        raise ValueError(f"Expected shape (n_samples, {expected}).")

    out: dict[str, np.ndarray] = {
        "offset": m[:, 0],
        "harmonic_amplitude": m[:, AMP_IDX],
        "harmonic_period": m[:, PERIOD_IDX],
        "harmonic_phase": m[:, PHASE_IDX],
        "sigma": m[:, -1],
    }
    if n_flares:
        flare = m[:, N_BACKGROUND_PARAMS:-1].reshape(m.shape[0], n_flares, N_FLARE_PARAMS)
        out["flare_amplitude"] = flare[:, :, 0]
        out["flare_peak_time"] = flare[:, :, 1]
        out["flare_rise_time"] = flare[:, :, 2]
        out["flare_decay_time"] = flare[:, :, 3]
    return out
