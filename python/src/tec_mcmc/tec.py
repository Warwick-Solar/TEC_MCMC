from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import interpolate_collapsed_plateaus


def load_tec_csv(
    path: str | Path,
    *,
    time_column: int = 0,
    time_scale: float = 1.0 / 60.0,
    header: int | None | str = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load the first CSV column as time and all remaining columns as curves."""
    df = pd.read_csv(path, header=header)
    raw = df.to_numpy(dtype=float)
    t_full = raw[:, time_column] * time_scale

    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(raw.shape[1]):
        if j == time_column:
            continue
        y = raw[:, j]
        good = np.isfinite(t_full) & np.isfinite(y)
        curves.append((t_full[good], y[good]))
    return curves


def preprocess_tec_curve(
    t: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Plateau preprocessing plus time shift used before MCMC.

    Returns `(t_relative, y_processed, absolute_start_time)`.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    y2 = interpolate_collapsed_plateaus(t, y, fill_edges=True)
    t0 = float(np.min(t))
    return t - t0, y2, t0


def prepare_curve(
    t_abs: np.ndarray,
    y_raw: np.ndarray,
    t_peaks_solar,
    *,
    rng: np.random.Generator,
):
    """Preprocess a curve before MCMC fit.

    Returns ``(t_rel, y, flare_indices, true_indices, t0)`` where
    ``flare_indices`` already contains the dummy flare (if any, see
    `add_dummy_flare_candidate`).
    """
    from .preprocess import add_dummy_flare_candidate, find_peak_indices

    t_rel, y, t0 = preprocess_tec_curve(t_abs, y_raw)
    true_idx = find_peak_indices(np.asarray(t_abs, float), t_peaks_solar, tolerance=_half_dt(t_rel))
    flare_idx, true_idx = add_dummy_flare_candidate(len(t_rel), true_idx, rng=rng)
    return t_rel, y, flare_idx, true_idx, t0


def _half_dt(t: np.ndarray) -> float:
    """Tolerance ``0.5 * (t[1] - t[0])`` (first sampling step)."""
    return 0.5 * float(t[1] - t[0]) if len(t) > 1 else np.inf
