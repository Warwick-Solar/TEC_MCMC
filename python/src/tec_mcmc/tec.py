from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import (
    add_dummy_flare_candidate,
    find_peak_indices,
    interpolate_collapsed_plateaus,
)


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


def prepare_curve(
    t_abs: np.ndarray,
    y_raw: np.ndarray,
    t_peaks_solar,
    *,
    rng: np.random.Generator,
):
    """Every pre-``mcmc_fit`` step of the IDL loop for one curve.

    Collapses constant plateaus and interpolates back, shifts time to start
    at 0, maps the known flare times to samples and adds the dummy flare (see
    `add_dummy_flare_candidate`).

    Returns ``(t_rel, y, flare_indices, true_indices, t0)``.
    """
    t_abs = np.asarray(t_abs, dtype=float)
    y = interpolate_collapsed_plateaus(t_abs, np.asarray(y_raw, dtype=float), fill_edges=True)
    t0 = float(np.min(t_abs))
    t_rel = t_abs - t0
    # IDL matches flare times within half of the *first* sampling step.
    tolerance = 0.5 * float(t_rel[1] - t_rel[0]) if len(t_rel) > 1 else np.inf
    true_idx = find_peak_indices(t_abs, t_peaks_solar, tolerance=tolerance)
    flare_idx, true_idx = add_dummy_flare_candidate(len(t_rel), true_idx, rng=rng)
    return t_rel, y, flare_idx, true_idx, t0
