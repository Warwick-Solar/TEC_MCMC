from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def collapse_constant_plateaus(
    t: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace each consecutive constant plateau by its midpoint sample."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("t and y must be one-dimensional arrays of equal length.")
    if t.size == 0:
        return t.copy(), y.copy()

    starts = [0]
    for q in range(1, y.size):
        if y[q] != y[q - 1]:
            starts.append(q)
    starts = np.asarray(starts, dtype=int)
    ends = np.r_[starts[1:] - 1, y.size - 1]
    mids = (starts + ends) // 2
    return t[mids], y[mids]


def interpolate_collapsed_plateaus(
    t: np.ndarray,
    y: np.ndarray,
    *,
    fill_edges: bool = True,
) -> np.ndarray:
    """Collapse plateaus and linearly interpolate back to the original times."""
    t = np.asarray(t, dtype=float)
    tc, yc = collapse_constant_plateaus(t, y)
    if tc.size == 1:
        return np.full_like(t, yc[0], dtype=float)
    fill = (yc[0], yc[-1]) if fill_edges else np.nan
    f = interp1d(
        tc,
        yc,
        kind="linear",
        bounds_error=False,
        fill_value=fill,
        assume_sorted=True,
    )
    return np.asarray(f(t), dtype=float)


def estimate_noise_savgol(y: np.ndarray, degree: int = 4) -> tuple[float, np.ndarray]:
    """Savitzky-Golay noise estimate following the IDL routine.

    Returns ``(std(y - smooth), smooth)``.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        trend = np.full_like(y, np.nanmean(y))
        return max(float(np.nanstd(y - trend, ddof=min(1, max(y.size - 1, 0)))), np.finfo(float).eps), trend

    half = y.size // 5
    if half < degree:
        half = 2
    window = 2 * half + 1
    if window > y.size:
        window = y.size if y.size % 2 else y.size - 1
    polyorder = min(degree, window - 1)
    trend = savgol_filter(y, window_length=window, polyorder=polyorder, mode="nearest")
    noise = float(np.nanstd(y - trend, ddof=1))
    return max(noise, np.finfo(float).eps), trend


def find_peak_indices(
    t_absolute: np.ndarray,
    known_peak_times: np.ndarray | list[float],
    *,
    tolerance: float | None = None,
) -> np.ndarray:
    """Map externally supplied flare times to nearest observed samples."""
    t = np.asarray(t_absolute, dtype=float)
    peaks = np.asarray(known_peak_times, dtype=float)
    if t.size == 0 or peaks.size == 0:
        return np.empty(0, dtype=int)

    if tolerance is None:
        if t.size < 2:
            tolerance = np.inf
        else:
            tolerance = 0.5 * float(np.nanmedian(np.diff(t)))

    indices: list[int] = []
    for peak in peaks:
        j = int(np.nanargmin(np.abs(t - peak)))
        if abs(t[j] - peak) <= tolerance:
            indices.append(j)
    return np.asarray(indices, dtype=int)


def add_dummy_flare_candidate(
    n_points: int,
    true_indices: np.ndarray | list[int],
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Adds a dummy flare for noise testing

    Returns `(candidate_indices, true_indices)`. A dummy is added only when the
    number of true candidates is 0, 1, or 2. With >=3 true candidates no dummy
    is added.
    """
    true_idx = np.asarray(true_indices, dtype=int)
    if n_points < 2:
        return true_idx.copy(), true_idx

    margin = n_points // 10
    if true_idx.size == 0:
        dummy = int(rng.integers(0, n_points))
        return np.asarray([dummy], dtype=int), true_idx

    if true_idx.size == 1:
        midpoint = n_points // 2
        if true_idx[0] < midpoint:
            lo = max(margin, midpoint)
            hi = max(lo + 1, n_points - margin)
        else:
            lo = margin
            hi = min(midpoint, n_points - margin)
            if hi <= lo:
                lo, hi = 0, max(1, midpoint)
        candidates = np.arange(lo, hi, dtype=int)
        dummy = int(rng.choice(candidates))
        return np.r_[true_idx, dummy], true_idx

    if true_idx.size == 2:
        valid = np.arange(margin, max(margin + 1, n_points - margin), dtype=int)
        distances = np.min(np.abs(valid[:, None] - true_idx[None, :]), axis=1)
        best = valid[distances == distances.max()]
        dummy = int(rng.choice(best))
        return np.r_[true_idx, dummy], true_idx

    return true_idx.copy(), true_idx
