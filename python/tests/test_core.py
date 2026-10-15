import numpy as np

from tec_mcmc.models import (
    flare_profile,
    full_model,
    harmonic_background,
    split_model_components,
)
from tec_mcmc.preprocess import (
    add_dummy_flare_candidate,
    collapse_constant_plateaus,
    find_peak_indices,
)


def test_harmonic_background_matches_manual_formula():
    t = np.linspace(0, 10, 11)
    p = np.array(
        [
            2.0,
            3.0, 20.0, 0.1,
            2.0, 15.0, -0.2,
            1.0, 10.0, 0.3,
            0.5, 8.0, -0.4,
        ]
    )
    expected = np.full_like(t, 2.0)
    for j in range(4):
        i = 1 + 3 * j
        expected += p[i] * np.sin(2 * np.pi * t / p[i + 1] + p[i + 2])
    np.testing.assert_allclose(harmonic_background(t, p), expected)


def test_flare_peaks_at_amplitude():
    t = np.array([8.0, 10.0, 12.0])
    f = flare_profile(t, amplitude=5.0, peak_time=10.0, rise_time=2.0, decay_time=4.0)
    assert np.isclose(f[1], 5.0)
    assert f[0] < f[1]
    assert f[2] < f[1]


def test_full_model_splits_into_background_plus_flares():
    t = np.linspace(0, 20, 21)
    background = np.array(
        [0.0, 1.0, 100.0, 0.0, 0.5, 50.0, 0.0, 0.2, 30.0, 0.0, 0.1, 20.0, 0.0]
    )
    flare = np.array([2.0, 10.0, 1.5, 5.0])
    params = np.r_[background, flare]
    model = full_model(t, params, n_flares=1)
    trend, flares = split_model_components(t, params, n_flares=1)
    np.testing.assert_allclose(model, trend + flares)


def test_plateau_collapse_keeps_final_plateau():
    t = np.arange(8.0)
    y = np.array([1, 1, 2, 2, 2, 3, 3, 3], dtype=float)
    tc, yc = collapse_constant_plateaus(t, y)
    np.testing.assert_array_equal(yc, [1.0, 2.0, 3.0])
    # Midpoints are floor((start+end)/2): [0, 3, 6].
    np.testing.assert_array_equal(tc, [0.0, 3.0, 6.0])


def test_find_peak_indices_and_dummy_candidate():
    t = np.arange(0.0, 101.0, 1.0)
    found = find_peak_indices(t, [20.1], tolerance=0.5)
    np.testing.assert_array_equal(found, [20])

    rng = np.random.default_rng(123)
    candidates, true_idx = add_dummy_flare_candidate(len(t), found, rng=rng)
    assert candidates.size == 2
    assert true_idx.size == 1
    assert candidates[0] == 20
    # Dummy should lie in the opposite half for one true flare.
    assert candidates[1] >= len(t) // 2


def test_load_tec_csv_keeps_first_row(tmp_path):
    from tec_mcmc.tec import load_tec_csv

    p = tmp_path / "x.csv"
    p.write_text("60,1.0,NaN\n120,2.0,5.0\n180,3.0,6.0\n")
    curves = load_tec_csv(p)
    assert len(curves) == 2
    np.testing.assert_allclose(curves[0][0], [1.0, 2.0, 3.0])  # seconds -> minutes
    np.testing.assert_allclose(curves[0][1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(curves[1][0], [2.0, 3.0])  # NaN row dropped per curve


def test_vectorised_loglike_matches_loop():
    from tec_mcmc.inference import _log_likelihoods

    rng = np.random.default_rng(0)
    t = np.linspace(0, 100, 60)
    n_fl = 2
    rows = []
    for _ in range(5):
        bg = [rng.normal()]
        for _ in range(4):
            bg += [rng.uniform(0, 2), rng.uniform(50, 400), rng.uniform(-3, 3)]
        fl = []
        for _ in range(n_fl):
            fl += [rng.uniform(0, 1), rng.uniform(10, 90), rng.uniform(0.5, 5), rng.uniform(2, 50)]
        rows.append(np.r_[bg, fl, 0.3])
    S = np.array(rows)
    y = rng.normal(size=t.size)
    ref = []
    for r in S:
        mu = full_model(t, r[:-1], n_fl)
        ref.append(-0.5 * np.sum(np.log(2 * np.pi * r[-1] ** 2) + (y - mu) ** 2 / r[-1] ** 2))
    np.testing.assert_allclose(_log_likelihoods(t, y, S, n_fl, chunk=2), ref)


def test_multistart_recovers_noiseless_sinusoid():
    from tec_mcmc.config import PriorConfig
    from tec_mcmc.optimize import multistart_least_squares

    t = np.linspace(0, 120, 200)
    y = 10 + 2.0 * np.sin(2 * np.pi * t / 300 + 0.5)
    opts = multistart_least_squares(
        t, y, np.array([], dtype=int), 0.01, PriorConfig(), n_starts=15,
        rng=np.random.default_rng(0),
    )
    assert opts[0].rms < 1e-2


def test_noise_estimate_small_for_smooth_signal():
    from tec_mcmc.preprocess import estimate_noise_savgol

    t = np.linspace(0, 10, 300)
    noise, trend = estimate_noise_savgol(np.sin(t))
    assert noise < 0.05 and trend.shape == t.shape


def test_split_rhat_detects_disagreeing_chains():
    from tec_mcmc.inference import _split_rhat_ess

    rng = np.random.default_rng(1)
    good = rng.normal(size=(4, 1000))
    bad = good + np.array([0.0, 0.0, 0.0, 3.0])[:, None]
    r_good, ess_good = _split_rhat_ess(good)
    r_bad, _ = _split_rhat_ess(bad)
    assert r_good < 1.01 and ess_good > 2000
    assert r_bad > 1.3


def test_model_start_ordered_and_cartesian_roundtrip():
    from tec_mcmc.config import SamplerConfig
    from tec_mcmc.inference import model_start

    th = np.array([1.0, 2.0, 300.0, 0.4, 1.0, 50.0, -2.0, 0.5, 50.0, 3.0, 0.1, 120.0, 0.0])
    d = model_start(th, 0.1, 0, period_low=50.0, period_high=1440.0, config=SamplerConfig())
    assert np.all(np.diff(d["harmonic_period"]) > 0)
    amp = np.hypot(d["b_sin"], d["b_cos"])
    order = np.argsort(th[[2, 5, 8, 11]], kind="stable")
    np.testing.assert_allclose(amp, th[[1, 4, 7, 10]][order])


def test_initial_values_inside_prior_bounds():
    from tec_mcmc.config import PriorConfig
    from tec_mcmc.inference import initial_values
    from tec_mcmc.optimize import parameter_bounds

    t = np.linspace(0, 600, 200)  # long span: the first period guess hits the upper bound
    y = 5 + np.sin(t / 30)
    flares = np.array([50, 150])
    theta, sigma = initial_values(t, y, flares, 0.05, PriorConfig())
    lo, hi = parameter_bounds(t, flares, 0.05, PriorConfig())
    assert theta.shape == (13 + 4 * 2,)
    assert np.all(theta > lo) and np.all(theta < hi)
    assert 0 < sigma < np.ptp(y)


def test_parameter_index_constants():
    from tec_mcmc.models import AMP_IDX, PERIOD_IDX, PHASE_IDX

    np.testing.assert_array_equal(AMP_IDX, [1, 4, 7, 10])
    np.testing.assert_array_equal(PERIOD_IDX, [2, 5, 8, 11])
    np.testing.assert_array_equal(PHASE_IDX, [3, 6, 9, 12])


def test_save_json_atomic_roundtrip(tmp_path):
    import json

    from tec_mcmc.batch import save_json_atomic

    out = tmp_path / "res.json"
    save_json_atomic({"flare_0": {"map_params": [1.0, 2.0]}}, out)
    assert json.loads(out.read_text()) == {"flare_0": {"map_params": [1.0, 2.0]}}
    assert not out.with_suffix(".tmp").exists()


def test_prepare_curve_shifts_time_and_adds_dummy():
    from tec_mcmc.tec import prepare_curve

    t_abs = 300.0 + np.arange(100.0)
    y = np.sin(t_abs / 10)
    t, y2, flare_idx, true_idx, t0 = prepare_curve(t_abs, y, [320.2], rng=np.random.default_rng(0))
    assert t0 == 300.0 and t[0] == 0.0 and y2.shape == y.shape
    np.testing.assert_array_equal(true_idx, [20])
    assert flare_idx.size == 2 and flare_idx[0] == 20
