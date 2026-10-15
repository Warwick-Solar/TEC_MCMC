"""Small end-to-end PyMC run on a synthetic curve with a known answer.

Slow (~1 min), so it is skipped by default.  Run it with::

    pytest -m mcmc

The draw counts are far too small for publication-quality inference; the test
only checks that the whole pipeline runs and recovers the trend roughly.
"""

import numpy as np
import pytest

from tec_mcmc import SamplerConfig, fit_curve
from tec_mcmc.models import full_model, split_model_components


@pytest.mark.mcmc
def test_synthetic_curve_end_to_end():
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 240.0, 121)
    background = np.array(
        [
            10.0,
            2.0, 900.0, 0.2,
            1.2, 500.0, -0.4,
            0.7, 260.0, 0.8,
            0.3, 120.0, -0.5,
        ]
    )
    flare = np.array([3.0, 120.0, 2.0, 20.0])
    true_params = np.r_[background, flare]
    noise = 0.35
    y = full_model(t, true_params, n_flares=1) + rng.normal(0.0, noise, size=t.size)
    flare_index = int(np.argmin(np.abs(t - 120.0)))

    config = SamplerConfig(
        draws=300, tune=300, chains=2, cores=1, sampler="demetropolisz",
        ppd_draws=200, random_seed=11, n_starts=10, progressbar=False,
    )
    result = fit_curve(t, y, [flare_index], config=config)

    n_model = 13 + 4
    assert result.posterior_samples.shape == (2 * 300, n_model + 1)
    assert result.map_params.shape == (n_model + 1,)
    assert np.all(np.isfinite(result.map_model))
    assert np.all(result.ppd_lower <= result.ppd_upper)
    assert set(result.diagnostics) >= {"rhat_curves_max", "converged"}

    true_trend, true_flares = split_model_components(t, true_params, n_flares=1)
    model_rmse = np.sqrt(np.mean((result.map_model - (true_trend + true_flares)) ** 2))
    assert model_rmse < noise  # best fit is closer to the truth than the noise
    assert 0.5 * noise < result.sigma_map < 1.5 * noise
