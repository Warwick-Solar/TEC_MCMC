"""Small end-to-end PyMC smoke test.

This verifies that the complete pipeline executes. The draw counts are too small
for publication-quality inference; increase them after the smoke test works.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tec_mcmc import SamplerConfig, fit_curve
from tec_mcmc.models import full_model, split_model_components


def main() -> None:
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
    true_model = full_model(t, true_params, n_flares=1)
    y = true_model + rng.normal(0.0, 0.35, size=t.size)

    flare_index = int(np.argmin(np.abs(t - 120.0)))

    config = SamplerConfig(
        draws=300,
        tune=300,
        chains=1,
        cores=1,
        sampler="demetropolisz",
        ppd_draws=200,
        random_seed=11,
        progressbar=True,
    )

    result = fit_curve(t, y, [flare_index], config=config)
    true_trend, _ = split_model_components(t, true_params, n_flares=1)
    rmse = np.sqrt(np.mean((result.map_trend - true_trend) ** 2))

    print(f"Posterior rows: {len(result.posterior_samples)}")
    print(f"MAP sigma: {result.sigma_map:.3f}")
    print(f"Trend RMSE in this tiny smoke run: {rmse:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, y, ".", ms=3, label="synthetic data")
    ax.plot(t, result.map_model, label="sampled-MAP full model")
    ax.plot(t, result.map_trend, label="background trend")
    ax.fill_between(t, result.ppd_lower, result.ppd_upper, alpha=0.2, label="95% PPD")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Signal")
    ax.legend()
    fig.tight_layout()

    out = Path(__file__).with_name("synthetic_smoke_test.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
