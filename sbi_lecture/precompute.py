"""Reproducible asset generation and fast loading for the Week 10 SBI lecture.

Provides two modes:
- `full`: Generates the complete 50,000-simulation Plinko ABC bank, 24,000-simulation
  neural SBI training suite (45 epochs), Gamma-to-GMM optimization checkpoints,
  skewed regression/MDN suite, multi-observation exact/neural posteriors, and
  200-experiment independent coverage/SBC diagnostics, saving all artifacts under `assets/sbi/`.
- `fast`: Quick regeneration mode (~5-10 seconds) with smaller simulation counts for rapid testing.

During live lecture execution, all widgets and slides load directly from `assets/sbi/`
with zero internet access and sub-second startup.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .abc import generate_plinko_abc_bank
from .density_estimation import (
    load_or_build_gmm_gamma_suite,
    train_or_load_regression_mdn_suite,
)
from .neural_sbi import (
    evaluate_neural_posterior_grid,
    load_or_train_neural_sbi_suite,
    train_neural_sbi_suite,
)
from .plinko import (
    DEFAULT_CONFIG,
    PlinkoConfig,
    exact_posterior_grid,
    generate_default_observation,
    simulate_trajectories,
)
from .validation import compute_or_load_coverage_suite


def default_assets_dir() -> Path:
    """Locate `<repo_root>/assets/sbi` regardless of current working directory."""
    here = Path(__file__).resolve().parent.parent
    return here / "assets" / "sbi"


def get_benchmark_observations(config: PlinkoConfig = DEFAULT_CONFIG) -> list[dict[str, object]]:
    """Return 4 deterministic Plinko observations for amortized inference demonstrations."""
    primary = generate_default_observation(config=config)
    specs = [
        ("Mystery Machine (Observed)", config.alpha_true, config.kappa_true, config.obs_seed),
        ("Experiment B (Left Bias)", -0.75, 0.90, 2027),
        ("Experiment C (Weak Restoring)", 0.35, 0.20, 2028),
        ("Experiment D (Strong Centering)", -0.30, 2.35, 2029),
    ]
    out = []
    for idx, (label, a_val, k_val, seed) in enumerate(specs):
        if idx == 0:
            out.append(
                {
                    "label": label,
                    "x_obs": primary["x_obs"],
                    "k_paths": primary["k_paths"],
                    "s_paths": primary["s_paths"],
                    "alpha_true": a_val,
                    "kappa_true": k_val,
                    "seed": seed,
                }
            )
        else:
            sim = simulate_trajectories(
                alpha=a_val,
                kappa=k_val,
                n_balls=config.n_balls,
                n_rows=config.n_rows,
                seed=seed,
            )
            out.append(
                {
                    "label": label,
                    "x_obs": sim["histogram"],
                    "k_paths": sim["k_paths"],
                    "s_paths": sim["s_paths"],
                    "alpha_true": a_val,
                    "kappa_true": k_val,
                    "seed": seed,
                }
            )
    return out


_BUNDLE_CACHE: dict[str, dict[str, object]] = {}


def load_lecture_bundle(
    assets_dir: str | Path | None = None,
    mode: str = "auto",
    force_recompute: bool = False,
    config: PlinkoConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Load (or compute and cache) all precomputed simulation banks, checkpoints, and posteriors."""
    if assets_dir in ("full", "fast", "auto"):
        mode = str(assets_dir)
        assets_dir = None
    adir = Path(assets_dir) if assets_dir is not None else default_assets_dir()
    cache_key = str(adir.resolve())
    if not force_recompute and cache_key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[cache_key]

    adir.mkdir(parents=True, exist_ok=True)
    abc_path = adir / "plinko_abc_bank.npz"
    gmm_path = adir / "gmm_gamma_checkpoints.npz"
    reg_path = adir / "regression_mdn_demo.npz"
    nsbi_path = adir / "neural_sbi_models.pt"
    cov_path = adir / "validation_coverage.npz"

    if force_recompute:
        for p in (abc_path, gmm_path, reg_path, nsbi_path, cov_path):
            if p.exists():
                p.unlink()

    is_fast = mode == "fast"
    n_abc = 8000 if is_fast else 50000
    n_train = 8000 if is_fast else 40000
    epochs = 30 if is_fast else 80
    n_val = 50 if is_fast else 200

    abc_bank = generate_plinko_abc_bank(
        n_simulations=n_abc,
        config=config,
        seed=1010,
        cache_path=abc_path,
    )
    gmm_suite = load_or_build_gmm_gamma_suite(cache_path=gmm_path)
    reg_suite = train_or_load_regression_mdn_suite(cache_path=reg_path)

    if nsbi_path.exists() and not force_recompute:
        neural_suite = load_or_train_neural_sbi_suite(nsbi_path, config=config)
    else:
        neural_suite = train_neural_sbi_suite(
            n_train=n_train,
            epochs=epochs,
            config=config,
            checkpoint_path=nsbi_path,
        )

    coverage_suite = compute_or_load_coverage_suite(
        neural_suite=neural_suite,
        abc_bank=abc_bank,
        n_validation=n_val,
        validation_seed=90909,
        config=config,
        cache_path=cov_path,
    )

    observations = get_benchmark_observations(config=config)
    obs_posteriors = []
    for obs in observations:
        x_i = obs["x_obs"]
        obs_posteriors.append(
            {
                "exact": exact_posterior_grid(x_i, n_alpha=121, n_kappa=121, config=config),
                "NPE": evaluate_neural_posterior_grid(
                    neural_suite["npe"], "NPE", x_i, n_alpha=121, n_kappa=121, config=config
                ),
                "NLE": evaluate_neural_posterior_grid(
                    neural_suite["nle"], "NLE", x_i, n_alpha=121, n_kappa=121, config=config
                ),
                "NRE": evaluate_neural_posterior_grid(
                    neural_suite["nre"], "NRE", x_i, n_alpha=121, n_kappa=121, config=config
                ),
            }
        )

    bundle = {
        "config": config,
        "assets_dir": adir,
        "observations": observations,
        "primary_obs": observations[0],
        "obs_posteriors": obs_posteriors,
        "abc_bank": abc_bank,
        "gmm_suite": gmm_suite,
        "reg_suite": reg_suite,
        "neural_suite": neural_suite,
        "coverage_suite": coverage_suite,
    }
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle

