"""Posterior comparison, posterior predictive checks, and empirical coverage / SBC validation.

Ensures strictly independent validation simulation datasets (disjoint seeds from training
and ABC banks) to evaluate frequentist coverage of Bayesian credible regions and
Simulation-Based Calibration (Talts et al., 2018).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.stats import binom

from .abc import estimate_2d_kde_grid, run_plinko_abc
from .neural_sbi import evaluate_neural_posterior_grid
from .plinko import (
    DEFAULT_CONFIG,
    PlinkoConfig,
    exact_posterior_grid,
    plinko_summary_statistics,
    sample_from_grid_posterior,
    sample_prior,
    simulate_plinko_histograms,
)


def compare_posterior_grids(
    reference_grid: dict,
    approx_grid: dict,
) -> dict[str, float]:
    """Compute quantitative discrepancies between an approximate 2D posterior and the exact DP benchmark."""
    p_ref = np.asarray(reference_grid["posterior"], dtype=float)
    q_app = np.asarray(approx_grid["posterior"], dtype=float)
    da = float(reference_grid["d_alpha"])
    dk = float(reference_grid["d_kappa"])

    tv_2d = 0.5 * float(np.sum(np.abs(p_ref - q_app)) * da * dk)
    mask = p_ref > 1e-12
    safe_q = np.maximum(q_app[mask], 1e-12)
    kl_forward = float(np.sum(p_ref[mask] * np.log(p_ref[mask] / safe_q)) * da * dk)

    alpha_grid = np.asarray(reference_grid["alpha_grid"], dtype=float)
    kappa_grid = np.asarray(reference_grid["kappa_grid"], dtype=float)
    marg_a = np.asarray(approx_grid["marginal_alpha"], dtype=float)
    marg_k = np.asarray(approx_grid["marginal_kappa"], dtype=float)
    mean_a = float(np.sum(alpha_grid * marg_a) * da)
    mean_k = float(np.sum(kappa_grid * marg_k) * dk)
    std_a = float(np.sqrt(max(np.sum(((alpha_grid - mean_a) ** 2) * marg_a) * da, 0.0)))
    std_k = float(np.sqrt(max(np.sum(((kappa_grid - mean_k) ** 2) * marg_k) * dk, 0.0)))

    cdf_a = np.cumsum(marg_a) * da
    cdf_k = np.cumsum(marg_k) * dk
    ci68_a = (
        float(np.interp(0.16, cdf_a, alpha_grid)),
        float(np.interp(0.84, cdf_a, alpha_grid)),
    )
    ci68_k = (
        float(np.interp(0.16, cdf_k, kappa_grid)),
        float(np.interp(0.84, cdf_k, kappa_grid)),
    )

    return {
        "tv_distance": tv_2d,
        "kl_divergence": max(kl_forward, 0.0),
        "mean_alpha": mean_a,
        "std_alpha": std_a,
        "mean_kappa": mean_k,
        "std_kappa": std_k,
        "ci68_alpha_low": ci68_a[0],
        "ci68_alpha_high": ci68_a[1],
        "ci68_kappa_low": ci68_k[0],
        "ci68_kappa_high": ci68_k[1],
    }


def compute_ppc_test_statistic(
    histograms: np.ndarray,
    stat_kind: str = "mean",
    ref_expected: np.ndarray | None = None,
) -> np.ndarray | float:
    """Evaluate a scalar discrepancy / test statistic T(x) on 1D or 2D Plinko histograms."""
    h = np.asarray(histograms, dtype=float)
    single = h.ndim == 1
    stats = plinko_summary_statistics(h)

    if stat_kind == "mean":
        val = stats["mean"]
    elif stat_kind == "std":
        val = stats["std"]
    elif stat_kind == "max_count":
        val = float(np.max(h)) if single else np.max(h, axis=-1)
    elif stat_kind == "chi2":
        if ref_expected is None:
            raise ValueError("ref_expected is required for stat_kind='chi2'")
        exp_c = np.maximum(np.asarray(ref_expected, dtype=float), 0.5)
        if single:
            val = float(np.sum(((h - exp_c) ** 2) / exp_c))
        else:
            val = np.sum(((h - exp_c[None, :]) ** 2) / exp_c[None, :], axis=-1)
    else:
        raise ValueError(f"Unknown PPC test statistic: {stat_kind}")
    return val


def run_posterior_predictive_check(
    posterior_samples: np.ndarray,
    x_obs: np.ndarray,
    n_predictive: int = 300,
    stat_kind: str = "mean",
    config: PlinkoConfig = DEFAULT_CONFIG,
    seed: int = 202605,
) -> dict[str, object]:
    """Simulate replicated Plinko histograms from posterior parameter draws and compute PPC summaries."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(posterior_samples, dtype=float)
    if pts.shape[0] == 0:
        raise ValueError("posterior_samples cannot be empty")
    idx = rng.choice(pts.shape[0], size=n_predictive, replace=pts.shape[0] < n_predictive)
    sel = pts[idx]

    rep_hists = simulate_plinko_histograms(
        sel[:, 0],
        sel[:, 1],
        n_balls=config.n_balls,
        n_rows=config.n_rows,
        seed=rng,
    )
    mean_hist = np.mean(rep_hists, axis=0)
    q05, q25, q50, q75, q95 = np.quantile(rep_hists, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)

    t_rep = np.asarray(
        compute_ppc_test_statistic(rep_hists, stat_kind=stat_kind, ref_expected=mean_hist),
        dtype=float,
    )
    t_obs = float(compute_ppc_test_statistic(x_obs, stat_kind=stat_kind, ref_expected=mean_hist))
    # Two-sided posterior predictive tail probability / exceedance fraction
    p_exceed = float(np.mean(t_rep >= t_obs))

    return {
        "rep_histograms": rep_hists,
        "sampled_thetas": sel,
        "mean_histogram": mean_hist,
        "q05": q05,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q95": q95,
        "t_rep": t_rep,
        "t_obs": t_obs,
        "p_value": p_exceed,
    }


def _hpd_credible_mass_rank_for_truth(
    posterior_2d: np.ndarray,
    alpha_grid: np.ndarray,
    kappa_grid: np.ndarray,
    d_alpha: float,
    d_kappa: float,
    alpha_true: float,
    kappa_true: float,
) -> tuple[float, float, float]:
    """Return (2D HPD mass rank c*, 1D alpha PIT rank, 1D kappa PIT rank) in [0, 1]."""
    post = np.asarray(posterior_2d, dtype=float)
    post = post / max(float(np.sum(post) * d_alpha * d_kappa), 1e-300)

    ia = int(np.clip(np.argmin(np.abs(alpha_grid - alpha_true)), 0, alpha_grid.size - 1))
    ik = int(np.clip(np.argmin(np.abs(kappa_grid - kappa_true)), 0, kappa_grid.size - 1))
    dens_true = float(post[ia, ik])

    # Smallest HPD region containing theta_true has probability mass sum_{p >= p(theta_true)} p dA
    hpd_rank = float(np.sum(post[post >= dens_true]) * d_alpha * d_kappa)
    hpd_rank = float(np.clip(hpd_rank, 0.0, 1.0))

    marg_a = np.sum(post, axis=1) * d_kappa
    marg_k = np.sum(post, axis=0) * d_alpha
    cdf_a = np.clip(np.cumsum(marg_a) * d_alpha, 0.0, 1.0)
    cdf_k = np.clip(np.cumsum(marg_k) * d_alpha * (d_kappa / d_alpha), 0.0, 1.0)
    pit_a = float(np.interp(alpha_true, alpha_grid, cdf_a))
    pit_k = float(np.interp(kappa_true, kappa_grid, cdf_k))
    return hpd_rank, pit_a, pit_k


def compute_or_load_coverage_suite(
    neural_suite: dict[str, object],
    abc_bank: dict[str, np.ndarray],
    n_validation: int = 200,
    n_grid: int = 81,
    validation_seed: int = 90909,
    config: PlinkoConfig = DEFAULT_CONFIG,
    cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Compute or load empirical 2D HPD coverage and 1D SBC ranks on independent simulations."""
    methods = ("Exact", "NPE", "NLE", "NRE", "ABC")
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            raw = np.load(p, allow_pickle=False)
            if int(raw["n_validation"]) >= n_validation:
                sl = slice(0, n_validation)
                return {
                    "n_validation": n_validation,
                    "validation_seed": int(raw["validation_seed"]),
                    "val_thetas": raw["val_thetas"][sl],
                    "val_histograms": raw["val_histograms"][sl],
                    "methods": methods,
                    "hpd_ranks": {m: raw[f"hpd_{m}"][sl] for m in methods},
                    "pit_alpha": {m: raw[f"pita_{m}"][sl] for m in methods},
                    "pit_kappa": {m: raw[f"pitk_{m}"][sl] for m in methods},
                }

    # Strictly fresh RNG disjoint from training (202601), ABC bank (1010), and x_obs (2026)
    rng = np.random.default_rng(validation_seed)
    val_thetas = sample_prior(n_validation, config=config, seed=rng)
    val_hists = simulate_plinko_histograms(
        val_thetas[:, 0],
        val_thetas[:, 1],
        n_balls=config.n_balls,
        n_rows=config.n_rows,
        seed=rng,
    )

    hpd_ranks = {m: np.zeros(n_validation, dtype=float) for m in methods}
    pit_alpha = {m: np.zeros(n_validation, dtype=float) for m in methods}
    pit_kappa = {m: np.zeros(n_validation, dtype=float) for m in methods}

    for j in range(n_validation):
        a_true, k_true = float(val_thetas[j, 0]), float(val_thetas[j, 1])
        xj = val_hists[j]

        # 1. Exact DP posterior
        g_ex = exact_posterior_grid(xj, n_alpha=n_grid, n_kappa=n_grid, config=config)
        hpd_ranks["Exact"][j], pit_alpha["Exact"][j], pit_kappa["Exact"][j] = (
            _hpd_credible_mass_rank_for_truth(
                g_ex["posterior"],
                g_ex["alpha_grid"],
                g_ex["kappa_grid"],
                g_ex["d_alpha"],
                g_ex["d_kappa"],
                a_true,
                k_true,
            )
        )

        # 2-4. Neural SBI posteriors
        for m_name, key in (("NPE", "npe"), ("NLE", "nle"), ("NRE", "nre")):
            g_m = evaluate_neural_posterior_grid(
                neural_suite[key],
                method=m_name,
                x_obs=xj,
                n_alpha=n_grid,
                n_kappa=n_grid,
                config=config,
            )
            hpd_ranks[m_name][j], pit_alpha[m_name][j], pit_kappa[m_name][j] = (
                _hpd_credible_mass_rank_for_truth(
                    g_m["posterior"],
                    g_m["alpha_grid"],
                    g_m["kappa_grid"],
                    g_m["d_alpha"],
                    g_m["d_kappa"],
                    a_true,
                    k_true,
                )
            )

        # 5. Rejection ABC (fixed reasonable Wasserstein tolerance epsilon = 0.35 bins)
        abc_res = run_plinko_abc(
            abc_bank,
            xj,
            epsilon=0.35,
            distance_kind="wasserstein",
            summary_kind="full_hist",
            compute_kde=False,
            config=config,
        )
        g_abc = estimate_2d_kde_grid(
            abc_res["accepted_thetas"],
            n_alpha=n_grid,
            n_kappa=n_grid,
            config=config,
        )
        hpd_ranks["ABC"][j], pit_alpha["ABC"][j], pit_kappa["ABC"][j] = (
            _hpd_credible_mass_rank_for_truth(
                g_abc["posterior"],
                g_abc["alpha_grid"],
                g_abc["kappa_grid"],
                g_abc["d_alpha"],
                g_abc["d_kappa"],
                a_true,
                k_true,
            )
        )

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        save_dict: dict[str, np.ndarray] = {
            "n_validation": np.array(n_validation, dtype=int),
            "validation_seed": np.array(validation_seed, dtype=int),
            "val_thetas": val_thetas,
            "val_histograms": val_hists,
        }
        for m in methods:
            save_dict[f"hpd_{m}"] = hpd_ranks[m]
            save_dict[f"pita_{m}"] = pit_alpha[m]
            save_dict[f"pitk_{m}"] = pit_kappa[m]
        np.savez_compressed(p, **save_dict)

    return {
        "n_validation": n_validation,
        "validation_seed": validation_seed,
        "val_thetas": val_thetas,
        "val_histograms": val_hists,
        "methods": methods,
        "hpd_ranks": hpd_ranks,
        "pit_alpha": pit_alpha,
        "pit_kappa": pit_kappa,
    }


def summarize_empirical_coverage(
    hpd_ranks: np.ndarray,
    nominal_levels: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute empirical coverage curve and 95% binomial sampling interval around diagonal."""
    ranks = np.asarray(hpd_ranks, dtype=float)
    n = int(ranks.size)
    if nominal_levels is None:
        nominal_levels = np.linspace(0.02, 0.98, 49)
    emp = np.mean(ranks[:, None] <= nominal_levels[None, :], axis=0)
    band_low = binom.ppf(0.025, n=n, p=nominal_levels) / max(n, 1)
    band_high = binom.ppf(0.975, n=n, p=nominal_levels) / max(n, 1)
    return {
        "nominal": nominal_levels,
        "empirical": emp,
        "band_low": band_low,
        "band_high": band_high,
        "n_experiments": n,
    }

