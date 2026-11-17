"""Rejection Approximate Bayesian Computation (ABC) utilities and lecture benchmarks.

Includes:
1. Original 1D chi-squared / Normal joint-slice ABC model (Slides 11-13, W04).
2. Corrected bivariate correlated-Gaussian ABC model (Slide 14, W05).
3. Mystery Plinko rejection ABC with cached simulation banks, summary statistics,
   and multiple distance functions (Slides 15-16, W06 & W07).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.stats import chi2, gaussian_kde, multivariate_normal, norm

from .plinko import (
    DEFAULT_CONFIG,
    PlinkoConfig,
    compute_hpd_levels,
    plinko_summary_statistics,
    sample_prior,
    simulate_plinko_histograms,
)


# -----------------------------------------------------------------------------
# 1. Original 1D Chi-Squared(5) + Normal(sqrt(theta), 1) ABC Benchmark (W04)
# -----------------------------------------------------------------------------

def simulate_chi2_normal_joint(
    n_samples: int = 2000,
    df: float = 5.0,
    noise_std: float = 1.0,
    seed: int | np.random.Generator | None = 42,
) -> dict[str, np.ndarray]:
    """Draw joint samples (theta_i, x_i) from theta ~ chi2(df), x | theta ~ N(sqrt(theta), noise_std^2)."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    theta_samples = chi2.rvs(df=df, size=n_samples, random_state=rng)
    x_samples = norm.rvs(loc=np.sqrt(theta_samples), scale=noise_std, random_state=rng)
    return {"theta": theta_samples, "x": x_samples}


def chi2_normal_joint_grid(
    theta_grid: np.ndarray | None = None,
    x_grid: np.ndarray | None = None,
    df: float = 5.0,
    noise_std: float = 1.0,
) -> dict[str, np.ndarray]:
    """Evaluate exact joint density p(theta, x) on a 2D grid for contour plotting."""
    if theta_grid is None:
        theta_grid = np.linspace(0.05, 17.0, 140)
    if x_grid is None:
        x_grid = np.linspace(-2.0, 7.0, 120)
    tt, xx = np.meshgrid(theta_grid, x_grid, indexing="xy")
    joint = chi2.pdf(tt, df=df) * norm.pdf(xx, loc=np.sqrt(np.maximum(tt, 0.0)), scale=noise_std)
    return {"theta_grid": theta_grid, "x_grid": x_grid, "joint_pdf": joint}


def chi2_normal_exact_posterior(
    x_obs: float,
    theta_grid: np.ndarray | None = None,
    df: float = 5.0,
    noise_std: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Compute exact normalized 1D posterior p(theta | x_obs) numerically on a dense grid."""
    if theta_grid is None:
        theta_grid = np.linspace(1e-4, 35.0, 1500)
    unnorm = chi2.pdf(theta_grid, df=df) * norm.pdf(
        x_obs, loc=np.sqrt(np.maximum(theta_grid, 0.0)), scale=noise_std
    )
    z = float(np.trapezoid(unnorm, theta_grid))
    pdf = unnorm / max(z, 1e-300)
    mean = float(np.trapezoid(theta_grid * pdf, theta_grid))
    var = float(np.trapezoid(((theta_grid - mean) ** 2) * pdf, theta_grid))
    return {
        "theta_grid": theta_grid,
        "posterior_pdf": pdf,
        "evidence": z,
        "mean": mean,
        "std": float(np.sqrt(max(var, 0.0))),
    }


def chi2_normal_abc_analytic_posterior(
    x_obs: float,
    epsilon: float,
    theta_grid: np.ndarray | None = None,
    df: float = 5.0,
    noise_std: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Exact finite-tolerance ABC posterior p_epsilon(theta | x_obs) for |X - x_obs| < epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive")
    if theta_grid is None:
        theta_grid = np.linspace(1e-4, 35.0, 1500)
    loc = np.sqrt(np.maximum(theta_grid, 0.0))
    acc_prob = norm.cdf((x_obs + epsilon - loc) / noise_std) - norm.cdf(
        (x_obs - epsilon - loc) / noise_std
    )
    unnorm = chi2.pdf(theta_grid, df=df) * np.maximum(acc_prob, 0.0)
    z = float(np.trapezoid(unnorm, theta_grid))
    pdf = unnorm / max(z, 1e-300)
    return {
        "theta_grid": theta_grid,
        "abc_posterior_pdf": pdf,
        "marginal_acceptance": z,
    }


def run_1d_rejection_abc(
    theta_samples: np.ndarray,
    x_samples: np.ndarray,
    x_obs: float,
    epsilon: float,
) -> dict[str, np.ndarray | float | int]:
    """Filter a fixed joint sample pool (theta_i, x_i) using |x_i - x_obs| < epsilon."""
    if epsilon <= 0:
        raise ValueError("Tolerance epsilon must be > 0")
    distances = np.abs(np.asarray(x_samples, dtype=float) - float(x_obs))
    accepted_mask = distances < float(epsilon)
    accepted_theta = np.asarray(theta_samples, dtype=float)[accepted_mask]
    n_total = int(distances.size)
    n_acc = int(np.sum(accepted_mask))
    return {
        "distances": distances,
        "accepted_mask": accepted_mask,
        "accepted_theta": accepted_theta,
        "n_accepted": n_acc,
        "n_total": n_total,
        "acceptance_rate": n_acc / max(n_total, 1),
    }


# -----------------------------------------------------------------------------
# 2. Corrected Correlated Bivariate Gaussian ABC Benchmark (W05)
# -----------------------------------------------------------------------------

def correlated_gaussian_exact_conditional(
    x_obs: float,
    rho: float = 0.85,
    mu_theta: float = 0.0,
    mu_x: float = 0.0,
    sigma_theta: float = 1.0,
    sigma_x: float = 1.5,
    theta_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Exact conditional distribution p(theta | x_obs) for bivariate normal (theta, x).

    Corrects the missing factor of rho in the original notebook:
        E[theta | x] = mu_theta + rho * (sigma_theta / sigma_x) * (x - mu_x)
        Var(theta | x) = sigma_theta^2 * (1 - rho^2)
    """
    if not (-0.999 <= rho <= 0.999):
        raise ValueError(f"Correlation rho must lie in [-0.999, 0.999], got {rho}")
    cond_mean = float(mu_theta + rho * (sigma_theta / sigma_x) * (x_obs - mu_x))
    cond_var = float((sigma_theta ** 2) * (1.0 - rho ** 2))
    cond_std = float(np.sqrt(max(cond_var, 1e-12)))

    if theta_grid is None:
        theta_grid = np.linspace(mu_theta - 4.0 * sigma_theta, mu_theta + 4.0 * sigma_theta, 400)
    pdf = norm.pdf(theta_grid, loc=cond_mean, scale=cond_std)
    return {
        "theta_grid": theta_grid,
        "posterior_pdf": pdf,
        "cond_mean": cond_mean,
        "cond_var": cond_var,
        "cond_std": cond_std,
    }


def simulate_correlated_gaussian_joint(
    n_samples: int = 10000,
    rho: float = 0.85,
    mu_theta: float = 0.0,
    mu_x: float = 0.0,
    sigma_theta: float = 1.0,
    sigma_x: float = 1.5,
    seed: int | np.random.Generator | None = 42,
) -> dict[str, np.ndarray]:
    """Sample (theta, x) from correlated bivariate normal using Cholesky parameterization.

    Using fixed standard normal draws transformed via (rho, sigma) makes changing rho
    smooth and deterministic in interactive explorations.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    z1 = rng.standard_normal(n_samples)
    z2 = rng.standard_normal(n_samples)
    x = mu_x + sigma_x * z1
    theta = mu_theta + sigma_theta * (rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z2)

    cov = np.array(
        [
            [sigma_theta ** 2, rho * sigma_theta * sigma_x],
            [rho * sigma_theta * sigma_x, sigma_x ** 2],
        ]
    )
    return {"theta": theta, "x": x, "mean": np.array([mu_theta, mu_x]), "cov": cov}


# -----------------------------------------------------------------------------
# 3. Mystery Plinko ABC & Summary-Statistic Distances (W06, W07)
# -----------------------------------------------------------------------------

DISTANCE_LABELS = {
    "wasserstein": "1D Wasserstein (Earth Mover, in bins)",
    "tv": "Total Variation (1/2 L1 on frequencies)",
    "l2": "Euclidean L2 on bin frequencies",
    "chi2_sym": "Symmetric Chi-Square discrepancy",
}

SUMMARY_LABELS = {
    "mean_only": "Mean landing bin only (1 summary — insufficient)",
    "mean_var": "Mean + variance of landing bins (2 summaries)",
    "full_hist": "Full 13-bin histogram frequencies (complete data)",
}


def generate_plinko_abc_bank(
    n_simulations: int = 50000,
    config: PlinkoConfig = DEFAULT_CONFIG,
    seed: int = 1010,
    cache_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Generate or load a cached prior-predictive simulation bank for Plinko ABC."""
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            data = np.load(p)
            if int(data["thetas"].shape[0]) >= n_simulations:
                sl = slice(0, n_simulations)
                return {
                    "thetas": data["thetas"][sl],
                    "histograms": data["histograms"][sl],
                    "freqs": data["freqs"][sl],
                    "means": data["means"][sl],
                    "vars": data["vars"][sl],
                    "stds": data["stds"][sl],
                    "mean_scale": float(data["mean_scale"]),
                    "var_scale": float(data["var_scale"]),
                }

    rng = np.random.default_rng(seed)
    thetas = sample_prior(n_simulations, config=config, seed=rng)
    hists = simulate_plinko_histograms(
        thetas[:, 0],
        thetas[:, 1],
        n_balls=config.n_balls,
        n_rows=config.n_rows,
        seed=rng,
    )
    stats = plinko_summary_statistics(hists)
    mean_scale = float(np.std(stats["mean"]))
    var_scale = float(np.std(stats["var"]))

    bank = {
        "thetas": thetas,
        "histograms": hists,
        "freqs": stats["freqs"],
        "means": stats["mean"],
        "vars": stats["var"],
        "stds": stats["std"],
        "mean_scale": mean_scale,
        "var_scale": var_scale,
    }
    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **bank)
    return bank


def compute_plinko_distances(
    bank: dict[str, np.ndarray],
    x_obs: np.ndarray,
    distance_kind: str = "wasserstein",
    summary_kind: str = "full_hist",
    n_simulations: int | None = None,
) -> np.ndarray:
    """Compute distances d(S(x_i), S(x_obs)) across the cached simulation bank without re-simulating."""
    sl = slice(0, n_simulations) if n_simulations is not None else slice(None)
    obs_stats = plinko_summary_statistics(x_obs)

    if summary_kind == "mean_only":
        scale_m = float(bank.get("mean_scale", np.std(bank["means"][sl])))
        return np.abs(bank["means"][sl] - obs_stats["mean"]) / max(scale_m, 1e-8)

    if summary_kind == "mean_var":
        scale_m = float(bank.get("mean_scale", np.std(bank["means"][sl])))
        scale_v = float(bank.get("var_scale", np.std(bank["vars"][sl])))
        dm = (bank["means"][sl] - obs_stats["mean"]) / max(scale_m, 1e-8)
        dv = (bank["vars"][sl] - obs_stats["var"]) / max(scale_v, 1e-8)
        if distance_kind == "tv":
            return 0.5 * (np.abs(dm) + np.abs(dv))
        return np.sqrt(0.5 * (dm ** 2 + dv ** 2))

    if summary_kind != "full_hist":
        raise ValueError(f"Unknown summary_kind: {summary_kind}")

    freqs = bank["freqs"][sl]
    obs_f = obs_stats["freqs"][None, :]

    if distance_kind == "wasserstein":
        cdf_sim = np.cumsum(freqs, axis=-1)
        cdf_obs = np.cumsum(obs_f, axis=-1)
        return np.sum(np.abs(cdf_sim[:, :-1] - cdf_obs[:, :-1]), axis=-1)
    if distance_kind == "tv":
        return 0.5 * np.sum(np.abs(freqs - obs_f), axis=-1)
    if distance_kind == "l2":
        return np.sqrt(np.sum((freqs - obs_f) ** 2, axis=-1))
    if distance_kind == "chi2_sym":
        denom = np.maximum(freqs + obs_f, 1e-8)
        return np.sqrt(0.5 * np.sum(((freqs - obs_f) ** 2) / denom, axis=-1))

    raise ValueError(f"Unknown distance_kind: {distance_kind}")


def estimate_2d_kde_grid(
    samples_2d: np.ndarray,
    n_alpha: int = 121,
    n_kappa: int = 121,
    config: PlinkoConfig = DEFAULT_CONFIG,
    bw_factor: float = 1.0,
) -> dict[str, np.ndarray | float | list[float]]:
    """Estimate a normalized 2D posterior density on the standard Plinko prior grid from samples."""
    d_alpha = (config.alpha_max - config.alpha_min) / n_alpha
    d_kappa = (config.kappa_max - config.kappa_min) / n_kappa
    alpha_grid = config.alpha_min + (np.arange(n_alpha) + 0.5) * d_alpha
    kappa_grid = config.kappa_min + (np.arange(n_kappa) + 0.5) * d_kappa
    aa, kk = np.meshgrid(alpha_grid, kappa_grid, indexing="ij")

    pts = np.asarray(samples_2d, dtype=float)
    if pts.shape[0] < 8:
        post = np.ones_like(aa) / config.prior_area
    else:
        # Add tiny jitter if near-singular (e.g. very few unique accepted samples)
        try:
            kde = gaussian_kde(pts.T)
            kde.set_bandwidth(kde.factor * bw_factor)
            raw = kde(np.vstack([aa.ravel(), kk.ravel()])).reshape(aa.shape)
        except np.linalg.LinAlgError:
            rng = np.random.default_rng(0)
            jittered = pts + rng.normal(0.0, 1e-3, size=pts.shape)
            kde = gaussian_kde(jittered.T)
            raw = kde(np.vstack([aa.ravel(), kk.ravel()])).reshape(aa.shape)
        mass = float(np.sum(raw) * d_alpha * d_kappa)
        post = raw / max(mass, 1e-300)

    marg_alpha = np.sum(post, axis=1) * d_kappa
    marg_kappa = np.sum(post, axis=0) * d_alpha
    hpd_levels = compute_hpd_levels(post, d_alpha, d_kappa, credible_masses=(0.68, 0.95))
    return {
        "alpha_grid": alpha_grid,
        "kappa_grid": kappa_grid,
        "d_alpha": d_alpha,
        "d_kappa": d_kappa,
        "posterior": post,
        "marginal_alpha": marg_alpha,
        "marginal_kappa": marg_kappa,
        "hpd_levels": hpd_levels,
    }


def run_plinko_abc(
    bank: dict[str, np.ndarray],
    x_obs: np.ndarray,
    epsilon: float,
    distance_kind: str = "wasserstein",
    summary_kind: str = "full_hist",
    n_simulations: int | None = None,
    compute_kde: bool = True,
    config: PlinkoConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Run rejection ABC on the cached Plinko simulation bank for a chosen tolerance and distance."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    sl = slice(0, n_simulations) if n_simulations is not None else slice(None)
    thetas = bank["thetas"][sl]
    hists = bank["histograms"][sl]
    dists = compute_plinko_distances(
        bank,
        x_obs,
        distance_kind=distance_kind,
        summary_kind=summary_kind,
        n_simulations=n_simulations,
    )
    accepted_mask = dists < float(epsilon)
    accepted_thetas = thetas[accepted_mask]
    accepted_hists = hists[accepted_mask]
    n_total = int(dists.size)
    n_acc = int(np.sum(accepted_mask))

    best_idx = int(np.argmin(dists))
    result: dict[str, object] = {
        "distances": dists,
        "accepted_mask": accepted_mask,
        "accepted_thetas": accepted_thetas,
        "accepted_histograms": accepted_hists,
        "rejected_thetas": thetas[~accepted_mask],
        "n_accepted": n_acc,
        "n_total": n_total,
        "acceptance_rate": n_acc / max(n_total, 1),
        "best_theta": thetas[best_idx],
        "best_histogram": hists[best_idx],
        "best_distance": float(dists[best_idx]),
    }
    if n_acc > 0:
        result["mean_alpha"] = float(np.mean(accepted_thetas[:, 0]))
        result["mean_kappa"] = float(np.mean(accepted_thetas[:, 1]))
        result["std_alpha"] = float(np.std(accepted_thetas[:, 0]))
        result["std_kappa"] = float(np.std(accepted_thetas[:, 1]))
    else:
        result["mean_alpha"] = float("nan")
        result["mean_kappa"] = float("nan")
        result["std_alpha"] = float("nan")
        result["std_kappa"] = float("nan")

    if compute_kde and n_acc >= 8:
        result["kde_grid"] = estimate_2d_kde_grid(accepted_thetas, config=config)
    else:
        result["kde_grid"] = None
    return result
