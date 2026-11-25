"""Mystery Plinko (generalized Galton board) forward simulator and exact DP reference.

Mathematical specification:
- A board has R rows (default R = 12).
- After t rows (t = 0, ..., R), a ball has made k_t in {0, ..., t} rightward moves,
  corresponding to horizontal displacement s_t = 2 * k_t - t.
- At row t (for t = 0, ..., R - 1), the probability of bouncing right is:
      P(right | s_t, alpha, kappa) = sigma(alpha - kappa * s_t / sqrt(R))
  where sigma(z) = 1 / (1 + exp(-z)) is the logistic sigmoid,
  alpha in [-1.5, 1.5] controls horizontal bias, and
  kappa in [0, 3] controls the restoring pull toward the center (s_t = 0).
- Dropping N balls independently produces a landing-bin histogram
      x = (n_0, n_1, ..., n_R),   sum_k n_k = N.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import expit, gammaln, xlogy
from scipy.stats import binom


@dataclass(frozen=True)
class PlinkoConfig:
    """Default configuration for the Mystery Plinko Machine lecture problem."""

    n_rows: int = 12
    n_balls: int = 100
    alpha_min: float = -1.5
    alpha_max: float = 1.5
    kappa_min: float = 0.0
    kappa_max: float = 3.0
    alpha_true: float = 0.6
    kappa_true: float = 1.2
    obs_seed: int = 2026

    @property
    def prior_area(self) -> float:
        return (self.alpha_max - self.alpha_min) * (self.kappa_max - self.kappa_min)

    @property
    def log_prior_value(self) -> float:
        return -float(np.log(self.prior_area))

    def in_prior_support(self, alpha: np.ndarray | float, kappa: np.ndarray | float) -> np.ndarray:
        a = np.asarray(alpha)
        k = np.asarray(kappa)
        return (
            (a >= self.alpha_min)
            & (a <= self.alpha_max)
            & (k >= self.kappa_min)
            & (k <= self.kappa_max)
        )


DEFAULT_CONFIG = PlinkoConfig()


def right_transition_prob(
    s_t: np.ndarray | float,
    alpha: np.ndarray | float,
    kappa: np.ndarray | float,
    n_rows: int = DEFAULT_CONFIG.n_rows,
) -> np.ndarray:
    """Compute P(right | s_t, alpha, kappa) = sigmoid(alpha - kappa * s_t / sqrt(R))."""
    z = np.asarray(alpha, dtype=float) - np.asarray(kappa, dtype=float) * (
        np.asarray(s_t, dtype=float) / np.sqrt(float(n_rows))
    )
    return expit(z)


def binomial_bin_probabilities(
    p_right: float,
    n_rows: int = DEFAULT_CONFIG.n_rows,
) -> np.ndarray:
    """Analytic Binomial(R, p_right) landing-bin probabilities for the kappa = 0 board."""
    if not (0.0 <= p_right <= 1.0):
        raise ValueError(f"p_right must lie in [0, 1], got {p_right}")
    ks = np.arange(n_rows + 1)
    return binom.pmf(ks, n=n_rows, p=p_right)


def exact_bin_probabilities(
    alpha: np.ndarray | float,
    kappa: np.ndarray | float,
    n_rows: int = DEFAULT_CONFIG.n_rows,
) -> np.ndarray:
    """Compute exact landing-bin probabilities pi_R(k | alpha, kappa) via dynamic programming.

    Supports scalar inputs (returns shape `(R + 1,)`) as well as broadcastable
    array inputs (returns shape `broadcast_shape + (R + 1,)`).
    """
    alpha_arr, kappa_arr = np.broadcast_arrays(
        np.asarray(alpha, dtype=float),
        np.asarray(kappa, dtype=float),
    )
    scalar_input = alpha_arr.ndim == 0 and np.ndim(alpha) == 0 and np.ndim(kappa) == 0
    batch_shape = alpha_arr.shape
    sqrt_r = np.sqrt(float(n_rows))

    pi_t = np.zeros(batch_shape + (1,), dtype=float)
    pi_t[..., 0] = 1.0

    for t in range(n_rows):
        ks = np.arange(t + 1, dtype=float)
        s_t = 2.0 * ks - float(t)  # shape (t + 1,)
        p_r = expit(alpha_arr[..., None] - kappa_arr[..., None] * (s_t / sqrt_r))

        pi_next = np.zeros(batch_shape + (t + 2,), dtype=float)
        pi_next[..., :-1] += pi_t * (1.0 - p_r)
        pi_next[..., 1:] += pi_t * p_r
        pi_t = pi_next

    # Guard against tiny floating-point drift while preserving exact normalization
    pi_t = np.clip(pi_t, 0.0, 1.0)
    pi_t = pi_t / np.sum(pi_t, axis=-1, keepdims=True)
    if scalar_input:
        return pi_t.reshape(n_rows + 1)
    return pi_t


def simulate_trajectories(
    alpha: float,
    kappa: float,
    n_balls: int = DEFAULT_CONFIG.n_balls,
    n_rows: int = DEFAULT_CONFIG.n_rows,
    seed: int | np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Simulate full stochastic trajectories of N balls through the Plinko machine.

    Returns a dictionary with:
    - `k_paths`: integer rightward counts of shape `(n_balls, n_rows + 1)`
    - `s_paths`: horizontal displacements `s_t = 2 k_t - t` of shape `(n_balls, n_rows + 1)`
    - `landing_bins`: final bin indices `k_R` of shape `(n_balls,)`
    - `histogram`: bin counts `x = (n_0, ..., n_R)` of shape `(n_rows + 1,)`
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    k_paths = np.zeros((n_balls, n_rows + 1), dtype=int)
    s_paths = np.zeros((n_balls, n_rows + 1), dtype=int)
    sqrt_r = np.sqrt(float(n_rows))

    for t in range(n_rows):
        s_t = 2 * k_paths[:, t] - t
        p_r = expit(float(alpha) - float(kappa) * (s_t / sqrt_r))
        step_right = (rng.random(n_balls) < p_r).astype(int)
        k_paths[:, t + 1] = k_paths[:, t] + step_right
        s_paths[:, t + 1] = 2 * k_paths[:, t + 1] - (t + 1)

    landing_bins = k_paths[:, -1]
    histogram = np.bincount(landing_bins, minlength=n_rows + 1).astype(int)
    return {
        "k_paths": k_paths,
        "s_paths": s_paths,
        "landing_bins": landing_bins,
        "histogram": histogram,
    }


def simulate_plinko_histograms(
    alphas: np.ndarray | float,
    kappas: np.ndarray | float,
    n_balls: int = DEFAULT_CONFIG.n_balls,
    n_rows: int = DEFAULT_CONFIG.n_rows,
    seed: int | np.random.Generator | None = None,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Vectorized forward simulator producing histograms strictly from stochastic peg bounces.

    Never calls the DP benchmark; genuinely simulates row-by-row Bernoulli bounces.
    If inputs are scalars, returns shape `(R + 1,)`.
    If inputs are 1D arrays of length B, returns shape `(B, R + 1)`.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    scalar = np.ndim(alphas) == 0 and np.ndim(kappas) == 0
    a_arr = np.atleast_1d(np.asarray(alphas, dtype=float)).ravel()
    k_arr = np.atleast_1d(np.asarray(kappas, dtype=float)).ravel()
    if a_arr.shape != k_arr.shape:
        raise ValueError(f"alphas and kappas must have matching lengths, got {a_arr.shape} vs {k_arr.shape}")

    n_sim = a_arr.size
    hists = np.zeros((n_sim, n_rows + 1), dtype=int)
    sqrt_r = np.sqrt(float(n_rows))

    for start in range(0, n_sim, chunk_size):
        end = min(start + chunk_size, n_sim)
        b = end - start
        a_sub = a_arr[start:end, None]  # (b, 1)
        k_sub = k_arr[start:end, None]  # (b, 1)
        k_curr = np.zeros((b, n_balls), dtype=int)

        for t in range(n_rows):
            s_t = 2 * k_curr - t
            p_r = expit(a_sub - k_sub * (s_t / sqrt_r))
            k_curr += (rng.random((b, n_balls)) < p_r).astype(int)

        # Aggregate final landing bins into histograms row-wise
        offsets = np.arange(b, dtype=int)[:, None] * (n_rows + 1)
        flat_counts = np.bincount(
            (k_curr + offsets).ravel(),
            minlength=b * (n_rows + 1),
        )
        hists[start:end] = flat_counts.reshape(b, n_rows + 1)

    return hists[0] if scalar else hists


def sample_prior(
    n_samples: int,
    config: PlinkoConfig = DEFAULT_CONFIG,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw uniform prior samples `theta = (alpha, kappa)` of shape `(n_samples, 2)`."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    alphas = rng.uniform(config.alpha_min, config.alpha_max, size=n_samples)
    kappas = rng.uniform(config.kappa_min, config.kappa_max, size=n_samples)
    return np.column_stack([alphas, kappas])


def exact_log_likelihood(
    x_obs: np.ndarray,
    alpha: np.ndarray | float,
    kappa: np.ndarray | float,
    n_rows: int = DEFAULT_CONFIG.n_rows,
) -> np.ndarray | float:
    """Compute exact multinomial log-likelihood log p(x_obs | alpha, kappa) via DP."""
    counts = np.asarray(x_obs, dtype=float)
    if counts.shape != (n_rows + 1,):
        raise ValueError(f"Expected histogram of shape ({n_rows + 1},), got {counts.shape}")
    n_total = float(np.sum(counts))
    log_norm = gammaln(n_total + 1.0) - np.sum(gammaln(counts + 1.0))

    probs = exact_bin_probabilities(alpha, kappa, n_rows=n_rows)
    # Clip away exact zeros only where counts > 0 to avoid -inf spikes on numerical underflow
    safe_probs = np.maximum(probs, 1e-300)
    ll = log_norm + np.sum(xlogy(counts, safe_probs), axis=-1)
    if np.ndim(alpha) == 0 and np.ndim(kappa) == 0:
        return float(ll)
    return ll


def compute_hpd_levels(
    density_2d: np.ndarray,
    d_alpha: float,
    d_kappa: float,
    credible_masses: tuple[float, ...] = (0.68, 0.95),
) -> list[float]:
    """Compute 2D Highest Posterior Density (HPD) contour thresholds for normalized 2D density.

    Returns density levels sorted in ascending order (as required by `matplotlib.pyplot.contour`),
    so `levels[0]` encloses the largest credible mass (e.g. 95%) and `levels[-1]` encloses 68%.
    """
    cell_mass = np.asarray(density_2d, dtype=float).ravel() * float(d_alpha) * float(d_kappa)
    dens_flat = np.asarray(density_2d, dtype=float).ravel()
    total = np.sum(cell_mass)
    if total <= 0:
        return [0.0 for _ in credible_masses]
    cell_mass = cell_mass / total

    order = np.argsort(dens_flat)[::-1]
    cum_mass = np.cumsum(cell_mass[order])
    sorted_dens = dens_flat[order]

    thresholds = []
    for mass in credible_masses:
        idx = int(np.searchsorted(cum_mass, mass, side="left"))
        idx = min(idx, sorted_dens.size - 1)
        thresholds.append(float(sorted_dens[idx]))
    return sorted(thresholds)


def exact_posterior_grid(
    x_obs: np.ndarray,
    n_alpha: int = 121,
    n_kappa: int = 121,
    config: PlinkoConfig = DEFAULT_CONFIG,
) -> dict[str, np.ndarray | float | list[float]]:
    """Evaluate normalized exact 2D posterior p(alpha, kappa | x_obs) on a regular grid.

    Uses cell-centered grid points on `[alpha_min, alpha_max] x [kappa_min, kappa_max]`
    so Riemann cell integration `sum(posterior * d_alpha * d_kappa) == 1.0` is exact
    and boundary artifacts are avoided.
    """
    d_alpha = (config.alpha_max - config.alpha_min) / n_alpha
    d_kappa = (config.kappa_max - config.kappa_min) / n_kappa
    alpha_grid = config.alpha_min + (np.arange(n_alpha) + 0.5) * d_alpha
    kappa_grid = config.kappa_min + (np.arange(n_kappa) + 0.5) * d_kappa

    aa, kk = np.meshgrid(alpha_grid, kappa_grid, indexing="ij")  # shape (n_alpha, n_kappa)
    log_like = exact_log_likelihood(x_obs, aa, kk, n_rows=config.n_rows)
    log_unnorm = log_like + config.log_prior_value

    max_lu = float(np.max(log_unnorm))
    unnorm = np.exp(log_unnorm - max_lu)
    log_evidence = max_lu + float(np.log(np.sum(unnorm) * d_alpha * d_kappa))
    posterior = unnorm / (np.sum(unnorm) * d_alpha * d_kappa)

    marg_alpha = np.sum(posterior, axis=1) * d_kappa  # shape (n_alpha,)
    marg_kappa = np.sum(posterior, axis=0) * d_alpha  # shape (n_kappa,)

    mean_alpha = float(np.sum(alpha_grid * marg_alpha) * d_alpha)
    mean_kappa = float(np.sum(kappa_grid * marg_kappa) * d_kappa)
    var_alpha = float(np.sum(((alpha_grid - mean_alpha) ** 2) * marg_alpha) * d_alpha)
    var_kappa = float(np.sum(((kappa_grid - mean_kappa) ** 2) * marg_kappa) * d_kappa)
    cov_ak = float(
        np.sum(
            (aa - mean_alpha) * (kk - mean_kappa) * posterior
        )
        * d_alpha
        * d_kappa
    )
    corr_ak = cov_ak / np.sqrt(max(var_alpha * var_kappa, 1e-30))

    map_idx = np.unravel_index(int(np.argmax(posterior)), posterior.shape)
    hpd_levels = compute_hpd_levels(posterior, d_alpha, d_kappa, credible_masses=(0.68, 0.95))

    return {
        "alpha_grid": alpha_grid,
        "kappa_grid": kappa_grid,
        "d_alpha": d_alpha,
        "d_kappa": d_kappa,
        "log_likelihood": log_like,
        "posterior": posterior,
        "marginal_alpha": marg_alpha,
        "marginal_kappa": marg_kappa,
        "log_evidence": log_evidence,
        "mean_alpha": mean_alpha,
        "mean_kappa": mean_kappa,
        "std_alpha": float(np.sqrt(max(var_alpha, 0.0))),
        "std_kappa": float(np.sqrt(max(var_kappa, 0.0))),
        "corr": float(corr_ak),
        "map_alpha": float(alpha_grid[map_idx[0]]),
        "map_kappa": float(kappa_grid[map_idx[1]]),
        "hpd_levels": hpd_levels,
    }


def sample_from_grid_posterior(
    grid_result: dict,
    n_samples: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Draw continuous samples `(alpha, kappa)` from a normalized cell-centered 2D grid posterior."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    post = np.asarray(grid_result["posterior"], dtype=float)
    alpha_grid = np.asarray(grid_result["alpha_grid"], dtype=float)
    kappa_grid = np.asarray(grid_result["kappa_grid"], dtype=float)
    d_alpha = float(grid_result["d_alpha"])
    d_kappa = float(grid_result["d_kappa"])

    weights = post.ravel()
    weights = weights / np.sum(weights)
    flat_idx = rng.choice(weights.size, size=n_samples, replace=True, p=weights)
    i_idx, j_idx = np.unravel_index(flat_idx, post.shape)

    alphas = alpha_grid[i_idx] + rng.uniform(-0.5 * d_alpha, 0.5 * d_alpha, size=n_samples)
    kappas = kappa_grid[j_idx] + rng.uniform(-0.5 * d_kappa, 0.5 * d_kappa, size=n_samples)
    return np.column_stack([alphas, kappas])


def plinko_summary_statistics(histograms: np.ndarray) -> dict[str, np.ndarray]:
    """Compute interpretable summary statistics for 1D `(R + 1,)` or 2D `(B, R + 1)` histograms."""
    h = np.asarray(histograms, dtype=float)
    single = h.ndim == 1
    if single:
        h = h[None, :]

    n_rows = h.shape[-1] - 1
    ks = np.arange(n_rows + 1, dtype=float)[None, :]
    totals = np.maximum(np.sum(h, axis=-1, keepdims=True), 1.0)
    freqs = h / totals

    mean_k = np.sum(freqs * ks, axis=-1)
    var_k = np.sum(freqs * ((ks - mean_k[:, None]) ** 2), axis=-1)
    std_k = np.sqrt(np.maximum(var_k, 0.0))
    skew_num = np.sum(freqs * ((ks - mean_k[:, None]) ** 3), axis=-1)
    skew_k = skew_num / np.maximum(std_k ** 3, 1e-8)

    if single:
        return {
            "freqs": freqs[0],
            "mean": float(mean_k[0]),
            "var": float(var_k[0]),
            "std": float(std_k[0]),
            "skew": float(skew_k[0]),
        }
    return {
        "freqs": freqs,
        "mean": mean_k,
        "var": var_k,
        "std": std_k,
        "skew": skew_k,
    }


def generate_default_observation(config: PlinkoConfig = DEFAULT_CONFIG) -> dict[str, object]:
    """Generate the reproducible benchmark Mystery Plinko observation for the lecture."""
    sim = simulate_trajectories(
        alpha=config.alpha_true,
        kappa=config.kappa_true,
        n_balls=config.n_balls,
        n_rows=config.n_rows,
        seed=config.obs_seed,
    )
    return {
        "x_obs": sim["histogram"],
        "k_paths": sim["k_paths"],
        "s_paths": sim["s_paths"],
        "alpha_true": config.alpha_true,
        "kappa_true": config.kappa_true,
        "config": config,
    }
