"""Neural Simulation-Based Inference estimators (NPE, NLE, NRE) for the Mystery Plinko Machine.

Implements all three paradigms explicitly in PyTorch on a shared prior-predictive
simulation dataset D_train = {(theta_i, x_i)}_{i=1}^M:

1. NPE (Neural Posterior Estimation):
   Full-covariance 2D Mixture Density Network in unconstrained logit-prior coordinates
   with the exact analytical change-of-variables Jacobian so that:
   - Every posterior sample strictly respects the bounded uniform prior support.
   - Exact physical densities integrate to 1 over [-1.5, 1.5] x [0, 3.0].

2. NLE (Neural Likelihood Estimation):
   Structured surrogate likelihood predicting normalized landing-bin probabilities
   pi_phi(k | theta) via softmax, combined with the exact Multinomial observation model.

3. NRE (Neural Ratio Estimation):
   Amortized likelihood-to-evidence ratio estimator trained via balanced binary
   cross-entropy classification between joint pairs p(theta, x) and product-of-marginals
   pairs p(theta)p(x), using a two-tower bilinear-residual interaction classifier and
   multi-negative variance reduction.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .plinko import (
    DEFAULT_CONFIG,
    PlinkoConfig,
    compute_hpd_levels,
    exact_posterior_grid,
    generate_default_observation,
    sample_prior,
    simulate_plinko_histograms,
)


# -----------------------------------------------------------------------------
# Coordinate bijections for bounded prior support [alpha_min, alpha_max] x [kappa_min, kappa_max]
# -----------------------------------------------------------------------------

def _theta_to_logit_torch(
    theta: torch.Tensor,
    config: PlinkoConfig = DEFAULT_CONFIG,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map physical theta in [min, max] to unconstrained u in R^2 and return log |det dtheta/du|."""
    lows = torch.tensor([config.alpha_min, config.kappa_min], device=theta.device, dtype=theta.dtype)
    spans = torch.tensor(
        [config.alpha_max - config.alpha_min, config.kappa_max - config.kappa_min],
        device=theta.device,
        dtype=theta.dtype,
    )
    z = torch.clamp((theta - lows) / spans, eps, 1.0 - eps)
    u = torch.log(z) - torch.log1p(-z)
    log_abs_det_jac = torch.sum(torch.log(spans) + torch.log(z) + torch.log1p(-z), dim=-1)
    return u, log_abs_det_jac


def _logit_to_theta_torch(
    u: torch.Tensor,
    config: PlinkoConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Map unconstrained u in R^2 back into strict interior of prior box."""
    lows = torch.tensor([config.alpha_min, config.kappa_min], device=u.device, dtype=u.dtype)
    spans = torch.tensor(
        [config.alpha_max - config.alpha_min, config.kappa_max - config.kappa_min],
        device=u.device,
        dtype=u.dtype,
    )
    z = torch.sigmoid(u)
    return lows + spans * z


def _normalize_theta_features(
    theta: torch.Tensor,
    config: PlinkoConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Affine-scale physical theta into [-1, 1]^2 for well-conditioned network inputs."""
    mids = torch.tensor(
        [0.5 * (config.alpha_min + config.alpha_max), 0.5 * (config.kappa_min + config.kappa_max)],
        device=theta.device,
        dtype=theta.dtype,
    )
    half_spans = torch.tensor(
        [0.5 * (config.alpha_max - config.alpha_min), 0.5 * (config.kappa_max - config.kappa_min)],
        device=theta.device,
        dtype=theta.dtype,
    )
    return (theta - mids) / half_spans


def _histogram_features_torch(x: torch.Tensor) -> torch.Tensor:
    """Convert raw bin counts x of shape (..., R+1) to normalized frequencies + standardized moments."""
    totals = torch.clamp(torch.sum(x, dim=-1, keepdim=True), min=1.0)
    freqs = x / totals
    n_bins = x.shape[-1]
    ks = torch.arange(n_bins, device=x.device, dtype=x.dtype) / float(n_bins - 1)
    mean_norm = torch.sum(freqs * ks, dim=-1, keepdim=True)  # in [0, 1]
    var_norm = torch.sum(freqs * ((ks - mean_norm) ** 2), dim=-1, keepdim=True) * 10.0
    return torch.cat([freqs, (mean_norm - 0.5) * 2.0, var_norm], dim=-1)


# -----------------------------------------------------------------------------
# 1. Neural Posterior Estimator (NPE) — Full-Covariance Logit-MDN
# -----------------------------------------------------------------------------

class PlinkoNPEModel(nn.Module):
    """Prior-amortized conditional density estimator q_phi(theta | x) on bounded prior support."""

    def __init__(
        self,
        n_rows: int = DEFAULT_CONFIG.n_rows,
        n_components: int = 8,
        hidden_dim: int = 128,
        config: PlinkoConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.n_components = n_components
        self.config = config
        in_dim = (n_rows + 1) + 2

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.logits_head = nn.Linear(hidden_dim, n_components)
        self.means_head = nn.Linear(hidden_dim, n_components * 2)
        # Lower-triangular Cholesky parameters per component: (log_l11, l21, log_l22)
        self.chol_head = nn.Linear(hidden_dim, n_components * 3)

    def _mixture_params(self, x_hist: torch.Tensor):
        feat = _histogram_features_torch(x_hist)
        h = self.encoder(feat)
        log_w = F.log_softmax(self.logits_head(h), dim=-1)  # (B, K)
        means = self.means_head(h).view(-1, self.n_components, 2)  # (B, K, 2)
        chol_raw = self.chol_head(h).view(-1, self.n_components, 3)
        l11 = F.softplus(chol_raw[..., 0]) + 0.03
        l21 = chol_raw[..., 1]
        l22 = F.softplus(chol_raw[..., 2]) + 0.03
        return log_w, means, l11, l21, l22

    def log_prob(self, theta: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        """Evaluate exact physical log density log q_phi(theta | x) including support Jacobian."""
        if x_hist.shape[0] == 1 and theta.shape[0] > 1:
            x_hist = x_hist.expand(theta.shape[0], -1)

        inside = (
            (theta[:, 0] > self.config.alpha_min)
            & (theta[:, 0] < self.config.alpha_max)
            & (theta[:, 1] > self.config.kappa_min)
            & (theta[:, 1] < self.config.kappa_max)
        )
        u, log_abs_det = _theta_to_logit_torch(theta, config=self.config)
        log_w, means, l11, l21, l22 = self._mixture_params(x_hist)

        du = u[:, None, :] - means  # (B, K, 2)
        y1 = du[..., 0] / l11
        y2 = (du[..., 1] - l21 * y1) / l22
        quad = y1 ** 2 + y2 ** 2
        log_det_l = torch.log(l11) + torch.log(l22)
        log_gauss_u = -0.5 * quad - log_det_l - float(np.log(2.0 * np.pi))
        log_q_u = torch.logsumexp(log_w + log_gauss_u, dim=-1)
        log_q_theta = log_q_u - log_abs_det
        return torch.where(inside, log_q_theta, torch.full_like(log_q_theta, -1e12))

    @torch.no_grad()
    def sample(
        self,
        x_hist: np.ndarray | torch.Tensor,
        n_samples: int = 2000,
        seed: int | np.random.Generator | None = 42,
    ) -> np.ndarray:
        """Draw exact direct posterior samples theta ~ q_phi(theta | x_obs) inside prior bounds."""
        self.eval()
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if not isinstance(x_hist, torch.Tensor):
            x_t = torch.tensor(np.asarray(x_hist, dtype=np.float32)).reshape(1, -1)
        else:
            x_t = x_hist.reshape(1, -1).float()

        log_w, means, l11, l21, l22 = self._mixture_params(x_t)
        weights = torch.exp(log_w[0]).cpu().numpy()
        weights = weights / np.sum(weights)
        comp_idx = rng.choice(self.n_components, size=n_samples, p=weights)
        eps = rng.standard_normal((n_samples, 2)).astype(np.float32)

        m_sel = means[0].cpu().numpy()[comp_idx]
        l11_sel = l11[0].cpu().numpy()[comp_idx]
        l21_sel = l21[0].cpu().numpy()[comp_idx]
        l22_sel = l22[0].cpu().numpy()[comp_idx]

        u1 = m_sel[:, 0] + l11_sel * eps[:, 0]
        u2 = m_sel[:, 1] + l21_sel * eps[:, 0] + l22_sel * eps[:, 1]
        u_t = torch.from_numpy(np.column_stack([u1, u2]).astype(np.float32))
        theta_t = _logit_to_theta_torch(u_t, config=self.config)
        return theta_t.cpu().numpy().astype(float)


# -----------------------------------------------------------------------------
# 2. Neural Likelihood Estimator (NLE) — Structured Categorical-Multinomial Net
# -----------------------------------------------------------------------------

class PlinkoNLEModel(nn.Module):
    """Structured surrogate likelihood predicting normalized bin probabilities pi_phi(k | theta)."""

    def __init__(
        self,
        n_rows: int = DEFAULT_CONFIG.n_rows,
        hidden_dim: int = 128,
        config: PlinkoConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_rows + 1),
        )

    def log_bin_probs(self, theta: torch.Tensor) -> torch.Tensor:
        t_norm = _normalize_theta_features(theta, config=self.config)
        return F.log_softmax(self.net(t_norm), dim=-1)

    def bin_probs(self, theta: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = (
                theta.float()
                if isinstance(theta, torch.Tensor)
                else torch.tensor(np.asarray(theta, dtype=np.float32))
            )
            single = t.ndim == 1
            if single:
                t = t[None, :]
            probs = torch.exp(self.log_bin_probs(t)).cpu().numpy().astype(float)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            return probs[0] if single else probs

    def log_likelihood_torch(self, x_hist: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Exact normalized multinomial log-likelihood log q_phi(x | theta)."""
        if x_hist.shape[0] == 1 and theta.shape[0] > 1:
            x_hist = x_hist.expand(theta.shape[0], -1)
        log_p = self.log_bin_probs(theta)
        n_tot = torch.sum(x_hist, dim=-1)
        log_comb = torch.lgamma(n_tot + 1.0) - torch.sum(torch.lgamma(x_hist + 1.0), dim=-1)
        return log_comb + torch.sum(x_hist * log_p, dim=-1)


# -----------------------------------------------------------------------------
# 3. Neural Ratio Estimator (NRE) — Joint vs Marginal Classifier
# -----------------------------------------------------------------------------

def log_ratio_from_classifier_logit(
    logits: np.ndarray | float,
    joint_class_prior: float = 0.5,
) -> np.ndarray | float:
    """Convert optimal binary classifier logit to log density ratio log[p(theta,x)/(p(theta)p(x))].

    For general positive class prior pi_1 = P(y = 1) in (0, 1):
        D*(theta, x) / (1 - D*(theta, x)) = (pi_1 / (1 - pi_1)) * r(theta, x)
        => log r(theta, x) = logit(D*(theta, x)) - log(pi_1 / (1 - pi_1)).
    When classes are balanced (pi_1 = 0.5), log(pi_1 / (1 - pi_1)) = 0.
    """
    if not (0.0 < joint_class_prior < 1.0):
        raise ValueError(f"joint_class_prior must lie strictly in (0, 1), got {joint_class_prior}")
    prior_log_odds = float(np.log(joint_class_prior / (1.0 - joint_class_prior)))
    res = np.asarray(logits, dtype=float) - prior_log_odds
    if np.ndim(logits) == 0:
        return float(res)
    return res


class PlinkoNREModel(nn.Module):
    """Two-tower bilinear-interaction classifier distinguishing p(theta, x) from p(theta)p(x)."""

    def __init__(
        self,
        n_rows: int = DEFAULT_CONFIG.n_rows,
        hidden_dim: int = 128,
        config: PlinkoConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.hidden_dim = hidden_dim
        self.config = config

        # Parameter tower: normalized (alpha, kappa) + quadratic features
        self.theta_tower = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Observation tower: normalized frequencies, smoothed log-frequencies, and standardized moments
        x_in_dim = 2 * (n_rows + 1) + 2
        self.hist_tower = nn.Sequential(
            nn.Linear(x_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Bilinear natural-parameter interaction + nonlinear residual MLP head
        self.bilin = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_theta(self, theta: torch.Tensor) -> torch.Tensor:
        t = _normalize_theta_features(theta, config=self.config)
        t_poly = torch.cat([t, t ** 2, t[:, :1] * t[:, 1:2]], dim=-1)
        return self.theta_tower(t_poly)

    def encode_hist(self, x_hist: torch.Tensor) -> torch.Tensor:
        base = _histogram_features_torch(x_hist)
        totals = torch.clamp(torch.sum(x_hist, dim=-1, keepdim=True), min=1.0)
        freqs = x_hist / totals
        log_f = (torch.log(freqs + 0.008) + 2.6) * 0.5
        return self.hist_tower(torch.cat([base, log_f], dim=-1))

    def score_encoded(self, h_t: torch.Tensor, h_x: torch.Tensor) -> torch.Tensor:
        if h_x.shape[0] == 1 and h_t.shape[0] > 1:
            h_x = h_x.expand(h_t.shape[0], -1)
        if h_t.shape[0] == 1 and h_x.shape[0] > 1:
            h_t = h_t.expand(h_x.shape[0], -1)
        dot_term = torch.sum(self.bilin(h_t) * h_x, dim=-1) / float(np.sqrt(self.hidden_dim))
        inter = torch.cat([h_t * h_x, (h_t - h_x) ** 2, h_x], dim=-1)
        return dot_term + self.head(inter).squeeze(-1)

    def forward(self, theta: torch.Tensor, x_hist: torch.Tensor) -> torch.Tensor:
        """Return raw classifier logit s_phi(theta, x) = log [D / (1 - D)]."""
        h_t = self.encode_theta(theta)
        h_x = self.encode_hist(x_hist)
        return self.score_encoded(h_t, h_x)

    @torch.no_grad()
    def classify_and_ratio(
        self,
        theta: np.ndarray,
        x_hist: np.ndarray,
        joint_class_prior: float = 0.5,
    ) -> dict[str, np.ndarray]:
        self.eval()
        t_t = torch.tensor(np.atleast_2d(np.asarray(theta, dtype=np.float32)))
        x_t = torch.tensor(np.atleast_2d(np.asarray(x_hist, dtype=np.float32)))
        logits = self.forward(t_t, x_t).cpu().numpy().astype(float)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        log_r = log_ratio_from_classifier_logit(logits, joint_class_prior=joint_class_prior)
        return {
            "logit": logits,
            "prob_joint": probs,
            "odds": np.exp(np.clip(logits, -30.0, 30.0)),
            "log_ratio": log_r,
            "ratio": np.exp(np.clip(log_r, -30.0, 30.0)),
        }


def make_nre_training_pairs(
    thetas: np.ndarray,
    histograms: np.ndarray,
    seed: int | np.random.Generator | None = 42,
) -> dict[str, np.ndarray]:
    """Construct balanced positive p(theta, x) and negative p(theta)p(x) pairs."""
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    n = thetas.shape[0]
    perm = rng.permutation(n)
    if n > 1:
        fixed = perm == np.arange(n)
        if np.any(fixed):
            perm = (np.arange(n) + rng.integers(1, n)) % n

    theta_all = np.vstack([thetas, thetas])
    x_all = np.vstack([histograms, histograms[perm]])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return {
        "thetas": theta_all,
        "histograms": x_all,
        "labels": labels,
        "perm_indices": perm,
    }


def _nre_balanced_bce_loss(
    nre: PlinkoNREModel,
    b_theta: torch.Tensor,
    b_hist: torch.Tensor,
    n_negatives: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Compute exact 50/50 balanced BCE loss using K_neg marginal permutations per positive pair.

    Population objective:
        0.5 * E_{p(theta,x)}[softplus(-s(theta,x))] + 0.5 * E_{p(theta)p(x)}[softplus(s(theta,x))].
    Bayes-optimal logit remains strictly s*(theta, x) = log [p(x | theta) / p(x)].
    """
    bsz = b_theta.shape[0]
    h_t = nre.encode_theta(b_theta)
    h_x = nre.encode_hist(b_hist)
    logits_pos = nre.score_encoded(h_t, h_x)
    loss_pos = F.softplus(-logits_pos).mean()

    k_use = max(1, min(int(n_negatives), bsz - 1))
    shifts = rng.choice(np.arange(1, bsz), size=k_use, replace=False)
    neg_terms = []
    for sh in shifts:
        h_x_neg = torch.roll(h_x, shifts=int(sh), dims=0)
        neg_terms.append(F.softplus(nre.score_encoded(h_t, h_x_neg)).mean())
    loss_neg = torch.stack(neg_terms).mean()
    return 0.5 * (loss_pos + loss_neg)


# -----------------------------------------------------------------------------
# Posterior Grid & MCMC Evaluation for NPE, NLE, and NRE
# -----------------------------------------------------------------------------

def evaluate_neural_posterior_grid(
    model: nn.Module,
    method: str,
    x_obs: np.ndarray,
    n_alpha: int = 121,
    n_kappa: int = 121,
    config: PlinkoConfig = DEFAULT_CONFIG,
) -> dict[str, object]:
    """Evaluate normalized 2D posterior p(alpha, kappa | x_obs) on standard grid for NPE, NLE, or NRE."""
    model.eval()
    d_alpha = (config.alpha_max - config.alpha_min) / n_alpha
    d_kappa = (config.kappa_max - config.kappa_min) / n_kappa
    alpha_grid = config.alpha_min + (np.arange(n_alpha) + 0.5) * d_alpha
    kappa_grid = config.kappa_min + (np.arange(n_kappa) + 0.5) * d_kappa
    aa, kk = np.meshgrid(alpha_grid, kappa_grid, indexing="ij")

    pts = np.column_stack([aa.ravel(), kk.ravel()]).astype(np.float32)
    theta_t = torch.from_numpy(pts)
    x_t = torch.tensor(np.asarray(x_obs, dtype=np.float32)).reshape(1, -1)

    with torch.no_grad():
        if method.upper() == "NPE":
            log_target = model.log_prob(theta_t, x_t).cpu().numpy().reshape(aa.shape)
            raw_surface = np.exp(np.clip(log_target, -80.0, 80.0))
            norm_check = float(np.sum(raw_surface) * d_alpha * d_kappa)
            posterior = raw_surface / max(norm_check, 1e-300)
            aux_surface = log_target
        elif method.upper() == "NLE":
            log_like = model.log_likelihood_torch(x_t, theta_t).cpu().numpy().reshape(aa.shape)
            log_unnorm = log_like + config.log_prior_value
            max_lu = float(np.max(log_unnorm))
            unnorm = np.exp(log_unnorm - max_lu)
            posterior = unnorm / (np.sum(unnorm) * d_alpha * d_kappa)
            aux_surface = log_like
            norm_check = 1.0
        elif method.upper() == "NRE":
            log_ratio = model(theta_t, x_t).cpu().numpy().reshape(aa.shape)
            log_unnorm = log_ratio + config.log_prior_value
            # Evidence-free ratio integral check: int p(theta) r(theta, x) dtheta ~ 1
            norm_check = float(np.sum(np.exp(np.clip(log_unnorm, -60.0, 60.0))) * d_alpha * d_kappa)
            max_lu = float(np.max(log_unnorm))
            unnorm = np.exp(log_unnorm - max_lu)
            posterior = unnorm / (np.sum(unnorm) * d_alpha * d_kappa)
            aux_surface = log_ratio
        else:
            raise ValueError(f"Unknown method: {method}")

    marg_alpha = np.sum(posterior, axis=1) * d_kappa
    marg_kappa = np.sum(posterior, axis=0) * d_alpha
    mean_alpha = float(np.sum(alpha_grid * marg_alpha) * d_alpha)
    mean_kappa = float(np.sum(kappa_grid * marg_kappa) * d_kappa)
    var_alpha = float(np.sum(((alpha_grid - mean_alpha) ** 2) * marg_alpha) * d_alpha)
    var_kappa = float(np.sum(((kappa_grid - mean_kappa) ** 2) * marg_kappa) * d_kappa)
    cov_ak = float(np.sum((aa - mean_alpha) * (kk - mean_kappa) * posterior) * d_alpha * d_kappa)
    corr_ak = cov_ak / np.sqrt(max(var_alpha * var_kappa, 1e-30))
    map_idx = np.unravel_index(int(np.argmax(posterior)), posterior.shape)
    hpd_levels = compute_hpd_levels(posterior, d_alpha, d_kappa, credible_masses=(0.68, 0.95))

    return {
        "method": method.upper(),
        "alpha_grid": alpha_grid,
        "kappa_grid": kappa_grid,
        "d_alpha": d_alpha,
        "d_kappa": d_kappa,
        "posterior": posterior,
        "aux_surface": aux_surface,
        "norm_check": norm_check,
        "marginal_alpha": marg_alpha,
        "marginal_kappa": marg_kappa,
        "mean_alpha": mean_alpha,
        "mean_kappa": mean_kappa,
        "std_alpha": float(np.sqrt(max(var_alpha, 0.0))),
        "std_kappa": float(np.sqrt(max(var_kappa, 0.0))),
        "corr": float(corr_ak),
        "map_alpha": float(alpha_grid[map_idx[0]]),
        "map_kappa": float(kappa_grid[map_idx[1]]),
        "hpd_levels": hpd_levels,
    }


def sample_posterior_metropolis(
    model: nn.Module,
    method: str,
    x_obs: np.ndarray,
    n_samples: int = 2000,
    burn_in: int = 500,
    proposal_scales: tuple[float, float] = (0.12, 0.22),
    config: PlinkoConfig = DEFAULT_CONFIG,
    seed: int = 42,
) -> np.ndarray:
    """Lightweight Metropolis-Hastings posterior sampler using learned NLE likelihood or NRE ratio."""
    model.eval()
    rng = np.random.default_rng(seed)
    x_t = torch.tensor(np.asarray(x_obs, dtype=np.float32)).reshape(1, -1)

    def eval_log_post(theta_2d: np.ndarray) -> float:
        if not bool(config.in_prior_support(theta_2d[0], theta_2d[1])):
            return -np.inf
        t_t = torch.tensor(theta_2d[None, :], dtype=torch.float32)
        with torch.no_grad():
            if method.upper() == "NLE":
                val = float(model.log_likelihood_torch(x_t, t_t).item()) + config.log_prior_value
            elif method.upper() == "NRE":
                val = float(model(t_t, x_t).item()) + config.log_prior_value
            else:
                raise ValueError("sample_posterior_metropolis is intended for NLE or NRE")
        return val

    curr = np.array(
        [
            0.5 * (config.alpha_min + config.alpha_max),
            0.5 * (config.kappa_min + config.kappa_max),
        ],
        dtype=float,
    )
    curr_lp = eval_log_post(curr)
    total_steps = burn_in + n_samples
    chain = np.zeros((n_samples, 2), dtype=float)
    scales = np.asarray(proposal_scales, dtype=float)

    for step in range(total_steps):
        prop = curr + rng.normal(0.0, scales, size=2)
        prop_lp = eval_log_post(prop)
        if np.isfinite(prop_lp) and np.log(rng.random()) < (prop_lp - curr_lp):
            curr = prop
            curr_lp = prop_lp
        if step >= burn_in:
            chain[step - burn_in] = curr
    return chain


# -----------------------------------------------------------------------------
# Reproducible Training & Checkpoint Persistence
# -----------------------------------------------------------------------------

def train_neural_sbi_suite(
    n_train: int = 40000,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 2e-3,
    nre_negatives: int = 24,
    seed: int = 202601,
    config: PlinkoConfig = DEFAULT_CONFIG,
    checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Train NPE, NLE, and NRE on a common prior-predictive dataset and optionally save checkpoint."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Strictly independent prior-predictive simulations (never uses alpha_true / kappa_true)
    thetas = sample_prior(n_train, config=config, seed=rng).astype(np.float32)
    hists = simulate_plinko_histograms(
        thetas[:, 0],
        thetas[:, 1],
        n_balls=config.n_balls,
        n_rows=config.n_rows,
        seed=rng,
    ).astype(np.float32)

    npe = PlinkoNPEModel(n_rows=config.n_rows, n_components=8, hidden_dim=128, config=config)
    nle = PlinkoNLEModel(n_rows=config.n_rows, hidden_dim=128, config=config)
    nre = PlinkoNREModel(n_rows=config.n_rows, hidden_dim=128, config=config)

    opt_npe = torch.optim.AdamW(npe.parameters(), lr=lr, weight_decay=5e-5)
    opt_nle = torch.optim.AdamW(nle.parameters(), lr=lr, weight_decay=5e-5)
    opt_nre = torch.optim.AdamW(nre.parameters(), lr=lr, weight_decay=1e-5)

    sched_npe = torch.optim.lr_scheduler.CosineAnnealingLR(opt_npe, T_max=epochs)
    sched_nle = torch.optim.lr_scheduler.CosineAnnealingLR(opt_nle, T_max=epochs)
    sched_nre = torch.optim.lr_scheduler.CosineAnnealingLR(opt_nre, T_max=epochs)

    theta_tensor = torch.from_numpy(thetas)
    hist_tensor = torch.from_numpy(hists)

    losses = {"npe": [], "nle": [], "nre": []}

    for _epoch in range(epochs):
        perm = rng.permutation(n_train)
        ep_npe, ep_nle, ep_nre, n_batches = 0.0, 0.0, 0.0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if idx.size < 32:
                continue
            b_theta = theta_tensor[idx]
            b_hist = hist_tensor[idx]

            # 1. NPE step: minimize empirical -log q_phi(theta | x)
            loss_npe = -torch.mean(npe.log_prob(b_theta, b_hist))
            opt_npe.zero_grad()
            loss_npe.backward()
            torch.nn.utils.clip_grad_norm_(npe.parameters(), 5.0)
            opt_npe.step()

            # 2. NLE step: minimize empirical -log q_phi(x | theta)
            loss_nle = -torch.mean(nle.log_likelihood_torch(b_hist, b_theta))
            opt_nle.zero_grad()
            loss_nle.backward()
            torch.nn.utils.clip_grad_norm_(nle.parameters(), 5.0)
            opt_nle.step()

            # 3. NRE step: variance-reduced balanced binary classification
            loss_nre = _nre_balanced_bce_loss(nre, b_theta, b_hist, nre_negatives, rng)
            opt_nre.zero_grad()
            loss_nre.backward()
            torch.nn.utils.clip_grad_norm_(nre.parameters(), 5.0)
            opt_nre.step()

            ep_npe += float(loss_npe.item())
            ep_nle += float(loss_nle.item())
            ep_nre += float(loss_nre.item())
            n_batches += 1

        sched_npe.step()
        sched_nle.step()
        sched_nre.step()
        losses["npe"].append(ep_npe / max(n_batches, 1))
        losses["nle"].append(ep_nle / max(n_batches, 1))
        losses["nre"].append(ep_nre / max(n_batches, 1))

    npe.eval()
    nle.eval()
    nre.eval()

    payload = {
        "arch_version": 2,
        "n_train": n_train,
        "epochs": epochs,
        "seed": seed,
        "npe_state_dict": npe.state_dict(),
        "nle_state_dict": nle.state_dict(),
        "nre_state_dict": nre.state_dict(),
        "losses": {k: np.array(v, dtype=float) for k, v in losses.items()},
        "sample_train_thetas": thetas[:1000],
        "sample_train_hists": hists[:1000],
    }
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, p)

    return {
        "npe": npe,
        "nle": nle,
        "nre": nre,
        "losses": payload["losses"],
        "n_train": n_train,
        "epochs": epochs,
        "seed": seed,
        "sample_train_thetas": payload["sample_train_thetas"],
        "sample_train_hists": payload["sample_train_hists"],
    }


def load_or_train_neural_sbi_suite(
    checkpoint_path: str | Path,
    config: PlinkoConfig = DEFAULT_CONFIG,
    fast_fallback_if_missing: bool = False,
) -> dict[str, object]:
    """Load precomputed NPE/NLE/NRE models from disk, or retrain if missing or outdated."""
    p = Path(checkpoint_path)
    if p.exists():
        payload = torch.load(p, map_location="cpu", weights_only=False)
        if int(payload.get("arch_version", 1)) == 2:
            npe = PlinkoNPEModel(n_rows=config.n_rows, n_components=8, hidden_dim=128, config=config)
            nle = PlinkoNLEModel(n_rows=config.n_rows, hidden_dim=128, config=config)
            nre = PlinkoNREModel(n_rows=config.n_rows, hidden_dim=128, config=config)
            npe.load_state_dict(payload["npe_state_dict"])
            nle.load_state_dict(payload["nle_state_dict"])
            nre.load_state_dict(payload["nre_state_dict"])
            npe.eval()
            nle.eval()
            nre.eval()
            return {
                "npe": npe,
                "nle": nle,
                "nre": nre,
                "losses": payload["losses"],
                "n_train": int(payload["n_train"]),
                "epochs": int(payload["epochs"]),
                "seed": int(payload["seed"]),
                "sample_train_thetas": payload["sample_train_thetas"],
                "sample_train_hists": payload["sample_train_hists"],
            }

    n_train = 8000 if fast_fallback_if_missing else 40000
    epochs = 30 if fast_fallback_if_missing else 80
    return train_neural_sbi_suite(
        n_train=n_train,
        epochs=epochs,
        config=config,
        checkpoint_path=p,
    )


# -----------------------------------------------------------------------------
# Epoch-by-Epoch Training Trajectories for Interactive Galton-Board Train Widgets
# -----------------------------------------------------------------------------

def build_or_load_training_trajectories(
    cache_path: str | Path,
    config: PlinkoConfig = DEFAULT_CONFIG,
    force_recompute: bool = False,
) -> dict[str, object]:
    """Record real PyTorch epoch-by-epoch training snapshots for NPE, NLE, and NRE on Galton Boards.

    Saves compact 31x31 posterior/likelihood/classifier snapshots across key epochs
    [0, 1, 2, 4, 7, 12, 20, 32, 50, 80] plus full 81-point train & validation loss curves
    for multiple simulation budgets and algorithmic settings.
    """
    p = Path(cache_path)
    if p.exists() and not force_recompute:
        return torch.load(p, map_location="cpu", weights_only=False)

    rng = np.random.default_rng(202610)
    torch.manual_seed(202610)

    # Master training pool (36,000) + strictly disjoint validation pool (3,000)
    n_pool = 36000
    n_val = 3000
    th_pool = torch.from_numpy(sample_prior(n_pool, config=config, seed=rng).astype(np.float32))
    x_pool = torch.from_numpy(
        simulate_plinko_histograms(
            th_pool[:, 0].numpy(), th_pool[:, 1].numpy(), n_balls=config.n_balls, n_rows=config.n_rows, seed=rng
        ).astype(np.float32)
    )
    th_val = torch.from_numpy(sample_prior(n_val, config=config, seed=rng).astype(np.float32))
    x_val = torch.from_numpy(
        simulate_plinko_histograms(
            th_val[:, 0].numpy(), th_val[:, 1].numpy(), n_balls=config.n_balls, n_rows=config.n_rows, seed=rng
        ).astype(np.float32)
    )

    obs0 = generate_default_observation(config=config)
    x_obs = np.asarray(obs0["x_obs"], dtype=int)
    na, nk = 31, 31
    ex_grid = exact_posterior_grid(x_obs, n_alpha=na, n_kappa=nk, config=config)
    da, dk = ex_grid["d_alpha"], ex_grid["d_kappa"]
    probe_theta = torch.tensor([[config.alpha_true, config.kappa_true]], dtype=torch.float32)

    snap_epochs = [0, 1, 2, 4, 7, 12, 20, 32, 50, 80]
    max_ep = 80

    def _tv(post_a: np.ndarray) -> float:
        return float(0.5 * np.sum(np.abs(post_a - ex_grid["posterior"])) * da * dk)

    # 1. NPE runs: vary N_sim and K_components
    npe_specs = [
        ("N36000_K8", 36000, 8, 256),
        ("N4000_K8", 4000, 8, 128),
        ("N800_K8", 800, 8, 64),
        ("N36000_K1", 36000, 1, 256),
    ]
    npe_runs: dict[str, dict[str, object]] = {}
    for key, n_use, k_comp, bsz in npe_specs:
        torch.manual_seed(3100 + n_use + k_comp)
        sub_rng = np.random.default_rng(3100 + n_use + k_comp)
        model = PlinkoNPEModel(n_rows=config.n_rows, n_components=k_comp, hidden_dim=96, config=config)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
        tr_loss, va_loss, tv_hist = [], [], []
        snaps: dict[int, np.ndarray] = {}

        for ep in range(max_ep + 1):
            model.eval()
            with torch.no_grad():
                l_tr_eval = float(-torch.mean(model.log_prob(th_pool[: min(n_use, 3000)], x_pool[: min(n_use, 3000)])).item())
                l_va_eval = float(-torch.mean(model.log_prob(th_val, x_val)).item())
                g = evaluate_neural_posterior_grid(model, "NPE", x_obs, n_alpha=na, n_kappa=nk, config=config)
            tr_loss.append(l_tr_eval)
            va_loss.append(l_va_eval)
            tv_hist.append(_tv(g["posterior"]))
            if ep in snap_epochs:
                snaps[ep] = g["posterior"].astype(np.float32)
            if ep == max_ep:
                break

            model.train()
            perm = sub_rng.permutation(n_use)
            for s in range(0, n_use, bsz):
                idx = perm[s : s + bsz]
                if idx.size < 16:
                    continue
                loss = -torch.mean(model.log_prob(th_pool[idx], x_pool[idx]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()

        npe_runs[key] = {
            "n_sim": n_use,
            "k_comp": k_comp,
            "train_loss": np.array(tr_loss, dtype=float),
            "val_loss": np.array(va_loss, dtype=float),
            "tv_hist": np.array(tv_hist, dtype=float),
            "snaps": snaps,
        }

    # 2. NLE runs: vary N_sim (store log-likelihood surface + predicted bin probs at theta*)
    nle_specs = [
        ("N36000", 36000, 256),
        ("N4000", 4000, 128),
        ("N600", 600, 48),
    ]
    nle_runs: dict[str, dict[str, object]] = {}
    for key, n_use, bsz in nle_specs:
        torch.manual_seed(4200 + n_use)
        sub_rng = np.random.default_rng(4200 + n_use)
        model = PlinkoNLEModel(n_rows=config.n_rows, hidden_dim=96, config=config)
        opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=3e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
        tr_loss, va_loss, tv_hist = [], [], []
        snaps_ll: dict[int, np.ndarray] = {}
        snaps_bins: dict[int, np.ndarray] = {}

        for ep in range(max_ep + 1):
            model.eval()
            with torch.no_grad():
                l_tr_eval = float(-torch.mean(model.log_likelihood_torch(x_pool[: min(n_use, 3000)], th_pool[: min(n_use, 3000)])).item())
                l_va_eval = float(-torch.mean(model.log_likelihood_torch(x_val, th_val)).item())
                g = evaluate_neural_posterior_grid(model, "NLE", x_obs, n_alpha=na, n_kappa=nk, config=config)
                b_probs = model.bin_probs(probe_theta)[0]
            tr_loss.append(l_tr_eval)
            va_loss.append(l_va_eval)
            tv_hist.append(_tv(g["posterior"]))
            if ep in snap_epochs:
                ll = g["aux_surface"] - np.max(g["aux_surface"])
                snaps_ll[ep] = np.clip(ll, -40.0, 0.0).astype(np.float32)
                snaps_bins[ep] = b_probs.astype(np.float32)
            if ep == max_ep:
                break

            model.train()
            perm = sub_rng.permutation(n_use)
            for s in range(0, n_use, bsz):
                idx = perm[s : s + bsz]
                if idx.size < 16:
                    continue
                loss = -torch.mean(model.log_likelihood_torch(x_pool[idx], th_pool[idx]))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()

        nle_runs[key] = {
            "n_sim": n_use,
            "train_loss": np.array(tr_loss, dtype=float),
            "val_loss": np.array(va_loss, dtype=float),
            "tv_hist": np.array(tv_hist, dtype=float),
            "snaps_loglike": snaps_ll,
            "snaps_bins": snaps_bins,
        }

    # 3. NRE runs: compare multi-negative K_neg=24 vs naive K_neg=1 vs small-N overfitting
    nre_specs = [
        ("N36000_K24", 36000, 24, 256),
        ("N36000_K1", 36000, 1, 256),
        ("N1200_K16", 1200, 16, 64),
    ]
    nre_runs: dict[str, dict[str, object]] = {}
    for key, n_use, k_neg, bsz in nre_specs:
        torch.manual_seed(5300 + n_use + k_neg)
        sub_rng = np.random.default_rng(5300 + n_use + k_neg)
        model = PlinkoNREModel(n_rows=config.n_rows, hidden_dim=96, config=config)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
        tr_loss, va_loss, tv_hist, norm_hist = [], [], [], []
        snaps_logit: dict[int, np.ndarray] = {}

        eval_rng = np.random.default_rng(999)
        for ep in range(max_ep + 1):
            model.eval()
            with torch.no_grad():
                n_ev = min(n_use, 2000)
                l_tr_eval = float(_nre_balanced_bce_loss(model, th_pool[:n_ev], x_pool[:n_ev], 8, eval_rng).item())
                l_va_eval = float(_nre_balanced_bce_loss(model, th_val[:2000], x_val[:2000], 8, eval_rng).item())
                g = evaluate_neural_posterior_grid(model, "NRE", x_obs, n_alpha=na, n_kappa=nk, config=config)
            tr_loss.append(l_tr_eval)
            va_loss.append(l_va_eval)
            tv_hist.append(_tv(g["posterior"]))
            norm_hist.append(float(g["norm_check"]))
            if ep in snap_epochs:
                snaps_logit[ep] = np.clip(g["aux_surface"], -25.0, 25.0).astype(np.float32)
            if ep == max_ep:
                break

            model.train()
            perm = sub_rng.permutation(n_use)
            for s in range(0, n_use, bsz):
                idx = perm[s : s + bsz]
                if idx.size < 32:
                    continue
                loss = _nre_balanced_bce_loss(model, th_pool[idx], x_pool[idx], k_neg, sub_rng)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()

        nre_runs[key] = {
            "n_sim": n_use,
            "k_neg": k_neg,
            "train_loss": np.array(tr_loss, dtype=float),
            "val_loss": np.array(va_loss, dtype=float),
            "tv_hist": np.array(tv_hist, dtype=float),
            "norm_hist": np.array(norm_hist, dtype=float),
            "snaps_logit": snaps_logit,
        }

    out = {
        "alpha_grid": ex_grid["alpha_grid"],
        "kappa_grid": ex_grid["kappa_grid"],
        "da": da,
        "dk": dk,
        "exact_posterior": ex_grid["posterior"],
        "x_obs": x_obs,
        "alpha_true": config.alpha_true,
        "kappa_true": config.kappa_true,
        "snap_epochs": snap_epochs,
        "max_epoch": max_ep,
        "npe": npe_runs,
        "nle": nle_runs,
        "nre": nre_runs,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, p)
    return out
