"""Density estimation, Gaussian mixture models, and loss-function demonstrations.

Preserves and extends the original lecture's core mathematical demonstrations:
1. 1D Gaussian mixture construction and interactive editing (Slide 21, W08).
2. Fitting a 5-component GMM to Gamma(a=3, scale=1/3) samples via empirical NLL /
   forward-KL minimization with saved optimization checkpoints (Slide 22, W09).
3. Skewed LogNormal regression comparing L2 (conditional mean), L1 (conditional median),
   and Mixture Density Network NLL (full conditional distribution & mode) (Slides 23-24, W10, Backup).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import gamma as scipy_gamma
from scipy.stats import lognorm, norm
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. 1D Gaussian Mixture Model (W08 & W09)
# -----------------------------------------------------------------------------

@dataclass
class GaussianMixture1D:
    """Normalized 1D Gaussian Mixture Model with strictly positive component scales."""

    weights: np.ndarray
    means: np.ndarray
    stds: np.ndarray

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float).ravel()
        m = np.asarray(self.means, dtype=float).ravel()
        s = np.asarray(self.stds, dtype=float).ravel()
        if not (w.size == m.size == s.size and w.size >= 1):
            raise ValueError("weights, means, and stds must be non-empty 1D arrays of equal length")
        if np.any(w < 0) or float(np.sum(w)) <= 0:
            raise ValueError("Mixture weights must be non-negative with positive sum")
        if np.any(s <= 0):
            raise ValueError("Component standard deviations must be strictly positive")
        self.weights = w / float(np.sum(w))
        self.means = m
        self.stds = s

    @classmethod
    def from_unconstrained(
        cls,
        logits: np.ndarray,
        means: np.ndarray,
        log_stds: np.ndarray,
        min_std: float = 0.03,
    ) -> "GaussianMixture1D":
        weights = softmax(np.asarray(logits, dtype=float).ravel())
        stds = np.maximum(np.exp(np.asarray(log_stds, dtype=float).ravel()), min_std)
        return cls(weights=weights, means=means, stds=stds)

    @property
    def n_components(self) -> int:
        return int(self.weights.size)

    def component_pdfs(self, x: np.ndarray, weighted: bool = True) -> np.ndarray:
        """Evaluate component densities at `x`, returning shape `(K, len(x))`."""
        x_arr = np.asarray(x, dtype=float)[None, :]
        comp = norm.pdf(x_arr, loc=self.means[:, None], scale=self.stds[:, None])
        if weighted:
            return self.weights[:, None] * comp
        return comp

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        log_w = np.log(np.maximum(self.weights, 1e-300))[:, None]
        log_comp = norm.logpdf(x_arr[None, :], loc=self.means[:, None], scale=self.stds[:, None])
        return logsumexp(log_w + log_comp, axis=0)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(x))

    def sample(self, n_samples: int, seed: int | np.random.Generator | None = 42) -> np.ndarray:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        comp_idx = rng.choice(self.n_components, size=n_samples, p=self.weights)
        return rng.normal(loc=self.means[comp_idx], scale=self.stds[comp_idx])


def gamma_target_distribution(shape: float = 3.0, scale: float = 1.0 / 3.0):
    """Return the original lecture's Gamma(a=3, scale=1/3) target distribution."""
    return scipy_gamma(a=shape, scale=scale)


def estimate_forward_kl_1d(
    target_dist,
    approx_gmm: GaussianMixture1D,
    x_min: float = 1e-4,
    x_max: float = 6.0,
    n_grid: int = 2000,
) -> float:
    """Numerically evaluate D_KL(p || q_phi) = int p(x) log(p(x) / q_phi(x)) dx."""
    xg = np.linspace(x_min, x_max, n_grid)
    p = target_dist.pdf(xg)
    mask = p > 1e-15
    log_p = np.log(p[mask])
    log_q = approx_gmm.log_prob(xg[mask])
    integrand = np.zeros_like(xg)
    integrand[mask] = p[mask] * (log_p - log_q)
    return float(max(np.trapezoid(integrand, xg), 0.0))


def fit_gmm_to_gamma_with_checkpoints(
    n_samples: int = 1000,
    n_components: int = 5,
    n_steps: int = 160,
    lr: float = 0.06,
    seed: int = 42,
) -> dict[str, object]:
    """Fit a K-component GMM to Gamma(3, 1/3) samples and record optimization trajectory."""
    rng = np.random.default_rng(seed)
    target = gamma_target_distribution()
    samples = target.rvs(size=n_samples, random_state=rng).astype(np.float32)

    torch.manual_seed(seed)
    logits = nn.Parameter(torch.zeros(n_components, dtype=torch.float32))
    # Deterministic spread initialization similar to standardized random init
    init_means = np.linspace(0.2, 2.4, n_components) + rng.normal(0.0, 0.15, size=n_components)
    means = nn.Parameter(torch.tensor(init_means, dtype=torch.float32))
    log_stds = nn.Parameter(torch.full((n_components,), -0.5, dtype=torch.float32))

    x_tensor = torch.tensor(samples, dtype=torch.float32)[:, None]  # (N, 1)
    optimizer = torch.optim.Adam([logits, means, log_stds], lr=lr)

    checkpoint_steps = [0, 2, 5, 10, 20, 40, 80, 120, n_steps]
    checkpoints: dict[int, dict[str, np.ndarray | float]] = {}
    nll_history = []
    kl_history = []

    for step in range(n_steps + 1):
        log_w = F.log_softmax(logits, dim=0)[None, :]
        stds = torch.clamp(torch.exp(log_stds), min=0.04)[None, :]
        log_comp = -0.5 * (((x_tensor - means[None, :]) / stds) ** 2) - torch.log(stds) - 0.5 * np.log(2.0 * np.pi)
        log_prob = torch.logsumexp(log_w + log_comp, dim=1)
        nll = -torch.mean(log_prob)

        nll_val = float(nll.detach().cpu().item())
        nll_history.append(nll_val)

        if step in checkpoint_steps:
            gmm_curr = GaussianMixture1D(
                weights=F.softmax(logits.detach(), dim=0).cpu().numpy(),
                means=means.detach().cpu().numpy().copy(),
                stds=torch.clamp(torch.exp(log_stds.detach()), min=0.04).cpu().numpy(),
            )
            kl_val = estimate_forward_kl_1d(target, gmm_curr)
            checkpoints[step] = {
                "weights": gmm_curr.weights,
                "means": gmm_curr.means,
                "stds": gmm_curr.stds,
                "nll": nll_val,
                "kl": kl_val,
            }
            kl_history.append(kl_val)

        if step < n_steps:
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

    target_entropy = -float(np.mean(target.logpdf(samples)))
    return {
        "samples": samples.astype(float),
        "n_samples": n_samples,
        "n_components": n_components,
        "checkpoint_steps": np.array(checkpoint_steps, dtype=int),
        "checkpoints": checkpoints,
        "nll_history": np.array(nll_history, dtype=float),
        "empirical_entropy": target_entropy,
    }


def load_or_build_gmm_gamma_suite(cache_path: str | Path | None = None) -> dict[str, object]:
    """Load precomputed Gamma-to-GMM optimization suite across sample sizes and component counts."""
    sample_sizes = (150, 1000, 4000)
    comp_counts = (1, 3, 5, 8)
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            raw = np.load(p, allow_pickle=False)
            suite: dict[str, object] = {
                "sample_sizes": tuple(int(v) for v in raw["sample_sizes"]),
                "comp_counts": tuple(int(v) for v in raw["comp_counts"]),
                "checkpoint_steps": raw["checkpoint_steps"],
                "runs": {},
            }
            for n_s in suite["sample_sizes"]:
                for k_c in suite["comp_counts"]:
                    key = f"N{n_s}_K{k_c}"
                    ckpts = {}
                    for step in suite["checkpoint_steps"]:
                        st = int(step)
                        ckpts[st] = {
                            "weights": raw[f"{key}_s{st}_w"],
                            "means": raw[f"{key}_s{st}_m"],
                            "stds": raw[f"{key}_s{st}_s"],
                            "nll": float(raw[f"{key}_s{st}_nll"]),
                            "kl": float(raw[f"{key}_s{st}_kl"]),
                        }
                    suite["runs"][(n_s, k_c)] = {
                        "samples": raw[f"{key}_samples"],
                        "nll_history": raw[f"{key}_nll_hist"],
                        "empirical_entropy": float(raw[f"{key}_ent"]),
                        "checkpoints": ckpts,
                    }
            return suite

    runs = {}
    flat_save: dict[str, np.ndarray] = {
        "sample_sizes": np.array(sample_sizes, dtype=int),
        "comp_counts": np.array(comp_counts, dtype=int),
    }
    checkpoint_steps = None
    for idx_n, n_s in enumerate(sample_sizes):
        for idx_k, k_c in enumerate(comp_counts):
            res = fit_gmm_to_gamma_with_checkpoints(
                n_samples=n_s,
                n_components=k_c,
                n_steps=160,
                seed=42 + idx_n * 10 + idx_k,
            )
            if checkpoint_steps is None:
                checkpoint_steps = res["checkpoint_steps"]
                flat_save["checkpoint_steps"] = checkpoint_steps
            runs[(n_s, k_c)] = res
            key = f"N{n_s}_K{k_c}"
            flat_save[f"{key}_samples"] = res["samples"]
            flat_save[f"{key}_nll_hist"] = res["nll_history"]
            flat_save[f"{key}_ent"] = np.array(res["empirical_entropy"])
            for st, ck in res["checkpoints"].items():
                flat_save[f"{key}_s{st}_w"] = ck["weights"]
                flat_save[f"{key}_s{st}_m"] = ck["means"]
                flat_save[f"{key}_s{st}_s"] = ck["stds"]
                flat_save[f"{key}_s{st}_nll"] = np.array(ck["nll"])
                flat_save[f"{key}_s{st}_kl"] = np.array(ck["kl"])

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **flat_save)

    return {
        "sample_sizes": sample_sizes,
        "comp_counts": comp_counts,
        "checkpoint_steps": checkpoint_steps,
        "runs": runs,
    }


# -----------------------------------------------------------------------------
# 2. Skewed Conditional Regression (L1 vs L2) & Conditional MDN (W10 & Backup)
# -----------------------------------------------------------------------------

def skewed_conditional_distribution(x: np.ndarray | float, skew_scale: float = 1.0):
    """Original notebook's skewed conditional law y | x ~ LogNormal(s = skew_scale * x, loc = 2x)."""
    x_arr = np.asarray(x, dtype=float)
    s = np.maximum(skew_scale * x_arr, 1e-4)
    return lognorm(s=s, loc=2.0 * x_arr, scale=1.0)


def skewed_conditional_summaries(
    x: np.ndarray | float,
    skew_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Exact theoretical mean (L2 optimum), median (L1 optimum), and mode of p(y | x)."""
    x_arr = np.asarray(x, dtype=float)
    s = skew_scale * x_arr
    mean = 2.0 * x_arr + np.exp(0.5 * (s ** 2))
    median = 2.0 * x_arr + np.ones_like(x_arr)
    mode = 2.0 * x_arr + np.exp(-(s ** 2))
    return {"mean": mean, "median": median, "mode": mode}


def sample_skewed_regression_data(
    n_samples: int = 1200,
    skew_scale: float = 1.0,
    x_min: float = 0.1,
    x_max: float = 1.5,
    seed: int | np.random.Generator | None = 42,
) -> dict[str, np.ndarray]:
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=n_samples)
    y = skewed_conditional_distribution(x, skew_scale=skew_scale).rvs(random_state=rng)
    return {"x": x, "y": y}


class _MLP1D(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MDN1D(nn.Module):
    """1D Mixture Density Network mirroring Cell 072 of the original notebook."""

    def __init__(self, hidden: int = 96, n_components: int = 8):
        super().__init__()
        self.n_components = n_components
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.logits_head = nn.Linear(hidden, n_components)
        self.means_head = nn.Linear(hidden, n_components)
        self.scales_head = nn.Linear(hidden, n_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.logits_head(h)
        means = self.means_head(h)
        stds = F.softplus(self.scales_head(h)) + 0.04
        return logits, means, stds

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, means, stds = self.forward(x)
        log_w = F.log_softmax(logits, dim=-1)
        y_col = y.reshape(-1, 1)
        log_comp = (
            -0.5 * (((y_col - means) / stds) ** 2)
            - torch.log(stds)
            - 0.5 * np.log(2.0 * np.pi)
        )
        return torch.logsumexp(log_w + log_comp, dim=-1)


def train_or_load_regression_mdn_suite(
    cache_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Train or load cached L2, L1, and MDN models across skewness settings for W10 & Backup."""
    skew_levels = (0.6, 1.0, 1.25)
    x_eval = np.linspace(0.1, 1.5, 80)
    y_eval = np.linspace(0.0, 12.0, 240)

    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            raw = np.load(p, allow_pickle=False)
            out: dict[str, object] = {
                "skew_levels": tuple(float(v) for v in raw["skew_levels"]),
                "x_eval": raw["x_eval"],
                "y_eval": raw["y_eval"],
                "by_skew": {},
            }
            for sk in out["skew_levels"]:
                tag = f"sk{int(round(sk * 100))}"
                out["by_skew"][sk] = {
                    "x_samples": raw[f"{tag}_xs"],
                    "y_samples": raw[f"{tag}_ys"],
                    "l2_pred": raw[f"{tag}_l2"],
                    "l1_pred": raw[f"{tag}_l1"],
                    "mdn_mode": raw[f"{tag}_mdn_mode"],
                    "mdn_cond_pdf": raw[f"{tag}_mdn_pdf"],  # shape (len(x_eval), len(y_eval))
                }
            return out

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_eval_t = torch.tensor(x_eval[:, None], dtype=torch.float32)

    flat_save: dict[str, np.ndarray] = {
        "skew_levels": np.array(skew_levels, dtype=float),
        "x_eval": x_eval,
        "y_eval": y_eval,
    }
    by_skew = {}

    for sk in skew_levels:
        vis_data = sample_skewed_regression_data(n_samples=1200, skew_scale=sk, seed=rng)

        # Train L2 regressor
        m_l2 = _MLP1D()
        opt_l2 = torch.optim.Adam(m_l2.parameters(), lr=2e-3)
        for _ in range(700):
            bx = rng.uniform(0.1, 1.5, size=(512, 1)).astype(np.float32)
            by = skewed_conditional_distribution(bx, skew_scale=sk).rvs(random_state=rng).astype(np.float32)
            pred = m_l2(torch.from_numpy(bx))
            loss = torch.mean((pred - torch.from_numpy(by)) ** 2)
            opt_l2.zero_grad()
            loss.backward()
            opt_l2.step()

        # Train L1 regressor
        m_l1 = _MLP1D()
        opt_l1 = torch.optim.Adam(m_l1.parameters(), lr=2e-3)
        for _ in range(700):
            bx = rng.uniform(0.1, 1.5, size=(512, 1)).astype(np.float32)
            by = skewed_conditional_distribution(bx, skew_scale=sk).rvs(random_state=rng).astype(np.float32)
            pred = m_l1(torch.from_numpy(bx))
            loss = torch.mean(torch.abs(pred - torch.from_numpy(by)))
            opt_l1.zero_grad()
            loss.backward()
            opt_l1.step()

        # Train MDN via NLL (forward KL)
        m_mdn = _MDN1D(hidden=96, n_components=10)
        opt_mdn = torch.optim.Adam(m_mdn.parameters(), lr=2e-3)
        for _ in range(900):
            bx = rng.uniform(0.1, 1.5, size=(512, 1)).astype(np.float32)
            by = skewed_conditional_distribution(bx, skew_scale=sk).rvs(random_state=rng).astype(np.float32)
            loss = -torch.mean(m_mdn.log_prob(torch.from_numpy(bx), torch.from_numpy(by)))
            opt_mdn.zero_grad()
            loss.backward()
            opt_mdn.step()

        with torch.no_grad():
            l2_pred = m_l2(x_eval_t).squeeze(-1).cpu().numpy()
            l1_pred = m_l1(x_eval_t).squeeze(-1).cpu().numpy()
            logits, means, stds = m_mdn(x_eval_t)
            w_np = F.softmax(logits, dim=-1).cpu().numpy()
            m_np = means.cpu().numpy()
            s_np = stds.cpu().numpy()

        # Evaluate conditional density on (x_eval, y_eval)
        cond_pdf = np.zeros((x_eval.size, y_eval.size), dtype=float)
        for i in range(x_eval.size):
            gmm_i = GaussianMixture1D(weights=w_np[i], means=m_np[i], stds=s_np[i])
            cond_pdf[i] = gmm_i.pdf(y_eval)
        mdn_mode = y_eval[np.argmax(cond_pdf, axis=1)]

        tag = f"sk{int(round(sk * 100))}"
        entry = {
            "x_samples": vis_data["x"],
            "y_samples": vis_data["y"],
            "l2_pred": l2_pred,
            "l1_pred": l1_pred,
            "mdn_mode": mdn_mode,
            "mdn_cond_pdf": cond_pdf,
        }
        by_skew[sk] = entry
        flat_save[f"{tag}_xs"] = entry["x_samples"]
        flat_save[f"{tag}_ys"] = entry["y_samples"]
        flat_save[f"{tag}_l2"] = entry["l2_pred"]
        flat_save[f"{tag}_l1"] = entry["l1_pred"]
        flat_save[f"{tag}_mdn_mode"] = entry["mdn_mode"]
        flat_save[f"{tag}_mdn_pdf"] = entry["mdn_cond_pdf"]

    if cache_path is not None:
        p = Path(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **flat_save)

    return {
        "skew_levels": skew_levels,
        "x_eval": x_eval,
        "y_eval": y_eval,
        "by_skew": by_skew,
    }

