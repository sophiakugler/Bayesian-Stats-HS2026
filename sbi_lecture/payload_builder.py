"""Export compact precomputed JSON data from `assets/sbi/` for client-side interactive widgets.

Extracts real cached outputs from:
  - `plinko_abc_bank.npz` (Galton/Plinko prior-predictive simulation bank)
  - `neural_sbi_models.pt` (trained PyTorch PlinkoNPEModel, PlinkoNLEModel, PlinkoNREModel)
  - `neural_training_trajectories.pt` (epoch-by-epoch PyTorch NPE, NLE, NRE training snapshots)
  - `gmm_gamma_checkpoints.npz` (Gamma(3, 1/3) -> GMM NLL optimization checkpoints)
  - `regression_mdn_demo.npz` (skewed LogNormal L2, L1, and MDN conditional densities)
  - `validation_coverage.npz` (200 independent unseen Galton-board HPD & SBC validation runs)

Usage:
    python -m sbi_lecture.payload_builder
"""

from __future__ import annotations

import json
import numpy as np
import torch

from .plinko import DEFAULT_CONFIG, exact_posterior_grid, plinko_summary_statistics
from .neural_sbi import build_or_load_training_trajectories, evaluate_neural_posterior_grid
from .precompute import default_assets_dir, load_lecture_bundle
from .validation import compare_posterior_grids


def _r(arr: np.ndarray | list | float, digits: int = 4):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        return round(float(a), digits)
    return np.round(a, digits).tolist()


def build_interactive_payload(mode: str = "full") -> dict[str, object]:
    bundle = load_lecture_bundle(mode=mode)
    cfg = DEFAULT_CONFIG
    ns = bundle["neural_suite"]
    adir = default_assets_dir()

    # 1. Galton-board benchmark observations + 41x41 2D grids for Exact, NPE, NLE, NRE
    na, nk = 41, 41
    obs_list = []
    for idx, obs in enumerate(bundle["observations"]):
        x_obs = np.asarray(obs["x_obs"], dtype=int)
        stats_obs = plinko_summary_statistics(x_obs)

        g_ex = exact_posterior_grid(x_obs, n_alpha=na, n_kappa=nk, config=cfg)
        g_npe = evaluate_neural_posterior_grid(ns["npe"], "NPE", x_obs, n_alpha=na, n_kappa=nk, config=cfg)
        g_nle = evaluate_neural_posterior_grid(ns["nle"], "NLE", x_obs, n_alpha=na, n_kappa=nk, config=cfg)
        g_nre = evaluate_neural_posterior_grid(ns["nre"], "NRE", x_obs, n_alpha=na, n_kappa=nk, config=cfg)

        # High-res 121x121 comparison metrics from bundle["obs_posteriors"]
        hi = bundle["obs_posteriors"][idx]
        metrics = {}
        for m_name in ("NPE", "NLE", "NRE"):
            c = compare_posterior_grids(hi["exact"], hi[m_name])
            metrics[m_name] = {
                "tv": round(c["tv_distance"], 3),
                "kl": round(c["kl_divergence"], 3),
                "mean_a": round(c["mean_alpha"], 3),
                "std_a": round(c["std_alpha"], 3),
                "mean_k": round(c["mean_kappa"], 3),
                "std_k": round(c["std_kappa"], 3),
            }
        c_ex = compare_posterior_grids(hi["exact"], hi["exact"])
        metrics["Exact"] = {
            "tv": 0.0,
            "kl": 0.0,
            "mean_a": round(c_ex["mean_alpha"], 3),
            "std_a": round(c_ex["std_alpha"], 3),
            "mean_k": round(c_ex["mean_kappa"], 3),
            "std_k": round(c_ex["std_kappa"], 3),
        }

        obs_list.append(
            {
                "label": str(obs["label"]),
                "x_obs": x_obs.tolist(),
                "alpha_true": round(float(obs["alpha_true"]), 3),
                "kappa_true": round(float(obs["kappa_true"]), 3),
                "mean_bin": round(float(stats_obs["mean"]), 3),
                "std_bin": round(float(stats_obs["std"]), 3),
                "post": {
                    "Exact": _r(g_ex["posterior"], 4),
                    "NPE": _r(g_npe["posterior"], 4),
                    "NLE": _r(g_nle["posterior"], 4),
                    "NRE": _r(g_nre["posterior"], 4),
                },
                "nle_loglike": _r(g_nle["aux_surface"] - np.max(g_nle["aux_surface"]), 3),
                "nre_logit": _r(g_nre["aux_surface"], 3),
                "metrics": metrics,
            }
        )

    # Also export NLE predicted bin probabilities on a 15x15 (alpha, kappa) grid for live slider inspection
    a_nle = np.linspace(cfg.alpha_min, cfg.alpha_max, 15)
    k_nle = np.linspace(cfg.kappa_min, cfg.kappa_max, 15)
    aa_n, kk_n = np.meshgrid(a_nle, k_nle, indexing="ij")
    pts_n = torch.tensor(np.column_stack([aa_n.ravel(), kk_n.ravel()]), dtype=torch.float32)
    nle_probs = ns["nle"].bin_probs(pts_n).reshape(15, 15, cfg.n_rows + 1)

    # 2. Subbank of 3000 real Galton-board prior-predictive simulations for instant 60-FPS ABC
    bank = bundle["abc_bank"]
    n_sub = 3000
    sub_thetas = np.asarray(bank["thetas"][:n_sub], dtype=float)
    sub_hists = np.asarray(bank["histograms"][:n_sub], dtype=int)
    sub_stats = plinko_summary_statistics(sub_hists)

    # 3. Gamma(3, 1/3) -> GMM training checkpoints (W09)
    gmm_s = bundle["gmm_suite"]
    gmm_out: dict[str, object] = {
        "steps": [int(s) for s in gmm_s["checkpoint_steps"]],
        "runs": {},
    }
    for n_s in gmm_s["sample_sizes"]:
        for k_c in gmm_s["comp_counts"]:
            run = gmm_s["runs"][(n_s, k_c)]
            hist_c, hist_edges = np.histogram(run["samples"], bins=32, range=(0.0, 4.2), density=True)
            hist_x = 0.5 * (hist_edges[:-1] + hist_edges[1:])
            ck_out = {}
            for st, ck in run["checkpoints"].items():
                ck_out[str(st)] = {
                    "w": _r(ck["weights"], 4),
                    "m": _r(ck["means"], 4),
                    "s": _r(ck["stds"], 4),
                    "nll": round(float(ck["nll"]), 4),
                    "kl": round(float(ck["kl"]), 4),
                }
            gmm_out["runs"][f"N{n_s}_K{k_c}"] = {
                "hist_x": _r(hist_x, 3),
                "hist_y": _r(hist_c, 3),
                "nll_hist": _r(run["nll_history"], 3),
                "entropy": round(float(run["empirical_entropy"]), 4),
                "ckpts": ck_out,
            }

    # 4. Skewed LogNormal L1 vs L2 vs MDN (W10)
    reg_s = bundle["reg_suite"]
    x_ev = np.asarray(reg_s["x_eval"])[::2]
    y_ev = np.asarray(reg_s["y_eval"])[::3]
    reg_out: dict[str, object] = {
        "x_eval": _r(x_ev, 3),
        "y_eval": _r(y_ev, 3),
        "by_skew": {},
    }
    for sk in reg_s["skew_levels"]:
        ent = reg_s["by_skew"][sk]
        reg_out["by_skew"][f"{sk:.2f}"] = {
            "xs": _r(ent["x_samples"][:450], 3),
            "ys": _r(ent["y_samples"][:450], 3),
            "l2": _r(ent["l2_pred"][::2], 3),
            "l1": _r(ent["l1_pred"][::2], 3),
            "mode": _r(ent["mdn_mode"][::2], 3),
            "pdf": _r(ent["mdn_cond_pdf"][::2, ::3], 3),
        }

    # 5. Independent 200-experiment Galton-board HPD Coverage & SBC Ranks (W15)
    cov_s = bundle["coverage_suite"]
    cov_out: dict[str, object] = {"n_val": int(cov_s["n_validation"]), "methods": {}}
    for m in cov_s["methods"]:
        cov_out["methods"][m] = {
            "hpd": _r(cov_s["hpd_ranks"][m], 4),
            "pit_a": _r(cov_s["pit_alpha"][m], 4),
            "pit_k": _r(cov_s["pit_kappa"][m], 4),
        }

    # 6. Epoch-by-epoch PyTorch training trajectories for Galton-board NPE, NLE, NRE train widgets
    tr_raw = build_or_load_training_trajectories(adir / "neural_training_trajectories.pt", config=cfg)
    traj_out: dict[str, object] = {
        "alpha": _r(tr_raw["alpha_grid"], 4),
        "kappa": _r(tr_raw["kappa_grid"], 4),
        "da": round(float(tr_raw["da"]), 5),
        "dk": round(float(tr_raw["dk"]), 5),
        "exact": _r(tr_raw["exact_posterior"], 4),
        "x_obs": np.asarray(tr_raw["x_obs"], dtype=int).tolist(),
        "alpha_true": round(float(tr_raw["alpha_true"]), 3),
        "kappa_true": round(float(tr_raw["kappa_true"]), 3),
        "snap_epochs": [int(e) for e in tr_raw["snap_epochs"]],
        "max_epoch": int(tr_raw["max_epoch"]),
        "npe": {},
        "nle": {},
        "nre": {},
    }
    for k, r in tr_raw["npe"].items():
        traj_out["npe"][k] = {
            "n_sim": int(r["n_sim"]),
            "k_comp": int(r["k_comp"]),
            "train": _r(r["train_loss"], 3),
            "val": _r(r["val_loss"], 3),
            "tv": _r(r["tv_hist"], 3),
            "snaps": {str(ep): _r(arr, 3) for ep, arr in r["snaps"].items()},
        }
    for k, r in tr_raw["nle"].items():
        traj_out["nle"][k] = {
            "n_sim": int(r["n_sim"]),
            "train": _r(r["train_loss"], 3),
            "val": _r(r["val_loss"], 3),
            "tv": _r(r["tv_hist"], 3),
            "snaps_ll": {str(ep): _r(arr, 2) for ep, arr in r["snaps_loglike"].items()},
            "snaps_bins": {str(ep): _r(arr, 4) for ep, arr in r["snaps_bins"].items()},
        }
    for k, r in tr_raw["nre"].items():
        traj_out["nre"][k] = {
            "n_sim": int(r["n_sim"]),
            "k_neg": int(r["k_neg"]),
            "train": _r(r["train_loss"], 4),
            "val": _r(r["val_loss"], 4),
            "tv": _r(r["tv_hist"], 3),
            "norm": _r(r["norm_hist"], 3),
            "snaps_logit": {str(ep): _r(arr, 2) for ep, arr in r["snaps_logit"].items()},
        }

    d_a = (cfg.alpha_max - cfg.alpha_min) / na
    d_k = (cfg.kappa_max - cfg.kappa_min) / nk
    a_grid = cfg.alpha_min + (np.arange(na) + 0.5) * d_a
    k_grid = cfg.kappa_min + (np.arange(nk) + 0.5) * d_k

    return {
        "grid": {
            "alpha": _r(a_grid, 4),
            "kappa": _r(k_grid, 4),
            "da": round(d_a, 5),
            "dk": round(d_k, 5),
            "alpha_min": cfg.alpha_min,
            "alpha_max": cfg.alpha_max,
            "kappa_min": cfg.kappa_min,
            "kappa_max": cfg.kappa_max,
            "n_rows": cfg.n_rows,
            "n_balls": cfg.n_balls,
        },
        "observations": obs_list,
        "nle_bin_grid": {
            "alpha": _r(a_nle, 3),
            "kappa": _r(k_nle, 3),
            "probs": _r(nle_probs, 4),
        },
        "abc_bank": {
            "alphas": _r(sub_thetas[:, 0], 3),
            "kappas": _r(sub_thetas[:, 1], 3),
            "hists": sub_hists.tolist(),
            "means": _r(sub_stats["mean"], 3),
            "vars": _r(sub_stats["var"], 3),
        },
        "gmm_gamma": gmm_out,
        "reg_mdn": reg_out,
        "coverage": cov_out,
        "train_traj": traj_out,
    }


def main() -> None:
    payload = build_interactive_payload()
    out_path = default_assets_dir() / "interactive_payload.json"
    txt = json.dumps(payload, separators=(",", ":"))
    out_path.write_text(txt, encoding="utf-8")
    print(f"Wrote {out_path} ({len(txt) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
