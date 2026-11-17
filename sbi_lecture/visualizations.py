"""Interactive widgets (W01-W15) and publication-quality lecture figures for Week 10 SBI.

Every interactive widget returns a `LectureWidget` object that:
1. Renders live interactive `ipywidgets` controls + responsive Matplotlib plots in a live Jupyter kernel.
2. Embeds a crisp high-DPI `image/png` fallback in its MIME bundle (and supports `SBI_RENDER_MODE=static`)
   so exported Reveal.js slides (`make_slides.sh`), PDFs (`make_pdf.sh`), and static notebook viewers
   always display publication-quality figures even without a Python widget backend.
3. Exposes `.controls`, `.reset()`, `.render_current_figure()`, and `.last_metrics` for automated testing.
"""

from __future__ import annotations

import io
import os
from typing import Callable

import ipywidgets as widgets
from IPython.display import Image, display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.special import expit, logit
from scipy.stats import binom, norm

from .abc import (
    DISTANCE_LABELS,
    SUMMARY_LABELS,
    chi2_normal_abc_analytic_posterior,
    chi2_normal_exact_posterior,
    chi2_normal_joint_grid,
    correlated_gaussian_exact_conditional,
    run_1d_rejection_abc,
    run_plinko_abc,
    simulate_chi2_normal_joint,
    simulate_correlated_gaussian_joint,
)
from .density_estimation import (
    GaussianMixture1D,
    gamma_target_distribution,
    skewed_conditional_distribution,
    skewed_conditional_summaries,
)
from .plinko import (
    DEFAULT_CONFIG,
    binomial_bin_probabilities,
    exact_bin_probabilities,
    plinko_summary_statistics,
    sample_from_grid_posterior,
    simulate_plinko_histograms,
    simulate_trajectories,
)
from .precompute import load_lecture_bundle
from .validation import (
    compare_posterior_grids,
    run_posterior_predictive_check,
    summarize_empirical_coverage,
)


PALETTE = {
    "obs": "#1f2937",
    "exact": "#111827",
    "truth": "#dc2626",
    "rejected": "#9ca3af",
    "abc": "#d97706",
    "NPE": "#2563eb",
    "NLE": "#059669",
    "NRE": "#7c3aed",
    "prior": "#6b7280",
    "accent": "#0284c7",
}


def apply_lecture_style() -> None:
    """Configure projection-friendly light Matplotlib defaults harmonized with course_tools."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11.0,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.8,
            "legend.frameon": False,
            "axes.edgecolor": "#4b5563",
            "axes.labelcolor": "#1f2937",
            "text.color": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 160) -> bytes:
    """Render a Matplotlib figure to PNG bytes and close it cleanly to prevent memory leaks."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


class LectureWidget:
    """Dual-mode interactive widget with deterministic reset and static slide export fallback."""

    def __init__(
        self,
        name: str,
        controls: dict[str, widgets.Widget],
        layout_box: widgets.Widget,
        render_fn: Callable[[], plt.Figure],
        reset_fn: Callable[[], None],
        dpi: int = 155,
    ):
        self.name = name
        self.controls = controls
        self.output = widgets.Output()
        self.render_fn = render_fn
        self.reset_fn = reset_fn
        self.dpi = dpi
        self.render_count = 0
        self.last_png: bytes = b""
        self.last_metrics: dict[str, object] = {}
        self._updating = False

        self.ui = widgets.VBox(
            [layout_box, self.output],
            layout=widgets.Layout(width="100%", padding="4px 0px"),
        )
        self.refresh()

    def refresh(self) -> None:
        if self._updating:
            return
        apply_lecture_style()
        rendered = self.render_fn()
        if isinstance(rendered, tuple):
            fig, metrics = rendered
            self.last_metrics = metrics
        else:
            fig = rendered
        png = fig_to_png_bytes(fig, dpi=self.dpi)
        self.last_png = png
        self.render_count += 1
        self.output.clear_output(wait=True)
        with self.output:
            display(Image(data=png))

    def reset(self) -> None:
        self._updating = True
        try:
            self.reset_fn()
        finally:
            self._updating = False
        self.refresh()

    def _repr_mimebundle_(self, include=None, exclude=None):
        mode = os.environ.get("SBI_RENDER_MODE", "dual").lower()
        if mode == "static":
            return {
                "image/png": self.last_png,
                "text/plain": f"<LectureWidget {self.name}>",
            }
        bundle = self.ui._repr_mimebundle_(include=include, exclude=exclude)
        if isinstance(bundle, tuple):
            data, metadata = bundle
        else:
            data, metadata = dict(bundle), {}
        # Embed default high-DPI PNG right in the MIME bundle so nbconvert --to slides / webpdf / GitHub
        # automatically falls back to the crisp rendered plot whenever widget state is absent!
        data["image/png"] = self.last_png
        data["text/plain"] = f"<LectureWidget {self.name}>"
        return data, metadata


def _make_reset_button(desc: str = "Reset") -> widgets.Button:
    return widgets.Button(
        description=desc,
        icon="refresh",
        button_style="",
        layout=widgets.Layout(width="96px"),
    )


# =============================================================================
# Static Pedagogical Illustrations (Slides 1, 4, 8, 9, 31 & Backup)
# =============================================================================

def _draw_plinko_board_on_ax(
    ax: plt.Axes,
    n_rows: int = 12,
    s_paths: np.ndarray | None = None,
    highlight_idx: int = 0,
    alpha: float | None = None,
    kappa: float | None = None,
    show_field: bool = False,
) -> None:
    """Render triangular Plinko peg board in (s_t, -t) coordinates with optional ball paths."""
    sqrt_r = np.sqrt(float(n_rows))
    peg_x, peg_y, peg_c = [], [], []
    for t in range(n_rows):
        for k in range(t + 1):
            s_t = 2 * k - t
            peg_x.append(s_t)
            peg_y.append(-t)
            if show_field and alpha is not None and kappa is not None:
                peg_c.append(float(expit(alpha - kappa * (s_t / sqrt_r))))
            else:
                peg_c.append(0.5)

    if show_field and alpha is not None and kappa is not None:
        sc = ax.scatter(
            peg_x,
            peg_y,
            c=peg_c,
            cmap="coolwarm",
            vmin=0.15,
            vmax=0.85,
            s=46,
            edgecolors="#374151",
            linewidths=0.7,
            zorder=4,
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
        cbar.set_label(r"$P(\mathrm{right}\mid s_t)$", fontsize=9.5)
    else:
        ax.scatter(
            peg_x,
            peg_y,
            color="#4b5563",
            s=36,
            zorder=4,
        )

    # Draw bin dividers at row -R
    bin_s = [2 * k - n_rows for k in range(n_rows + 1)]
    for bx in np.linspace(-n_rows - 1, n_rows + 1, n_rows + 2):
        ax.plot([bx, bx], [-n_rows + 0.25, -n_rows - 0.85], color="#9ca3af", lw=1.1, zorder=2)
    for k, bs in enumerate(bin_s):
        ax.text(bs, -n_rows - 0.65, str(k), ha="center", va="center", fontsize=8.5, color="#374151")

    if s_paths is not None and s_paths.shape[0] > 0:
        ys = -np.arange(n_rows + 1)
        n_show = min(s_paths.shape[0], 45)
        rng = np.random.default_rng(123)
        for idx in range(n_show):
            if idx == highlight_idx:
                continue
            jit = rng.normal(0.0, 0.08, size=ys.size)
            jit[0] = 0.0
            ax.plot(
                s_paths[idx] + jit,
                ys,
                color=PALETTE["NPE"],
                alpha=0.18,
                lw=1.1,
                zorder=3,
            )
        if 0 <= highlight_idx < s_paths.shape[0]:
            ax.plot(
                s_paths[highlight_idx],
                ys,
                color=PALETTE["truth"],
                alpha=0.95,
                lw=2.4,
                marker="o",
                markersize=3.8,
                zorder=5,
                label="Highlighted ball path",
            )

    ax.set_xlim(-n_rows - 1.6, n_rows + 1.6)
    ax.set_ylim(-n_rows - 1.2, 0.9)
    ax.set_yticks([0, -n_rows // 2, -n_rows])
    ax.set_yticklabels(["Row 0 (drop)", f"Row {n_rows // 2}", f"Bins (R={n_rows})"])
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def show_slide01_banner(bundle: dict | None = None) -> Image:
    """Clean Galton-board mystery banner for Title Slide 1."""
    b = bundle or load_lecture_bundle()
    obs = b["primary_obs"]
    apply_lecture_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    _draw_plinko_board_on_ax(ax1, n_rows=12, s_paths=obs["s_paths"][:35], highlight_idx=0)
    ax1.set_title("Stochastic Forward Simulator (R = 12 rows)")

    ks = np.arange(13)
    ax2.bar(ks, obs["x_obs"], color=PALETTE["obs"], alpha=0.88, width=0.76, label="Observed counts (N = 100)")
    ax2.set_xlabel("Landing bin $k$")
    ax2.set_ylabel("Ball count $n_k$")
    ax2.set_xticks(ks)
    ax2.set_title(r"Observed Outcome $x_{\mathrm{obs}}$ — Unknown $(\alpha, \kappa) = (?, ?)$")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return Image(data=fig_to_png_bytes(fig))


def show_slide04_timeline() -> Image:
    """Illustrated historical timeline connecting inverse probability, Monte Carlo, ABC, and neural SBI."""
    apply_lecture_style()
    fig, ax = plt.subplots(figsize=(10.4, 3.1))
    ax.axhline(0.0, color="#374151", lw=2.4, zorder=1)

    milestones = [
        (0.0, "1763 — Bayes & Laplace", "Inverse probability:\ninferring hidden causes\nfrom observed counts", "#1f2937", 0.55),
        (1.0, "1940s — Monte Carlo", "Ulam & von Neumann:\nforward stochastic simulation\non early computers", "#0284c7", -0.58),
        (2.0, "1997–2002 — ABC", "Tavaré, Pritchard, Beaumont:\nrejection inference via\nsimulated summaries", "#d97706", 0.55),
        (3.0, "2016+ — Neural SBI", "NPE, NLE, NRE:\nlearning densities & ratios\nfrom all simulations", "#2563eb", -0.58),
    ]
    for x_pos, title, desc, col, y_off in milestones:
        ax.scatter([x_pos], [0.0], s=130, color=col, zorder=4, edgecolors="white", linewidths=1.5)
        ax.plot([x_pos, x_pos], [0.0, y_off * 0.62], color=col, lw=1.6, ls="--", zorder=2)
        va = "bottom" if y_off > 0 else "top"
        ax.text(
            x_pos,
            y_off * 0.66,
            f"{title}\n{desc}",
            ha="center",
            va=va,
            fontsize=10.0,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#f9fafb", edgecolor=col, lw=1.4),
        )

    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    fig.tight_layout()
    return Image(data=fig_to_png_bytes(fig))


def show_slide08_marginalization(bundle: dict | None = None) -> Image:
    """Illustrate one simulated path vs combinatorial path marginalization into bin k = 7."""
    b = bundle or load_lecture_bundle()
    obs = b["primary_obs"]
    apply_lecture_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))

    # Find paths landing in bin 7
    k_paths = obs["k_paths"]
    s_paths = obs["s_paths"]
    idx_bin7 = np.where(k_paths[:, -1] == 7)[0]
    one_idx = int(idx_bin7[0]) if idx_bin7.size > 0 else 0

    _draw_plinko_board_on_ax(ax1, n_rows=12, s_paths=s_paths[one_idx : one_idx + 1], highlight_idx=0)
    ax1.set_title("Forward Step: Simulate 1 Trajectory $z \\sim p(z\\mid\\theta)$\n(12 coin flips — trivial!)")

    # Generate many distinct trajectories ending in bin 7 to show combinatorial paths
    sim_many = simulate_trajectories(0.6, 1.2, n_balls=400, n_rows=12, seed=77)
    paths_7 = sim_many["s_paths"][sim_many["landing_bins"] == 7][:60]
    _draw_plinko_board_on_ax(ax2, n_rows=12, s_paths=paths_7, highlight_idx=0)
    ax2.set_title(
        r"Likelihood Evaluation: Sum All $\binom{12}{7}=792$ Paths to Bin 7"
        "\n"
        r"$p(K=7\mid\theta) = \sum_{z \to 7} \prod_{t=0}^{11} p(z_{t+1}\mid z_t, \theta)$"
    )
    fig.tight_layout()
    return Image(data=fig_to_png_bytes(fig))


def show_slide09_challenge(bundle: dict | None = None) -> Image:
    """Compare mystery observation against three plausible parameter candidates under finite-N=100 noise."""
    b = bundle or load_lecture_bundle()
    x_obs = b["primary_obs"]["x_obs"]
    candidates = [
        ("Candidate A: α = +0.25, κ = 0.0 (Pure Binomial)", 0.25, 0.0, "#0284c7", 901),
        ("Candidate B: α = +0.60, κ = 1.2 (Restoring Bias)", 0.60, 1.2, "#059669", 902),
        ("Candidate C: α = +0.95, κ = 2.2 (Strong Pull + Bias)", 0.95, 2.2, "#7c3aed", 903),
    ]
    apply_lecture_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharey=True)
    ks = np.arange(13)
    for ax, (title, a_val, k_val, col, sd) in zip(axes, candidates):
        sim_h = simulate_plinko_histograms(a_val, k_val, n_balls=100, n_rows=12, seed=sd)
        ax.bar(ks - 0.18, x_obs, width=0.36, color=PALETTE["obs"], alpha=0.85, label="Observed $x_{\\mathrm{obs}}$")
        ax.bar(ks + 0.18, sim_h, width=0.36, color=col, alpha=0.82, label="100-ball simulation")
        st_sim = plinko_summary_statistics(sim_h)
        st_obs = plinko_summary_statistics(x_obs)
        ax.set_title(f"{title}\nSim mean={st_sim['mean']:.2f}, std={st_sim['std']:.2f}", fontsize=10.2)
        ax.set_xlabel("Landing bin $k$")
        ax.set_xticks(range(0, 13, 2))
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Count (out of 100 balls)")
    fig.suptitle(
        f"Mystery Observation has mean = {st_obs['mean']:.2f}, std = {st_obs['std']:.2f} — Which parameters generated it?",
        fontsize=11.8,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    return Image(data=fig_to_png_bytes(fig))


def show_slide31_truth_reveal(bundle: dict | None = None) -> Image:
    """Truth reveal comparing Exact DP, ABC, NPE, NLE, and NRE posteriors against true (alpha, kappa)."""
    b = bundle or load_lecture_bundle()
    obs = b["primary_obs"]
    posts = b["obs_posteriors"][0]
    abc_res = run_plinko_abc(b["abc_bank"], obs["x_obs"], epsilon=0.30, distance_kind="wasserstein")
    abc_grid = abc_res["kde_grid"]

    apply_lecture_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0]})

    methods = [
        ("Exact DP", posts["exact"], PALETTE["exact"], "-", 2.2),
        ("NPE", posts["NPE"], PALETTE["NPE"], "--", 1.9),
        ("NLE", posts["NLE"], PALETTE["NLE"], "-.", 1.9),
        ("NRE", posts["NRE"], PALETTE["NRE"], ":", 2.1),
        ("ABC (ε=0.30)", abc_grid, PALETTE["abc"], "-", 1.5),
    ]
    for label, g, col, ls, lw in methods:
        aa, kk = np.meshgrid(g["alpha_grid"], g["kappa_grid"], indexing="ij")
        # Plot 68% HPD contour (highest threshold)
        ax1.contour(aa, kk, g["posterior"], levels=[g["hpd_levels"][-1]], colors=[col], linestyles=[ls], linewidths=lw)
        ax1.plot([], [], color=col, ls=ls, lw=lw, label=f"{label} (68% HPD)")

    ax1.scatter(
        [obs["alpha_true"]],
        [obs["kappa_true"]],
        marker="*",
        s=240,
        color=PALETTE["truth"],
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
        label=f"True settings ($\\alpha={obs['alpha_true']}, \\kappa={obs['kappa_true']}$)",
    )
    ax1.scatter(
        [posts["exact"]["map_alpha"]],
        [posts["exact"]["map_kappa"]],
        marker="x",
        s=90,
        color=PALETTE["exact"],
        linewidths=2.0,
        zorder=9,
        label="Exact posterior MAP",
    )
    ax1.set_xlim(-0.2, 1.45)
    ax1.set_ylim(0.2, 2.6)
    ax1.set_xlabel(r"Horizontal bias $\alpha$")
    ax1.set_ylabel(r"Restoring strength $\kappa$")
    ax1.set_title("68% Credible Contours vs Hidden True Settings")
    ax1.legend(loc="upper left", fontsize=8.8)

    # Right panel: 1D marginal forest plot (mean & 68% CI)
    y_pos = np.arange(len(methods))[::-1]
    ax2.axvline(obs["alpha_true"], color=PALETTE["truth"], ls="--", lw=1.6, alpha=0.8, label="True $\\alpha = 0.60$")
    ax2.axvline(obs["kappa_true"], color=PALETTE["truth"], ls=":", lw=1.6, alpha=0.8, label="True $\\kappa = 1.20$")
    for y, (label, g, col, _, _) in zip(y_pos, methods):
        comp = compare_posterior_grids(posts["exact"], g)
        ax2.errorbar(
            comp["mean_alpha"],
            y + 0.13,
            xerr=[[comp["mean_alpha"] - comp["ci68_alpha_low"]], [comp["ci68_alpha_high"] - comp["mean_alpha"]]],
            fmt="o",
            color=col,
            capsize=3,
            lw=1.8,
        )
        ax2.errorbar(
            comp["mean_kappa"],
            y - 0.13,
            xerr=[[comp["mean_kappa"] - comp["ci68_kappa_low"]], [comp["ci68_kappa_high"] - comp["mean_kappa"]]],
            fmt="s",
            color=col,
            capsize=3,
            lw=1.8,
            alpha=0.75,
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m[0] for m in methods])
    ax2.set_xlabel("Parameter value (circles: $\\alpha$, squares: $\\kappa$, bars: 68% CI)")
    ax2.set_title("Finite-Sample Offset vs Approximation Error")
    ax2.legend(loc="lower right", fontsize=8.8)
    fig.tight_layout()
    return Image(data=fig_to_png_bytes(fig))


def show_backup_activations_and_mdn(bundle: dict | None = None) -> Image:
    """Backup figure showing original lecture's MDN conditional density slices at x in {0.1, 0.4, 0.9, 1.4}."""
    b = bundle or load_lecture_bundle()
    reg = b["reg_suite"]
    entry = reg["by_skew"][1.0]
    x_eval = reg["x_eval"]
    y_eval = reg["y_eval"]

    apply_lecture_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.8))

    z = np.linspace(-4, 4, 200)
    ax1.plot(z, np.maximum(0, z), label="ReLU $\\max(0, z)$", lw=2.0)
    ax1.plot(z, z * expit(z), label="Swish / SiLU $z\\,\\sigma(z)$", lw=2.0)
    ax1.plot(z, np.tanh(z), label="$\\tanh(z)$", lw=2.0)
    ax1.plot(z, np.log1p(np.exp(z)), label="Softplus $\\log(1 + e^z) > 0$", lw=2.0, ls="--")
    ax1.set_ylim(-1.3, 3.5)
    ax1.set_xlabel("Pre-activation $z$")
    ax1.set_ylabel("$\\sigma(z)$")
    ax1.set_title("Nonlinear Activations & Positive Scale Maps")
    ax1.legend(loc="upper left", fontsize=9.0)

    ax2.plot([], [], color="grey", lw=2.0, label="True $p(y\\mid x)$")
    ax2.plot([], [], color="grey", lw=2.0, ls="--", label="Learned MDN $q_\\phi(y\\mid x)$")
    for idx, x_target in enumerate([0.15, 0.45, 0.90, 1.40]):
        ix = int(np.argmin(np.abs(x_eval - x_target)))
        xv = float(x_eval[ix])
        col = f"C{idx}"
        ax2.plot(y_eval, skewed_conditional_distribution(xv, 1.0).pdf(y_eval), color=col, lw=1.9, label=f"$x = {xv:.2f}$")
        ax2.plot(y_eval, entry["mdn_cond_pdf"][ix], color=col, lw=1.9, ls="--")
    ax2.set_xlim(0.2, 8.5)
    ax2.set_xlabel("$y$")
    ax2.set_ylabel("Conditional density")
    ax2.set_title("MDN Recovers Skewed Conditional Densities (Cell 074)")
    ax2.legend(loc="upper right", fontsize=8.8, ncol=2)
    fig.tight_layout()
    return Image(data=fig_to_png_bytes(fig))


# =============================================================================
# W01 — Forward versus Inverse Problem (Slide 2)
# =============================================================================

def w01_forward_inverse(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    obs = b["primary_obs"]
    exact_post = b["obs_posteriors"][0]["exact"]

    mode_w = widgets.ToggleButtons(
        options=[("Forward (θ → x)", "forward"), ("Inverse (x_obs → θ?)", "inverse")],
        value="forward",
        description="Direction:",
    )
    alpha_w = widgets.FloatSlider(value=0.6, min=-1.5, max=1.5, step=0.1, description="α (bias):")
    kappa_w = widgets.FloatSlider(value=1.2, min=0.0, max=3.0, step=0.1, description="κ (pull):")
    sim_btn = widgets.Button(description="Simulate", icon="play", layout=widgets.Layout(width="105px"))
    reset_btn = _make_reset_button()
    seed_box = {"counter": 2026}

    def render() -> plt.Figure:
        fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(10.2, 3.8))
        ks = np.arange(13)

        if mode_w.value == "forward":
            a_val, k_val = float(alpha_w.value), float(kappa_w.value)
            hist = simulate_plinko_histograms(a_val, k_val, n_balls=100, n_rows=12, seed=seed_box["counter"])
            probs = exact_bin_probabilities(a_val, k_val, n_rows=12)

            ax_p.axvspan(-1.5, 1.5, ymin=0, ymax=1, color="#f3f4f6", alpha=0.5)
            ax_p.scatter([a_val], [k_val], s=180, color=PALETTE["NPE"], zorder=5, label="Chosen $\\theta = (\\alpha, \\kappa)$")
            ax_p.set_xlim(-1.5, 1.5)
            ax_p.set_ylim(0.0, 3.0)
            ax_p.set_xlabel(r"Horizontal bias $\alpha$")
            ax_p.set_ylabel(r"Restoring strength $\kappa$")
            ax_p.set_title("1. Known Cause: Parameter $\\theta$")
            ax_p.legend(loc="upper left")

            ax_d.bar(ks, hist, color=PALETTE["NPE"], alpha=0.82, label="Simulated $x \\sim p(x\\mid\\theta)$")
            ax_d.plot(ks, 100.0 * probs, "o--", color=PALETTE["exact"], lw=1.7, label="Expected counts")
            ax_d.set_xticks(ks)
            ax_d.set_ylim(0, 62)
            ax_d.set_xlabel("Landing bin $k$")
            ax_d.set_ylabel("Ball count (N = 100)")
            ax_d.set_title("2. Forward Prediction: Simulated Data $x$")
            ax_d.legend(loc="upper left")
            lw_obj_metrics = {"mode": "forward", "sum_counts": int(np.sum(hist))}
        else:
            aa, kk = np.meshgrid(exact_post["alpha_grid"], exact_post["kappa_grid"], indexing="ij")
            ax_p.contourf(
                aa,
                kk,
                exact_post["posterior"],
                levels=12,
                cmap="Blues",
                alpha=0.85,
            )
            ax_p.contour(
                aa,
                kk,
                exact_post["posterior"],
                levels=exact_post["hpd_levels"],
                colors=[PALETTE["NPE"]],
                linewidths=1.8,
            )
            cands = [(0.45, 0.95), (0.75, 1.35), (1.05, 1.80)]
            for idx_c, (ca, ck) in enumerate(cands):
                ax_p.scatter([ca], [ck], s=70, color=f"C{idx_c + 1}", edgecolors="black", zorder=6, label=f"Plausible $\\theta_{idx_c+1}$")
            ax_p.set_xlim(-1.5, 1.5)
            ax_p.set_ylim(0.0, 3.0)
            ax_p.set_xlabel(r"Horizontal bias $\alpha$")
            ax_p.set_ylabel(r"Restoring strength $\kappa$")
            ax_p.set_title("2. Inverse Target: Plausible $p(\\theta \\mid x_{\\mathrm{obs}})$")
            ax_p.legend(loc="upper left", fontsize=8.8)

            ax_d.bar(ks, obs["x_obs"], color=PALETTE["obs"], alpha=0.85, label="Observed $x_{\\mathrm{obs}}$")
            for idx_c, (ca, ck) in enumerate(cands):
                ax_d.plot(ks, 100.0 * exact_bin_probabilities(ca, ck, 12), lw=1.8, color=f"C{idx_c + 1}", label=f"Model at $\\theta_{idx_c+1}$")
            ax_d.set_xticks(ks)
            ax_d.set_ylim(0, 62)
            ax_d.set_xlabel("Landing bin $k$")
            ax_d.set_ylabel("Ball count (N = 100)")
            ax_d.set_title("1. Observed Effect: Data $x_{\\mathrm{obs}}$")
            ax_d.legend(loc="upper left", fontsize=8.8)
            lw_obj_metrics = {"mode": "inverse", "sum_counts": int(np.sum(obs["x_obs"]))}

        fig.tight_layout()
        return fig, lw_obj_metrics

    def reset() -> None:
        seed_box["counter"] = 2026
        mode_w.value = "forward"
        alpha_w.value = 0.6
        kappa_w.value = 1.2

    controls = {
        "mode": mode_w,
        "alpha": alpha_w,
        "kappa": kappa_w,
        "simulate": sim_btn,
        "reset": reset_btn,
    }
    row1 = widgets.HBox([mode_w, sim_btn, reset_btn])
    row2 = widgets.HBox([alpha_w, kappa_w])
    lw = LectureWidget("W01_Forward_Inverse", controls, widgets.VBox([row1, row2]), render, reset)

    for w in (mode_w, alpha_w, kappa_w):
        w.observe(lambda _c: lw.refresh(), names="value")

    def _on_sim(_b):
        seed_box["counter"] += 1
        lw.refresh()

    sim_btn.on_click(_on_sim)
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W02 — Animated Plinko Simulator (Slides 5 & 7)
# =============================================================================

def w02_plinko_simulator(
    initial_alpha: float = 0.0,
    initial_kappa: float = 0.0,
    advanced_mode: bool = True,
) -> LectureWidget:
    alpha_w = widgets.FloatSlider(value=initial_alpha, min=-1.5, max=1.5, step=0.05, description="α (bias):")
    kappa_w = widgets.FloatSlider(
        value=initial_kappa,
        min=0.0,
        max=3.0,
        step=0.05,
        description="κ (pull):",
        disabled=not advanced_mode,
    )
    balls_w = widgets.IntSlider(value=100, min=1, max=300, step=1, description="Balls N:")
    rows_w = widgets.IntSlider(value=12, min=6, max=16, step=2, description="Rows R:")
    seed_w = widgets.BoundedIntText(value=2026, min=0, max=999999, description="Seed:", layout=widgets.Layout(width="165px"))
    field_w = widgets.Checkbox(value=advanced_mode, description="Show P(right) field", indent=False)
    drop1_btn = widgets.Button(description="Drop 1 Ball", icon="arrow-down", layout=widgets.Layout(width="115px"))
    drop100_btn = widgets.Button(description="Drop 100", icon="fast-forward", layout=widgets.Layout(width="105px"))
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        a_val = float(alpha_w.value)
        k_val = float(kappa_w.value) if advanced_mode else 0.0
        n_b = int(balls_w.value)
        n_r = int(rows_w.value)
        sd = int(seed_w.value)

        sim = simulate_trajectories(a_val, k_val, n_balls=n_b, n_rows=n_r, seed=sd)
        st = plinko_summary_statistics(sim["histogram"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
        _draw_plinko_board_on_ax(
            ax1,
            n_rows=n_r,
            s_paths=sim["s_paths"],
            highlight_idx=n_b - 1,
            alpha=a_val,
            kappa=k_val,
            show_field=bool(field_w.value),
        )
        ax1.set_title(f"Plinko Board (R={n_r}, α={a_val:+.2f}, κ={k_val:.2f})")

        ks = np.arange(n_r + 1)
        ax2.bar(ks, sim["histogram"], color=PALETTE["NPE"], alpha=0.84, width=0.76, label=f"Empirical ({n_b} balls)")
        ax2.axvline(st["mean"], color=PALETTE["truth"], ls="--", lw=1.8, label=f"Mean = {st['mean']:.2f} (std = {st['std']:.2f})")
        ax2.set_xticks(ks)
        ax2.set_xlabel("Landing bin $k_R$")
        ax2.set_ylabel("Ball count $n_k$")
        ax2.set_ylim(0, max(int(np.max(sim["histogram"])) + 6, 12))
        ax2.set_title("Simulated Landing Histogram")
        ax2.legend(loc="upper left")
        fig.tight_layout()
        return fig, {
            "histogram": sim["histogram"].copy(),
            "mean": st["mean"],
            "std": st["std"],
            "n_balls": n_b,
        }

    def reset() -> None:
        alpha_w.value = initial_alpha
        kappa_w.value = initial_kappa
        balls_w.value = 100
        rows_w.value = 12
        seed_w.value = 2026
        field_w.value = advanced_mode

    controls = {
        "alpha": alpha_w,
        "kappa": kappa_w,
        "n_balls": balls_w,
        "n_rows": rows_w,
        "seed": seed_w,
        "show_field": field_w,
        "drop_one": drop1_btn,
        "drop_hundred": drop100_btn,
        "reset": reset_btn,
    }
    row1 = widgets.HBox([alpha_w, kappa_w, field_w])
    row2 = widgets.HBox([balls_w, rows_w, seed_w, drop1_btn, drop100_btn, reset_btn])
    lw = LectureWidget("W02_Plinko_Simulator", controls, widgets.VBox([row1, row2]), render, reset)

    for w in (alpha_w, kappa_w, balls_w, rows_w, seed_w, field_w):
        w.observe(lambda _c: lw.refresh(), names="value")

    def _drop_one(_b):
        lw._updating = True
        balls_w.value = 1
        seed_w.value = int(seed_w.value) + 1
        lw._updating = False
        lw.refresh()

    def _drop_hundred(_b):
        lw._updating = True
        balls_w.value = 100
        seed_w.value = int(seed_w.value) + 1
        lw._updating = False
        lw.refresh()

    drop1_btn.on_click(_drop_one)
    drop100_btn.on_click(_drop_hundred)
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W03 — Analytic versus Generalized Galton Board (Slide 6)
# =============================================================================

def w03_analytic_vs_generalized_galton() -> LectureWidget:
    mode_w = widgets.ToggleButtons(
        options=[("Analytic Binomial (κ = 0)", "analytic"), ("Generalized Restoring (κ > 0)", "generalized")],
        value="analytic",
        description="Model:",
    )
    p_w = widgets.FloatSlider(value=0.60, min=0.10, max=0.90, step=0.02, description="p (right):")
    kappa_w = widgets.FloatSlider(value=1.20, min=0.0, max=3.0, step=0.10, description="κ (pull):")
    n_w = widgets.SelectionSlider(
        options=[25, 100, 500, 2500, 10000],
        value=100,
        description="Balls N:",
    )
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        p_val = float(p_w.value)
        a_val = float(logit(np.clip(p_val, 1e-4, 1.0 - 1e-4)))
        k_val = 0.0 if mode_w.value == "analytic" else float(kappa_w.value)
        n_b = int(n_w.value)

        emp_counts = simulate_plinko_histograms(a_val, k_val, n_balls=n_b, n_rows=12, seed=2026)
        emp_freq = emp_counts / float(n_b)
        ks = np.arange(13)

        if mode_w.value == "analytic":
            theo_prob = binomial_bin_probabilities(p_val, n_rows=12)
            theo_label = f"Closed-form Binomial(12, p={p_val:.2f})"
            theo_col = PALETTE["truth"]
        else:
            theo_prob = exact_bin_probabilities(a_val, k_val, n_rows=12)
            theo_label = f"State-dependent exact PMF (α={a_val:+.2f}, κ={k_val:.2f})"
            theo_col = PALETTE["NLE"]

        tv_err = 0.5 * float(np.sum(np.abs(emp_freq - theo_prob)))

        fig, ax = plt.subplots(figsize=(9.2, 3.8))
        ax.bar(ks, emp_freq, width=0.72, color=PALETTE["NPE"], alpha=0.72, label=f"Empirical frequency (N = {n_b})")
        ax.plot(ks, theo_prob, "o-", color=theo_col, lw=2.3, markersize=6.5, label=theo_label)
        if mode_w.value == "generalized":
            ax.plot(
                ks,
                binomial_bin_probabilities(p_val, n_rows=12),
                "s--",
                color=PALETTE["rejected"],
                lw=1.5,
                label=f"Reference κ=0 Binomial(12, {p_val:.2f})",
            )
        ax.set_xticks(ks)
        ax.set_ylim(0.0, 0.58)
        ax.set_xlabel("Landing bin $k$")
        ax.set_ylabel("Probability / Relative frequency")
        ax.set_title(f"Empirical vs Theoretical Distribution — Total Variation Error = {tv_err:.3f}")
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig, {"tv_error": tv_err, "theo_prob": theo_prob.copy(), "mode": mode_w.value}

    def reset() -> None:
        mode_w.value = "analytic"
        p_w.value = 0.60
        kappa_w.value = 1.20
        n_w.value = 100

    controls = {"mode": mode_w, "p_right": p_w, "kappa": kappa_w, "n_balls": n_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([mode_w, reset_btn]), widgets.HBox([p_w, kappa_w, n_w])])
    lw = LectureWidget("W03_Analytic_Galton", controls, ui_box, render, reset)
    for w in (mode_w, p_w, kappa_w, n_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W04 — Original 1D ABC Explorer: chi2(5) prior, Normal(sqrt(theta), 1) (Slides 11-13)
# =============================================================================

def w04_abc_1d_explorer(joint_only_default: bool = False) -> LectureWidget:
    max_pool = 10000
    joint_contours = chi2_normal_joint_grid()

    view_w = widgets.ToggleButtons(
        options=[("Joint Scatter + Contours", "joint"), ("ABC Slice + Posterior", "slice")],
        value="joint" if joint_only_default else "slice",
        description="View:",
    )
    xobs_w = widgets.FloatSlider(value=2.5, min=0.0, max=5.5, step=0.1, description="x_obs:")
    eps_w = widgets.FloatSlider(value=0.25, min=0.03, max=2.0, step=0.02, description="ε (tol):")
    n_w = widgets.SelectionSlider(options=[500, 1500, 2500, 5000, 10000], value=2500, description="Sim pool N:")
    seed_w = widgets.BoundedIntText(value=42, min=0, max=99999, description="Seed:", layout=widgets.Layout(width="155px"))
    reset_btn = _make_reset_button()

    pool_cache: dict[int, dict[str, np.ndarray]] = {}

    def _get_pool(sd: int) -> dict[str, np.ndarray]:
        if sd not in pool_cache:
            pool_cache[sd] = simulate_chi2_normal_joint(n_samples=max_pool, seed=sd)
        return pool_cache[sd]

    def render() -> plt.Figure:
        pool = _get_pool(int(seed_w.value))
        n_use = int(n_w.value)
        th = pool["theta"][:n_use]
        xs = pool["x"][:n_use]
        x_obs = float(xobs_w.value)
        eps = float(eps_w.value)

        abc_res = run_1d_rejection_abc(th, xs, x_obs=x_obs, epsilon=eps)
        exact = chi2_normal_exact_posterior(x_obs=x_obs)
        abc_exact = chi2_normal_abc_analytic_posterior(x_obs=x_obs, epsilon=eps)

        if view_w.value == "joint":
            fig, ax = plt.subplots(figsize=(8.8, 4.0))
            ax.scatter(th, xs, s=5, alpha=0.42, color=PALETTE["NPE"], label=r"Simulated pairs $(\theta_i, x_i) \sim p(\theta, x)$")
            ax.contour(
                joint_contours["theta_grid"],
                joint_contours["x_grid"],
                joint_contours["joint_pdf"],
                levels=7,
                colors=[PALETTE["exact"]],
                alpha=0.65,
                linewidths=1.2,
            )
            ax.set_xlim(0.0, 17.0)
            ax.set_ylim(-2.0, 7.0)
            ax.set_xlabel(r"Parameter $\theta \sim \chi^2(\mathrm{df}=5)$")
            ax.set_ylabel(r"Simulated data $x \mid \theta \sim \mathcal{N}(\sqrt{\theta}, 1)$")
            ax.set_title(f"Joint Distribution $p(\\theta, x)$ ({n_use} forward simulations)")
            ax.legend(loc="lower right")
        else:
            fig, (ax_top, ax_bot) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(9.0, 4.7),
                gridspec_kw={"height_ratios": [0.52, 1.0], "hspace": 0.08},
            )
            acc = abc_res["accepted_mask"]
            ax_bot.scatter(th[~acc], xs[~acc], s=4, alpha=0.25, color=PALETTE["rejected"], label="Rejected pairs")
            ax_bot.scatter(th[acc], xs[acc], s=14, alpha=0.85, color=PALETTE["abc"], label=f"Accepted ({abc_res['n_accepted']}/{n_use})")
            ax_bot.contour(
                joint_contours["theta_grid"],
                joint_contours["x_grid"],
                joint_contours["joint_pdf"],
                levels=6,
                colors=[PALETTE["exact"]],
                alpha=0.4,
                linewidths=1.0,
            )
            ax_bot.axhspan(x_obs - eps, x_obs + eps, color=PALETTE["abc"], alpha=0.16)
            ax_bot.axhline(x_obs, color=PALETTE["obs"], lw=1.8, label=f"$x_{{\\mathrm{{obs}}}} = {x_obs:.2f}$")
            ax_bot.axhline(x_obs - eps, color=PALETTE["abc"], ls="--", lw=1.3)
            ax_bot.axhline(x_obs + eps, color=PALETTE["abc"], ls="--", lw=1.3, label=f"$x_{{\\mathrm{{obs}}}} \\pm \\epsilon$ ($\\epsilon={eps:.2f}$)")
            ax_bot.set_xlim(0.0, 17.0)
            ax_bot.set_ylim(-2.0, 7.0)
            ax_bot.set_xlabel(r"Parameter $\theta$")
            ax_bot.set_ylabel(r"Observation $x$")
            ax_bot.legend(loc="lower right", ncol=2, fontsize=8.8)

            mask_g = exact["theta_grid"] <= 17.0
            ax_top.plot(
                exact["theta_grid"][mask_g],
                exact["posterior_pdf"][mask_g],
                color=PALETTE["exact"],
                lw=2.2,
                label=r"Exact posterior $p(\theta\mid x_{\mathrm{obs}})$",
            )
            ax_top.plot(
                abc_exact["theta_grid"][mask_g],
                abc_exact["abc_posterior_pdf"][mask_g],
                color=PALETTE["abc"],
                ls="--",
                lw=1.6,
                label=r"Exact finite-$\epsilon$ $p_\epsilon(\theta\mid x_{\mathrm{obs}})$",
            )
            if abc_res["n_accepted"] >= 3:
                bins = np.linspace(0.0, 17.0, min(max(int(np.sqrt(abc_res["n_accepted"])) + 4, 8), 28))
                ax_top.hist(
                    abc_res["accepted_theta"],
                    bins=bins,
                    density=True,
                    color=PALETTE["abc"],
                    alpha=0.45,
                    label="Accepted $\\theta_i$ histogram",
                )
            ax_top.set_ylabel("Density")
            ax_top.set_title(
                f"ABC Acceptance Rate: {abc_res['n_accepted']} / {n_use} ({100.0 * abc_res['acceptance_rate']:.1f}%)"
            )
            ax_top.legend(loc="upper right", fontsize=8.8)

        if view_w.value == "joint":
            fig.tight_layout()
        return fig, {
            "n_accepted": abc_res["n_accepted"],
            "acceptance_rate": abc_res["acceptance_rate"],
            "exact_mean": exact["mean"],
        }

    def reset() -> None:
        view_w.value = "joint" if joint_only_default else "slice"
        xobs_w.value = 2.5
        eps_w.value = 0.25
        n_w.value = 2500
        seed_w.value = 42

    controls = {"view": view_w, "x_obs": xobs_w, "epsilon": eps_w, "n_samples": n_w, "seed": seed_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([view_w, seed_w, reset_btn]), widgets.HBox([xobs_w, eps_w, n_w])])
    lw = LectureWidget("W04_1D_ABC", controls, ui_box, render, reset)
    for w in (view_w, xobs_w, eps_w, n_w, seed_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W05 — Correlated-Gaussian ABC Explorer (Slide 14)
# =============================================================================

def w05_correlated_gaussian_abc() -> LectureWidget:
    rho_w = widgets.FloatSlider(value=0.85, min=-0.95, max=0.95, step=0.05, description="ρ (corr):")
    eps1_w = widgets.FloatSlider(value=0.12, min=0.03, max=0.60, step=0.03, description="ε₁ (tight):")
    eps2_w = widgets.FloatSlider(value=1.50, min=0.40, max=2.50, step=0.10, description="ε₂ (wide):")
    xobs_w = widgets.FloatSlider(value=0.50, min=-2.5, max=2.5, step=0.10, description="x_obs:")
    n_w = widgets.SelectionSlider(options=[2000, 5000, 10000, 20000], value=10000, description="Samples N:")
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        rho = float(rho_w.value)
        e1 = float(eps1_w.value)
        e2 = float(eps2_w.value)
        x_obs = float(xobs_w.value)
        n_s = int(n_w.value)

        sim = simulate_correlated_gaussian_joint(n_samples=n_s, rho=rho, seed=42)
        exact = correlated_gaussian_exact_conditional(x_obs=x_obs, rho=rho)
        abc1 = run_1d_rejection_abc(sim["theta"], sim["x"], x_obs=x_obs, epsilon=e1)
        abc2 = run_1d_rejection_abc(sim["theta"], sim["x"], x_obs=x_obs, epsilon=e2)

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(9.2, 4.7),
            gridspec_kw={"height_ratios": [0.52, 1.0], "hspace": 0.08},
        )
        ax_bot.scatter(sim["theta"], sim["x"], s=2.5, alpha=0.28, color="#6b7280", label=r"Joint $(\theta_i, x_i)$")
        ax_bot.axhline(x_obs, color=PALETTE["obs"], lw=1.8, label=f"$x_{{\\mathrm{{obs}}}} = {x_obs:+.2f}$")
        for eps_val, col, tag in ((e1, PALETTE["NPE"], "ε₁"), (e2, PALETTE["abc"], "ε₂")):
            ax_bot.axhline(x_obs - eps_val, color=col, ls="--", lw=1.4, label=f"{tag} = {eps_val:.2f}")
            ax_bot.axhline(x_obs + eps_val, color=col, ls="--", lw=1.4)

        ax_bot.set_xlim(-3.5, 3.5)
        ax_bot.set_ylim(-4.5, 4.5)
        ax_bot.set_xlabel(r"Parameter $\theta$")
        ax_bot.set_ylabel(r"Observation $x$")
        ax_bot.legend(loc="lower right", ncol=2, fontsize=8.8)

        ax_top.plot(
            exact["theta_grid"],
            exact["posterior_pdf"],
            color=PALETTE["exact"],
            lw=2.3,
            label=f"Exact $\\mathcal{{N}}(\\mu={exact['cond_mean']:+.2f}, \\sigma={exact['cond_std']:.2f})$",
        )
        bins = np.linspace(-3.5, 3.5, 34)
        if abc2["n_accepted"] >= 5:
            ax_top.hist(
                abc2["accepted_theta"],
                bins=bins,
                density=True,
                alpha=0.38,
                color=PALETTE["abc"],
                label=f"ε₂={e2:.2f} ({abc2['n_accepted']}/{n_s}, {100*abc2['acceptance_rate']:.1f}%)",
            )
        if abc1["n_accepted"] >= 5:
            ax_top.hist(
                abc1["accepted_theta"],
                bins=bins,
                density=True,
                alpha=0.52,
                color=PALETTE["NPE"],
                label=f"ε₁={e1:.2f} ({abc1['n_accepted']}/{n_s}, {100*abc1['acceptance_rate']:.1f}%)",
            )
        ax_top.set_ylabel("Density")
        ax_top.set_title(
            r"Conditional Gaussian Posterior: $\mathbb{E}[\theta\mid x]=\mu_\theta+\rho(\sigma_\theta/\sigma_x)(x-\mu_x)$, "
            r"$\mathrm{Var}(\theta\mid x)=\sigma_\theta^2(1-\rho^2)$",
            fontsize=10.5,
        )
        ax_top.legend(loc="upper left", fontsize=8.6)
        return fig, {
            "cond_mean": exact["cond_mean"],
            "cond_var": exact["cond_var"],
            "n_acc_tight": abc1["n_accepted"],
            "n_acc_wide": abc2["n_accepted"],
        }

    def reset() -> None:
        rho_w.value = 0.85
        eps1_w.value = 0.12
        eps2_w.value = 1.50
        xobs_w.value = 0.50
        n_w.value = 10000

    controls = {"rho": rho_w, "eps_small": eps1_w, "eps_large": eps2_w, "x_obs": xobs_w, "n_samples": n_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([rho_w, xobs_w, n_w, reset_btn]), widgets.HBox([eps1_w, eps2_w])])
    lw = LectureWidget("W05_Correlated_Gaussian_ABC", controls, ui_box, render, reset)
    for w in (rho_w, eps1_w, eps2_w, xobs_w, n_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W06 — Plinko ABC Explorer (Slide 15)
# =============================================================================

def w06_plinko_abc_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    bank = b["abc_bank"]
    x_obs = b["primary_obs"]["x_obs"]
    exact_post = b["obs_posteriors"][0]["exact"]
    max_b = int(bank["thetas"].shape[0])

    dist_w = widgets.Dropdown(
        options=[(v, k) for k, v in DISTANCE_LABELS.items()],
        value="wasserstein",
        description="Distance:",
    )
    eps_w = widgets.FloatSlider(value=0.35, min=0.05, max=1.50, step=0.03, description="ε (tol):")
    budget_opts = sorted({v for v in (2000, 10000, 25000, max_b) if v <= max_b})
    budget_w = widgets.SelectionSlider(options=budget_opts, value=budget_opts[-1], description="Budget:")
    ref_w = widgets.Checkbox(value=False, description="Overlay exact reference", indent=False)
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        dk = str(dist_w.value)
        eps = float(eps_w.value)
        nb = int(budget_w.value)
        res = run_plinko_abc(bank, x_obs, epsilon=eps, distance_kind=dk, summary_kind="full_hist", n_simulations=nb)

        fig, (ax_h, ax_p) = plt.subplots(1, 2, figsize=(10.5, 4.1))
        ks = np.arange(13)
        ax_h.bar(ks - 0.18, x_obs, width=0.36, color=PALETTE["obs"], alpha=0.88, label="Observed $x_{\\mathrm{obs}}$")
        ax_h.bar(
            ks + 0.18,
            res["best_histogram"],
            width=0.36,
            color=PALETTE["abc"],
            alpha=0.85,
            label=f"Closest sim ($d={res['best_distance']:.3f}$)",
        )
        if res["n_accepted"] > 0:
            mean_acc_h = np.mean(res["accepted_histograms"], axis=0)
            ax_h.plot(ks, mean_acc_h, "o--", color=PALETTE["NPE"], lw=1.6, label="Mean accepted hist")
        ax_h.set_xticks(ks)
        ax_h.set_xlabel("Landing bin $k$")
        ax_h.set_ylabel("Ball count (N = 100)")
        ax_h.set_title("Data Space: Comparing Histograms")
        ax_h.legend(loc="upper left", fontsize=8.8)

        rej = res["rejected_thetas"]
        n_rej_plot = min(rej.shape[0], 4500)
        ax_p.scatter(rej[:n_rej_plot, 0], rej[:n_rej_plot, 1], s=4, alpha=0.20, color=PALETTE["rejected"], label="Rejected")
        acc = res["accepted_thetas"]
        if acc.shape[0] > 0:
            ax_p.scatter(acc[:, 0], acc[:, 1], s=14, alpha=0.72, color=PALETTE["abc"], label=f"Accepted ({res['n_accepted']})")
        if res["kde_grid"] is not None:
            g = res["kde_grid"]
            aa, kk = np.meshgrid(g["alpha_grid"], g["kappa_grid"], indexing="ij")
            ax_p.contour(aa, kk, g["posterior"], levels=g["hpd_levels"], colors=["#b45309"], linewidths=1.8)
        if ref_w.value:
            aa, kk = np.meshgrid(exact_post["alpha_grid"], exact_post["kappa_grid"], indexing="ij")
            ax_p.contour(aa, kk, exact_post["posterior"], levels=exact_post["hpd_levels"], colors=[PALETTE["exact"]], linestyles="--", linewidths=1.8)
            ax_p.plot([], [], color=PALETTE["exact"], ls="--", lw=1.8, label="Exact posterior (68/95%)")

        ax_p.set_xlim(-1.5, 1.5)
        ax_p.set_ylim(0.0, 3.0)
        ax_p.set_xlabel(r"Horizontal bias $\alpha$")
        ax_p.set_ylabel(r"Restoring strength $\kappa$")
        ax_p.set_title(f"Accepted: {res['n_accepted']} / {nb} ({100.0 * res['acceptance_rate']:.2f}%)")
        ax_p.legend(loc="upper left", fontsize=8.8)
        fig.tight_layout()
        return fig, {
            "n_accepted": res["n_accepted"],
            "acceptance_rate": res["acceptance_rate"],
        }

    def reset() -> None:
        dist_w.value = "wasserstein"
        eps_w.value = 0.35
        budget_w.value = budget_opts[-1]
        ref_w.value = False

    controls = {"distance": dist_w, "epsilon": eps_w, "budget": budget_w, "show_reference": ref_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([dist_w, ref_w, reset_btn]), widgets.HBox([eps_w, budget_w])])
    lw = LectureWidget("W06_Plinko_ABC", controls, ui_box, render, reset)
    for w in (dist_w, eps_w, budget_w, ref_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W07 — Summary-Statistic & Distance Explorer (Slide 16)
# =============================================================================

def w07_summary_statistic_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    bank = b["abc_bank"]
    x_obs = b["primary_obs"]["x_obs"]
    exact_post = b["obs_posteriors"][0]["exact"]

    sum_w = widgets.ToggleButtons(
        options=[
            ("Mean Only (1D)", "mean_only"),
            ("Mean + Variance (2D)", "mean_var"),
            ("Full Histogram (13D)", "full_hist"),
            ("Side-by-Side All 3", "compare_all"),
        ],
        value="compare_all",
        description="Summary:",
    )
    eps_w = widgets.FloatSlider(value=0.18, min=0.04, max=0.80, step=0.02, description="ε (scaled):")
    dist_w = widgets.Dropdown(options=[("Euclidean / Wasserstein", "wasserstein"), ("L1 / Total Variation", "tv")], value="wasserstein", description="Metric:")
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        eps = float(eps_w.value)
        dk = str(dist_w.value)
        aa_ref, kk_ref = np.meshgrid(exact_post["alpha_grid"], exact_post["kappa_grid"], indexing="ij")

        if sum_w.value == "compare_all":
            fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7), sharex=True, sharey=True)
            specs = [
                ("1. Mean Only $S(x) = \\bar{k}$", "mean_only", eps * 0.45, "#dc2626"),
                ("2. Mean + Variance $(\\bar{k}, s_k^2)$", "mean_var", eps, "#059669"),
                ("3. Full Histogram $x$", "full_hist", eps * 1.85, "#2563eb"),
            ]
            counts_list = []
            for ax, (title, sk, ep_use, col) in zip(axes, specs):
                res = run_plinko_abc(bank, x_obs, epsilon=ep_use, distance_kind=dk, summary_kind=sk)
                acc = res["accepted_thetas"]
                counts_list.append(res["n_accepted"])
                if acc.shape[0] > 0:
                    ax.scatter(acc[:1200, 0], acc[:1200, 1], s=8, alpha=0.45, color=col)
                if res["kde_grid"] is not None:
                    g = res["kde_grid"]
                    ax.contour(aa_ref, kk_ref, g["posterior"], levels=[g["hpd_levels"][-1]], colors=[col], linewidths=2.0)
                ax.contour(aa_ref, kk_ref, exact_post["posterior"], levels=[exact_post["hpd_levels"][-1]], colors=[PALETTE["exact"]], linestyles="--", linewidths=1.5)
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(0.0, 3.0)
                ax.set_xlabel(r"Bias $\alpha$")
                ax.set_title(f"{title}\nAcc: {res['n_accepted']} ({100*res['acceptance_rate']:.1f}%)", fontsize=10.2)
            axes[0].set_ylabel(r"Restoring pull $\kappa$")
            metrics = {"accepted_counts": counts_list}
        else:
            sk = str(sum_w.value)
            ep_use = eps * 1.85 if sk == "full_hist" else eps
            res = run_plinko_abc(bank, x_obs, epsilon=ep_use, distance_kind=dk, summary_kind=sk)
            fig, ax = plt.subplots(figsize=(7.2, 4.1))
            acc = res["accepted_thetas"]
            if acc.shape[0] > 0:
                ax.scatter(acc[:, 0], acc[:, 1], s=12, alpha=0.55, color=PALETTE["abc"], label=f"Accepted ({res['n_accepted']})")
            if res["kde_grid"] is not None:
                g = res["kde_grid"]
                ax.contour(aa_ref, kk_ref, g["posterior"], levels=g["hpd_levels"], colors=[PALETTE["abc"]], linewidths=2.0)
            ax.contour(aa_ref, kk_ref, exact_post["posterior"], levels=exact_post["hpd_levels"], colors=[PALETTE["exact"]], linestyles="--", linewidths=1.8)
            ax.plot([], [], color=PALETTE["exact"], ls="--", lw=1.8, label="Exact posterior (68/95% HPD)")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(0.0, 3.0)
            ax.set_xlabel(r"Horizontal bias $\alpha$")
            ax.set_ylabel(r"Restoring strength $\kappa$")
            ax.set_title(f"{SUMMARY_LABELS[sk]} — Acc: {res['n_accepted']}/{res['n_total']} ({100*res['acceptance_rate']:.2f}%)")
            ax.legend(loc="upper left")
            metrics = {"n_accepted": res["n_accepted"]}

        fig.tight_layout()
        return fig, metrics

    def reset() -> None:
        sum_w.value = "compare_all"
        eps_w.value = 0.18
        dist_w.value = "wasserstein"

    controls = {"summary": sum_w, "epsilon": eps_w, "distance": dist_w, "reset": reset_btn}
    ui_box = widgets.VBox([sum_w, widgets.HBox([eps_w, dist_w, reset_btn])])
    lw = LectureWidget("W07_Summary_Explorer", controls, ui_box, render, reset)
    for w in (sum_w, eps_w, dist_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W08 — Gaussian Mixture Editor (Slide 21)
# =============================================================================

def w08_gmm_editor() -> LectureWidget:
    k_w = widgets.ToggleButtons(options=[("2 Components", 2), ("3 Components", 3)], value=3, description="Components:")
    w1 = widgets.FloatSlider(value=0.35, min=0.05, max=2.0, step=0.05, description="w₁ (raw):")
    m1 = widgets.FloatSlider(value=-0.2, min=-2.0, max=4.0, step=0.1, description="μ₁:")
    s1 = widgets.FloatSlider(value=0.35, min=0.10, max=1.5, step=0.05, description="σ₁:")

    w2 = widgets.FloatSlider(value=0.45, min=0.05, max=2.0, step=0.05, description="w₂ (raw):")
    m2 = widgets.FloatSlider(value=1.0, min=-2.0, max=4.0, step=0.1, description="μ₂:")
    s2 = widgets.FloatSlider(value=0.55, min=0.10, max=1.5, step=0.05, description="σ₂:")

    w3 = widgets.FloatSlider(value=0.20, min=0.05, max=2.0, step=0.05, description="w₃ (raw):")
    m3 = widgets.FloatSlider(value=2.3, min=-2.0, max=4.0, step=0.1, description="μ₃:")
    s3 = widgets.FloatSlider(value=0.65, min=0.10, max=1.5, step=0.05, description="σ₃:")

    gamma_w = widgets.Checkbox(value=False, description="Overlay Gamma(3, 1/3) target", indent=False)
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        k_use = int(k_w.value)
        raw_w = [float(w1.value), float(w2.value), float(w3.value)][:k_use]
        means = [float(m1.value), float(m2.value), float(m3.value)][:k_use]
        stds = [float(s1.value), float(s2.value), float(s3.value)][:k_use]

        gmm = GaussianMixture1D(weights=np.array(raw_w), means=np.array(means), stds=np.array(stds))
        xg = np.linspace(-2.5, 5.0, 500)
        comp_weighted = gmm.component_pdfs(xg, weighted=True)
        mix_pdf = gmm.pdf(xg)
        area = float(np.trapezoid(mix_pdf, xg))

        fig, ax = plt.subplots(figsize=(9.2, 3.8))
        for idx in range(k_use):
            ax.plot(
                xg,
                comp_weighted[idx],
                ls="--",
                lw=1.7,
                color=f"C{idx}",
                label=f"Comp {idx+1}: $w={gmm.weights[idx]:.2f}, \\mu={gmm.means[idx]:.1f}, \\sigma={gmm.stds[idx]:.2f}$",
            )
            ax.fill_between(xg, 0.0, comp_weighted[idx], color=f"C{idx}", alpha=0.14)
        ax.plot(xg, mix_pdf, color=PALETTE["exact"], lw=2.5, label=f"Mixture $q_\\phi(x)$ ($\\int q_\\phi\\,dx \\approx {area:.3f}$)")
        if gamma_w.value:
            ax.plot(xg, gamma_target_distribution().pdf(xg), color=PALETTE["truth"], ls=":", lw=2.2, label="Target Gamma(3, 1/3)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("Density")
        ax.set_title("Constructing Flexible Non-Gaussian Densities from Weighted Gaussians")
        ax.legend(loc="upper right", fontsize=8.8)
        fig.tight_layout()
        return fig, {"weights": gmm.weights.copy(), "integral": area}

    def reset() -> None:
        k_w.value = 3
        w1.value, m1.value, s1.value = 0.35, -0.2, 0.35
        w2.value, m2.value, s2.value = 0.45, 1.0, 0.55
        w3.value, m3.value, s3.value = 0.20, 2.3, 0.65
        gamma_w.value = False

    controls = {
        "n_components": k_w,
        "w1": w1, "m1": m1, "s1": s1,
        "w2": w2, "m2": m2, "s2": s2,
        "w3": w3, "m3": m3, "s3": s3,
        "show_gamma": gamma_w,
        "reset": reset_btn,
    }
    ui_box = widgets.VBox(
        [
            widgets.HBox([k_w, gamma_w, reset_btn]),
            widgets.HBox([w1, m1, s1]),
            widgets.HBox([w2, m2, s2]),
            widgets.HBox([w3, m3, s3]),
        ]
    )
    lw = LectureWidget("W08_GMM_Editor", controls, ui_box, render, reset)
    for w in controls.values():
        if w is not reset_btn:
            w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W09 — Density-Estimation Training Explorer: Gamma(3, 1/3) -> GMM (Slide 22)
# =============================================================================

def w09_gmm_training_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    suite = b["gmm_suite"]
    steps = [int(s) for s in suite["checkpoint_steps"]]
    target = gamma_target_distribution()

    step_w = widgets.SelectionSlider(options=steps, value=steps[-1], description="Train step:")
    n_w = widgets.ToggleButtons(options=[(f"N={n}", n) for n in suite["sample_sizes"]], value=1000, description="Samples:")
    k_w = widgets.ToggleButtons(options=[(f"K={k}", k) for k in suite["comp_counts"]], value=5, description="Mixture K:")
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        st = int(step_w.value)
        n_s = int(n_w.value)
        k_c = int(k_w.value)
        run = suite["runs"][(n_s, k_c)]
        ck = run["checkpoints"][st]
        gmm = GaussianMixture1D(weights=ck["weights"], means=ck["means"], stds=ck["stds"])

        fig, (ax_d, ax_l) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1.0]})
        xg = np.linspace(-0.4, 4.6, 400)
        ax_d.hist(run["samples"], bins=28, density=True, alpha=0.35, color=PALETTE["NPE"], label=f"Samples $x_i \\sim p$ (N={n_s})")
        ax_d.plot(xg, target.pdf(xg), color=PALETTE["exact"], lw=2.2, label="True $p(x) = \\mathrm{Gamma}(3, 1/3)$")
        for c_pdf in gmm.component_pdfs(xg, weighted=True):
            ax_d.plot(xg, c_pdf, color=PALETTE["abc"], ls=":", lw=1.1, alpha=0.8)
        ax_d.plot(xg, gmm.pdf(xg), color=PALETTE["abc"], lw=2.4, label=f"Fitted $q_\\phi(x)$ (K={k_c}, step={st})")
        ax_d.set_xlim(-0.4, 4.5)
        ax_d.set_xlabel("$x$")
        ax_d.set_ylabel("Density")
        ax_d.set_title(f"Forward KL $D_{{\\mathrm{{KL}}}}(p \\| q_\\phi) \\approx {ck['kl']:.3f}$ nats")
        ax_d.legend(loc="upper right", fontsize=8.7)

        iters = np.arange(run["nll_history"].size)
        ax_l.plot(iters, run["nll_history"], color=PALETTE["NPE"], lw=2.0, label=r"Empirical NLL $-\frac{1}{N}\sum \log q_\phi(x_i)$")
        ax_l.axhline(run["empirical_entropy"], color=PALETTE["exact"], ls="--", lw=1.6, label=r"Entropy floor $\mathbb{E}_p[-\log p(x)]$")
        ax_l.scatter([st], [ck["nll"]], s=95, color=PALETTE["truth"], zorder=5, label=f"Step {st}: NLL = {ck['nll']:.3f}")
        ax_l.set_xlabel("Optimization iteration")
        ax_l.set_ylabel("Loss (Negative Log-Likelihood)")
        ax_l.set_title("Minimizing NLL Minimizes Forward KL")
        ax_l.legend(loc="upper right", fontsize=8.7)
        fig.tight_layout()
        return fig, {"nll": ck["nll"], "kl": ck["kl"], "step": st}

    def reset() -> None:
        step_w.value = steps[-1]
        n_w.value = 1000
        k_w.value = 5

    controls = {"step": step_w, "n_samples": n_w, "n_components": k_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([step_w, reset_btn]), widgets.HBox([n_w, k_w])])
    lw = LectureWidget("W09_GMM_Training", controls, ui_box, render, reset)
    for w in (step_w, n_w, k_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W10 — L1 vs L2 Regression & Conditional Density Explorer (Slide 23)
# =============================================================================

def w10_loss_regression_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    reg = b["reg_suite"]

    loss_w = widgets.ToggleButtons(
        options=[
            ("Both L2 & L1", "both"),
            ("L2 Only (Mean)", "l2"),
            ("L1 Only (Median)", "l1"),
            ("+ Full Density (MDN)", "all_mdn"),
        ],
        value="both",
        description="Loss:",
    )
    x_w = widgets.FloatSlider(value=1.15, min=0.15, max=1.45, step=0.05, description="Slice x:")
    skew_w = widgets.ToggleButtons(
        options=[("Mild (0.6)", 0.6), ("Original (1.0)", 1.0), ("Heavy (1.25)", 1.25)],
        value=1.0,
        description="Skewness:",
    )
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        sk = float(skew_w.value)
        entry = reg["by_skew"][sk]
        x_eval = reg["x_eval"]
        y_eval = reg["y_eval"]
        x_sl = float(x_w.value)
        l_mode = str(loss_w.value)

        theo = skewed_conditional_summaries(x_eval, skew_scale=sk)
        xg_bg = np.linspace(0.1, 1.5, 110)
        yg_bg = np.linspace(0.0, 12.0, 120)
        pdf_bg = skewed_conditional_distribution(xg_bg[None, :], skew_scale=sk).pdf(yg_bg[:, None])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
        ax1.imshow(pdf_bg, origin="lower", extent=(0.1, 1.5, 0.0, 12.0), cmap="bone_r", aspect="auto", alpha=0.72)
        ax1.scatter(entry["x_samples"][:700], entry["y_samples"][:700], s=3, color="#111827", alpha=0.35, label="Samples $(x_i, y_i)$")

        if l_mode in ("l2", "both", "all_mdn"):
            ax1.plot(x_eval, entry["l2_pred"], color=PALETTE["NPE"], lw=2.2, label="MLP with $L_2$ loss")
            ax1.plot(x_eval, theo["mean"], color=PALETTE["NPE"], ls="--", lw=1.5, label="True mean $\\mathbb{E}[y\\mid x]$")
        if l_mode in ("l1", "both", "all_mdn"):
            ax1.plot(x_eval, entry["l1_pred"], color=PALETTE["abc"], lw=2.2, label="MLP with $L_1$ loss")
            ax1.plot(x_eval, theo["median"], color=PALETTE["abc"], ls="--", lw=1.5, label="True median")
        if l_mode == "all_mdn":
            ax1.plot(x_eval, entry["mdn_mode"], color=PALETTE["NLE"], lw=2.1, label="MDN mode ($L_{\\mathrm{KL}}$)")
            ax1.plot(x_eval, theo["mode"], color=PALETTE["NLE"], ls="--", lw=1.4, label="True mode")

        ax1.axvline(x_sl, color=PALETTE["truth"], ls=":", lw=1.9)
        ax1.set_xlim(0.1, 1.5)
        ax1.set_ylim(0.0, 12.0)
        ax1.set_xlabel("Input $x$")
        ax1.set_ylabel("Target $y$")
        ax1.set_title("Point Regression Losses vs Conditional Distribution")
        ax1.legend(loc="upper left", fontsize=8.3, ncol=2)

        sl_pdf = skewed_conditional_distribution(x_sl, skew_scale=sk).pdf(y_eval)
        sl_sum = skewed_conditional_summaries(x_sl, skew_scale=sk)
        ax2.plot(y_eval, sl_pdf, color=PALETTE["exact"], lw=2.3, label=f"True $p(y \\mid x={x_sl:.2f})$")
        ax2.fill_between(y_eval, 0.0, sl_pdf, color="#e5e7eb", alpha=0.6)
        if l_mode in ("l2", "both", "all_mdn"):
            ax2.axvline(float(sl_sum["mean"]), color=PALETTE["NPE"], lw=2.0, label=f"Mean ($L_2$) = {float(sl_sum['mean']):.2f}")
        if l_mode in ("l1", "both", "all_mdn"):
            ax2.axvline(float(sl_sum["median"]), color=PALETTE["abc"], lw=2.0, label=f"Median ($L_1$) = {float(sl_sum['median']):.2f}")
        if l_mode == "all_mdn":
            ix = int(np.argmin(np.abs(x_eval - x_sl)))
            ax2.plot(y_eval, entry["mdn_cond_pdf"][ix], color=PALETTE["NLE"], ls="--", lw=2.2, label="Learned MDN $q_\\phi(y\\mid x)$")
            ax2.axvline(float(sl_sum["mode"]), color=PALETTE["NLE"], ls=":", lw=1.8, label=f"Mode = {float(sl_sum['mode']):.2f}")

        ax2.set_xlim(max(0.0, 2.0 * x_sl - 0.3), min(12.0, float(sl_sum["mean"]) + 4.5))
        ax2.set_xlabel(f"Outcome $y$ at slice $x = {x_sl:.2f}$")
        ax2.set_ylabel("Conditional density $p(y\\mid x)$")
        ax2.set_title("Neither Mean nor Median Captures Full Skewed Uncertainty")
        ax2.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        return fig, {
            "slice_mean": float(sl_sum["mean"]),
            "slice_median": float(sl_sum["median"]),
            "slice_mode": float(sl_sum["mode"]),
        }

    def reset() -> None:
        loss_w.value = "both"
        x_w.value = 1.15
        skew_w.value = 1.0

    controls = {"loss_mode": loss_w, "x_slice": x_w, "skewness": skew_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([loss_w, reset_btn]), widgets.HBox([x_w, skew_w])])
    lw = LectureWidget("W10_L1_L2_Regression", controls, ui_box, render, reset)
    for w in (loss_w, x_w, skew_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W11 — Simulation-to-Inference Amortized Pipeline (Slide 25)
# =============================================================================

def w11_amortized_pipeline_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    observations = b["observations"]
    posts = b["obs_posteriors"]
    ns = b["neural_suite"]

    obs_w = widgets.ToggleButtons(
        options=[(o["label"].split(" (")[0], idx) for idx, o in enumerate(observations)],
        value=0,
        description="Evaluate on:",
    )
    show_cloud_w = widgets.SelectionSlider(options=[200, 500, 1000], value=500, description="Show train:")
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        idx = int(obs_w.value)
        obs = observations[idx]
        g_npe = posts[idx]["NPE"]
        g_ex = posts[idx]["exact"]
        n_vis = int(show_cloud_w.value)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.2, 3.6))

        th_tr = ns["sample_train_thetas"][:n_vis]
        st_tr = plinko_summary_statistics(ns["sample_train_hists"][:n_vis])
        sc = ax1.scatter(th_tr[:, 0], th_tr[:, 1], c=st_tr["mean"], cmap="viridis", s=14, alpha=0.75)
        plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.03, label="Mean bin $\\bar{k}$")
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(0.0, 3.0)
        ax1.set_xlabel(r"$\alpha_i \sim p(\alpha)$")
        ax1.set_ylabel(r"$\kappa_i \sim p(\kappa)$")
        ax1.set_title(f"1. Offline Training Bank\n(N = {ns['n_train']} sims — paid ONCE)", fontsize=10.5)

        ks = np.arange(13)
        ax2.bar(ks, obs["x_obs"], color=PALETTE["obs"], alpha=0.86)
        ax2.set_xticks(range(0, 13, 2))
        ax2.set_xlabel("Landing bin $k$")
        ax2.set_ylabel("Counts")
        ax2.set_title(f"2. Any New Observation $x_{{\\mathrm{{obs}}}}$\n({obs['label']})", fontsize=10.5)

        aa, kk = np.meshgrid(g_npe["alpha_grid"], g_npe["kappa_grid"], indexing="ij")
        ax3.contourf(aa, kk, g_npe["posterior"], levels=12, cmap="Blues", alpha=0.85)
        ax3.contour(aa, kk, g_npe["posterior"], levels=g_npe["hpd_levels"], colors=[PALETTE["NPE"]], linewidths=2.0)
        ax3.contour(aa, kk, g_ex["posterior"], levels=g_ex["hpd_levels"], colors=[PALETTE["exact"]], linestyles="--", linewidths=1.5)
        ax3.plot([], [], color=PALETTE["NPE"], lw=2.0, label="Amortized $q_\\phi(\\theta\\mid x)$")
        ax3.plot([], [], color=PALETTE["exact"], ls="--", lw=1.5, label="Exact posterior")
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(0.0, 3.0)
        ax3.set_xlabel(r"Bias $\alpha$")
        ax3.set_ylabel(r"Pull $\kappa$")
        ax3.set_title("3. Instant Posterior (<1 ms)\nZero New Simulations!", fontsize=10.5)
        ax3.legend(loc="upper left", fontsize=8.3)
        fig.tight_layout()
        return fig, {"obs_index": idx, "npe_mean_alpha": g_npe["mean_alpha"]}

    def reset() -> None:
        obs_w.value = 0
        show_cloud_w.value = 500

    controls = {"observation": obs_w, "n_shown": show_cloud_w, "reset": reset_btn}
    ui_box = widgets.VBox([obs_w, widgets.HBox([show_cloud_w, reset_btn])])
    lw = LectureWidget("W11_Amortized_Pipeline", controls, ui_box, render, reset)
    for w in (obs_w, show_cloud_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W12 — Unified NPE / NLE / NRE Comparison Explorer (Slides 27, 28, 30)
# =============================================================================

def w12_neural_sbi_comparison_explorer(
    default_method: str = "NPE",
    show_truth_default: bool = False,
    bundle: dict | None = None,
) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    observations = b["observations"]
    posts = b["obs_posteriors"]
    ns = b["neural_suite"]
    abc_bank = b["abc_bank"]

    method_w = widgets.ToggleButtons(
        options=[("NPE", "NPE"), ("NLE", "NLE"), ("NRE", "NRE"), ("Compare All 3", "ALL")],
        value=default_method,
        description="Estimator:",
    )
    obs_w = widgets.Dropdown(
        options=[(o["label"], idx) for idx, o in enumerate(observations)],
        value=0,
        description="Data:",
    )
    ref_w = widgets.Checkbox(value=True, description="Exact DP posterior", indent=False)
    abc_w = widgets.Checkbox(value=False, description="ABC (ε=0.30)", indent=False)
    truth_w = widgets.Checkbox(value=show_truth_default, description="Show true θ", indent=False)
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        idx = int(obs_w.value)
        obs = observations[idx]
        p_set = posts[idx]
        g_ex = p_set["exact"]
        m_sel = str(method_w.value)

        fig = plt.figure(figsize=(9.8, 4.6))
        gs = GridSpec(2, 3, width_ratios=[1.15, 0.36, 0.95], height_ratios=[0.36, 1.0], wspace=0.14, hspace=0.14, figure=fig)
        ax_marg_a = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_marg_a)
        ax_marg_k = fig.add_subplot(gs[1, 1], sharey=ax_joint)
        ax_info = fig.add_subplot(gs[:, 2])

        aa, kk = np.meshgrid(g_ex["alpha_grid"], g_ex["kappa_grid"], indexing="ij")
        active_methods = ["NPE", "NLE", "NRE"] if m_sel == "ALL" else [m_sel]

        if m_sel != "ALL":
            g_one = p_set[m_sel]
            if m_sel == "NPE":
                samps = ns["npe"].sample(obs["x_obs"], n_samples=500, seed=42)
            else:
                samps = sample_from_grid_posterior(g_one, n_samples=500, seed=42)
            ax_joint.scatter(samps[:, 0], samps[:, 1], s=7, alpha=0.25, color=PALETTE[m_sel], label=f"{m_sel} samples")

        for m_name in active_methods:
            gm = p_set[m_name]
            col = PALETTE[m_name]
            ax_joint.contour(aa, kk, gm["posterior"], levels=gm["hpd_levels"], colors=[col], linewidths=2.0)
            ax_joint.plot([], [], color=col, lw=2.0, label=f"{m_name} (68/95% HPD)")
            ax_marg_a.plot(gm["alpha_grid"], gm["marginal_alpha"], color=col, lw=2.0)
            ax_marg_k.plot(gm["marginal_kappa"], gm["kappa_grid"], color=col, lw=2.0)

        if ref_w.value:
            ax_joint.contour(aa, kk, g_ex["posterior"], levels=g_ex["hpd_levels"], colors=[PALETTE["exact"]], linestyles="--", linewidths=1.7)
            ax_joint.plot([], [], color=PALETTE["exact"], ls="--", lw=1.7, label="Exact DP reference")
            ax_marg_a.plot(g_ex["alpha_grid"], g_ex["marginal_alpha"], color=PALETTE["exact"], ls="--", lw=1.7)
            ax_marg_k.plot(g_ex["marginal_kappa"], g_ex["kappa_grid"], color=PALETTE["exact"], ls="--", lw=1.7)

        if abc_w.value:
            abc_r = run_plinko_abc(abc_bank, obs["x_obs"], epsilon=0.30, distance_kind="wasserstein")
            if abc_r["kde_grid"] is not None:
                ga = abc_r["kde_grid"]
                ax_joint.contour(aa, kk, ga["posterior"], levels=[ga["hpd_levels"][-1]], colors=[PALETTE["abc"]], linestyles=":", linewidths=1.8)
                ax_joint.plot([], [], color=PALETTE["abc"], ls=":", lw=1.8, label="ABC (68% HPD)")
                ax_marg_a.plot(ga["alpha_grid"], ga["marginal_alpha"], color=PALETTE["abc"], ls=":", lw=1.6)
                ax_marg_k.plot(ga["marginal_kappa"], ga["kappa_grid"], color=PALETTE["abc"], ls=":", lw=1.6)

        if truth_w.value:
            ax_joint.scatter([obs["alpha_true"]], [obs["kappa_true"]], marker="*", s=200, color=PALETTE["truth"], edgecolors="black", zorder=10, label="True $\\theta$")
            ax_marg_a.axvline(obs["alpha_true"], color=PALETTE["truth"], ls=":", lw=1.5)
            ax_marg_k.axhline(obs["kappa_true"], color=PALETTE["truth"], ls=":", lw=1.5)

        ax_joint.set_xlim(-1.5, 1.5)
        ax_joint.set_ylim(0.0, 3.0)
        ax_joint.set_xlabel(r"Horizontal bias $\alpha$")
        ax_joint.set_ylabel(r"Restoring strength $\kappa$")
        ax_joint.legend(loc="upper left", fontsize=8.2)
        plt.setp(ax_marg_a.get_xticklabels(), visible=False)
        plt.setp(ax_marg_k.get_yticklabels(), visible=False)
        ax_marg_a.set_ylabel("$p(\\alpha\\mid x)$")
        ax_marg_k.set_xlabel("$p(\\kappa\\mid x)$")

        # Right panel: if NLE or NRE selected individually, show learned unnormalized log-surface + summary;
        # if ALL or NPE, show observed histogram + quantitative TV table!
        if m_sel in ("NLE", "NRE"):
            g_one = p_set[m_sel]
            surf = np.maximum(g_one["aux_surface"], float(np.max(g_one["aux_surface"])) - 15.0)
            im = ax_info.contourf(aa, kk, surf, levels=14, cmap="Greens" if m_sel == "NLE" else "Purples")
            plt.colorbar(im, ax=ax_info, fraction=0.046, pad=0.03)
            ax_info.set_xlim(-1.5, 1.5)
            ax_info.set_ylim(0.0, 3.0)
            ax_info.set_xlabel(r"$\alpha$")
            ax_info.set_ylabel(r"$\kappa$")
            title_s = (
                "Learned Log-Likelihood $\\log q_\\phi(x_{\\mathrm{obs}}\\mid\\theta)$\n(Combine with $p(\\theta)$ & normalize!)"
                if m_sel == "NLE"
                else "Learned Log-Ratio $\\log \\hat{r}_\\phi(\\theta, x_{\\mathrm{obs}})$\n(Multiply by $p(\\theta)$ & normalize!)"
            )
            ax_info.set_title(title_s, fontsize=9.8)
        else:
            ks = np.arange(13)
            ax_info.bar(ks, obs["x_obs"], color=PALETTE["obs"], alpha=0.85)
            lines = ["2D TV distance vs Exact DP:"]
            for mn in ("NPE", "NLE", "NRE"):
                comp = compare_posterior_grids(g_ex, p_set[mn])
                lines.append(
                    f"• {mn}: TV={comp['tv_distance']:.3f} | "
                    f"α={comp['mean_alpha']:+.2f}±{comp['std_alpha']:.2f}, "
                    f"κ={comp['mean_kappa']:.2f}±{comp['std_kappa']:.2f}"
                )
            ax_info.text(
                0.02,
                0.97,
                "\n".join(lines),
                transform=ax_info.transAxes,
                va="top",
                ha="left",
                fontsize=8.3,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9fafb", edgecolor="#d1d5db"),
            )
            ax_info.set_ylim(0, max(int(np.max(obs["x_obs"])) + 22, 55))
            ax_info.set_xticks(range(0, 13, 2))
            ax_info.set_xlabel("Landing bin $k$")
            ax_info.set_ylabel("Observed count")
            ax_info.set_title("Conditioning Data & Accuracy", fontsize=10.2)

        return fig, {
            m: compare_posterior_grids(g_ex, p_set[m])["tv_distance"] for m in ("NPE", "NLE", "NRE")
        }

    def reset() -> None:
        method_w.value = default_method
        obs_w.value = 0
        ref_w.value = True
        abc_w.value = False
        truth_w.value = show_truth_default

    controls = {
        "method": method_w,
        "observation": obs_w,
        "show_reference": ref_w,
        "show_abc": abc_w,
        "show_truth": truth_w,
        "reset": reset_btn,
    }
    ui_box = widgets.VBox([widgets.HBox([method_w, reset_btn]), widgets.HBox([obs_w, ref_w, abc_w, truth_w])])
    lw = LectureWidget("W12_Neural_SBI_Comparison", controls, ui_box, render, reset)
    for w in (method_w, obs_w, ref_w, abc_w, truth_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W13 — NRE Joint-vs-Independent Classification Explorer (Slide 29)
# =============================================================================

def w13_nre_classification_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    ns = b["neural_suite"]
    nre = ns["nre"]
    x_obs = b["primary_obs"]["x_obs"]
    g_nre = b["obs_posteriors"][0]["NRE"]

    pair_w = widgets.ToggleButtons(
        options=[
            ("Compare Joint vs Marginal", "both"),
            ("Joint p(θ, x) [y = 1]", "joint"),
            ("Independent p(θ)p(x) [y = 0]", "marginal"),
        ],
        value="both",
        description="Pairs:",
    )
    out_w = widgets.Dropdown(
        options=[
            ("Classifier Prob D(θ, x)", "prob"),
            ("Log-Ratio log r(θ, x) = logit(D)", "log_ratio"),
        ],
        value="prob",
        description="Surface:",
    )
    a_w = widgets.FloatSlider(value=0.75, min=-1.4, max=1.4, step=0.05, description="Probe α:")
    k_w = widgets.FloatSlider(value=1.35, min=0.1, max=2.9, step=0.05, description="Probe κ:")
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        th_tr = ns["sample_train_thetas"][:600]
        h_tr = ns["sample_train_hists"][:600]
        means_joint = plinko_summary_statistics(h_tr)["mean"]
        rng = np.random.default_rng(77)
        means_marg = means_joint[rng.permutation(means_joint.size)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        pm = str(pair_w.value)
        if pm in ("marginal", "both"):
            ax1.scatter(th_tr[:, 0], means_marg, s=11, alpha=0.40, color=PALETTE["rejected"], label="Marginal $p(\\theta)p(x)$ ($y=0$)")
        if pm in ("joint", "both"):
            ax1.scatter(th_tr[:, 0], means_joint, s=12, alpha=0.65, color=PALETTE["NRE"], label="Joint $p(\\theta, x)$ ($y=1$)")
        ax1.set_xlabel(r"Bias parameter $\alpha$")
        ax1.set_ylabel(r"Summary mean bin $\bar{k}(x)$")
        ax1.set_title("Classification Training Pairs: Matched vs Shuffled")
        ax1.legend(loc="upper left", fontsize=8.8)

        aa, kk = np.meshgrid(g_nre["alpha_grid"], g_nre["kappa_grid"], indexing="ij")
        log_r_surf = g_nre["aux_surface"]
        if out_w.value == "prob":
            surf = expit(log_r_surf)
            cbar_lbl = r"$D_\phi(\theta, x_{\mathrm{obs}})$"
            cmap = "PuBuGn"
        else:
            surf = np.maximum(log_r_surf, float(np.max(log_r_surf)) - 12.0)
            cbar_lbl = r"$\log \hat{r}_\phi(\theta, x_{\mathrm{obs}})$"
            cmap = "Purples"

        im = ax2.contourf(aa, kk, surf, levels=14, cmap=cmap)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.03, label=cbar_lbl)

        pa, pk = float(a_w.value), float(k_w.value)
        probe = nre.classify_and_ratio(np.array([[pa, pk]]), x_obs)
        d_val = float(probe["prob_joint"][0])
        odds_val = float(probe["odds"][0])
        lr_val = float(probe["log_ratio"][0])

        ax2.scatter([pa], [pk], s=130, color=PALETTE["truth"], edgecolors="white", linewidths=1.5, zorder=8)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(0.0, 3.0)
        ax2.set_xlabel(r"Horizontal bias $\alpha$")
        ax2.set_ylabel(r"Restoring strength $\kappa$")
        ax2.set_title(
            f"Probe ({pa:+.2f}, {pk:.2f}): D={d_val:.3f}, D/(1-D)={odds_val:.2f}, log r={lr_val:+.2f}",
            fontsize=10.0,
        )
        fig.tight_layout()
        return fig, {"prob_joint": d_val, "odds": odds_val, "log_ratio": lr_val}

    def reset() -> None:
        pair_w.value = "both"
        out_w.value = "prob"
        a_w.value = 0.75
        k_w.value = 1.35

    controls = {"pair_mode": pair_w, "output_view": out_w, "alpha_probe": a_w, "kappa_probe": k_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([pair_w, reset_btn]), widgets.HBox([out_w, a_w, k_w])])
    lw = LectureWidget("W13_NRE_Classification", controls, ui_box, render, reset)
    for w in (pair_w, out_w, a_w, k_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W14 — Posterior Predictive Explorer (Slide 32)
# =============================================================================

def w14_posterior_predictive_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    x_obs = b["primary_obs"]["x_obs"]
    posts = b["obs_posteriors"][0]
    ns = b["neural_suite"]
    abc_bank = b["abc_bank"]

    method_w = widgets.Dropdown(
        options=[
            ("NPE Posterior", "NPE"),
            ("NLE Posterior", "NLE"),
            ("NRE Posterior", "NRE"),
            ("ABC Posterior (ε=0.30)", "ABC"),
            ("Exact DP Posterior", "Exact"),
            ("Misspecified κ=0 Binomial Model!", "Misspecified"),
        ],
        value="NPE",
        description="Model:",
    )
    nrep_w = widgets.SelectionSlider(options=[50, 150, 250, 500], value=250, description="Simulations:")
    stat_w = widgets.Dropdown(
        options=[
            ("Spread / Std s_k(x)", "std"),
            ("Mean landing bin k̄(x)", "mean"),
            ("Peak bin count max(x)", "max_count"),
            ("Chi-square discrepancy", "chi2"),
        ],
        value="std",
        description="Statistic T:",
    )
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        m_name = str(method_w.value)
        n_rep = int(nrep_w.value)
        sk = str(stat_w.value)

        if m_name == "NPE":
            samps = ns["npe"].sample(x_obs, n_samples=n_rep, seed=2026)
        elif m_name in ("NLE", "NRE"):
            samps = sample_from_grid_posterior(posts[m_name], n_samples=n_rep, seed=2026)
        elif m_name == "Exact":
            samps = sample_from_grid_posterior(posts["exact"], n_samples=n_rep, seed=2026)
        elif m_name == "ABC":
            samps = run_plinko_abc(abc_bank, x_obs, epsilon=0.30)["accepted_thetas"]
        else:
            # Misspecified kappa = 0 model fitted to empirical mean k_bar / R = sigma(alpha)
            rng_m = np.random.default_rng(2026)
            k_tot = float(np.sum(np.arange(13) * x_obs))
            p_draws = rng_m.beta(1.0 + k_tot, 1.0 + (1200.0 - k_tot), size=n_rep)
            samps = np.column_stack([logit(np.clip(p_draws, 1e-4, 1 - 1e-4)), np.zeros(n_rep)])

        ppc = run_posterior_predictive_check(samps, x_obs, n_predictive=n_rep, stat_kind=sk, seed=202605)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
        ks = np.arange(13)
        col = PALETTE.get(m_name, "#dc2626" if m_name == "Misspecified" else PALETTE["exact"])

        ax1.fill_between(ks, ppc["q05"], ppc["q95"], color=col, alpha=0.18, step="mid", label="90% predictive band")
        ax1.fill_between(ks, ppc["q25"], ppc["q75"], color=col, alpha=0.34, step="mid", label="50% predictive band")
        for r_idx in range(min(5, ppc["rep_histograms"].shape[0])):
            ax1.plot(ks, ppc["rep_histograms"][r_idx], color=col, alpha=0.35, lw=1.0)
        ax1.plot(ks, x_obs, "o-", color=PALETTE["obs"], lw=2.2, markersize=5.5, label="Observed $x_{\\mathrm{obs}}$")
        ax1.set_xticks(ks)
        ax1.set_xlabel("Landing bin $k$")
        ax1.set_ylabel("Ball count (N = 100)")
        ax1.set_title(f"Posterior Predictive Envelope ({m_name})")
        ax1.legend(loc="upper left", fontsize=8.8)

        ax2.hist(ppc["t_rep"], bins=22, density=True, color=col, alpha=0.55, label=r"Predictive $T(x^{\mathrm{rep}})$")
        ax2.axvline(ppc["t_obs"], color=PALETTE["obs"], lw=2.4, ls="--", label=f"Observed $T(x_{{\\mathrm{{obs}}}}) = {ppc['t_obs']:.2f}$")
        ax2.set_xlabel(f"Test statistic $T(x)$ [{sk}]")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Predictive Tail Fraction $P(T^{{\\mathrm{{rep}}}} \\geq T^{{\\mathrm{{obs}}}}) = {ppc['p_value']:.2f}$")
        ax2.legend(loc="upper right", fontsize=8.8)
        fig.tight_layout()
        return fig, {"p_value": ppc["p_value"], "t_obs": ppc["t_obs"]}

    def reset() -> None:
        method_w.value = "NPE"
        nrep_w.value = 250
        stat_w.value = "std"

    controls = {"method": method_w, "n_rep": nrep_w, "stat_kind": stat_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([method_w, stat_w, reset_btn]), nrep_w])
    lw = LectureWidget("W14_Posterior_Predictive", controls, ui_box, render, reset)
    for w in (method_w, nrep_w, stat_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw


# =============================================================================
# W15 — Coverage & Calibration Explorer (Slide 33)
# =============================================================================

def w15_coverage_calibration_explorer(bundle: dict | None = None) -> LectureWidget:
    b = bundle or load_lecture_bundle()
    cov_suite = b["coverage_suite"]
    max_v = int(cov_suite["n_validation"])

    method_w = widgets.ToggleButtons(
        options=[
            ("Compare All", "ALL"),
            ("NPE", "NPE"),
            ("NLE", "NLE"),
            ("NRE", "NRE"),
            ("ABC", "ABC"),
            ("Exact", "Exact"),
        ],
        value="ALL",
        description="Method:",
    )
    val_opts = sorted({v for v in (40, 100, max_v) if v <= max_v})
    nval_w = widgets.ToggleButtons(options=[(f"M={v}", v) for v in val_opts], value=val_opts[-1], description="Experiments:")
    nom_w = widgets.FloatSlider(value=0.68, min=0.10, max=0.95, step=0.02, description="Nominal 1-α:")
    reset_btn = _make_reset_button()

    def render() -> plt.Figure:
        m_sel = str(method_w.value)
        n_v = int(nval_w.value)
        nom = float(nom_w.value)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
        ref_sum = summarize_empirical_coverage(cov_suite["hpd_ranks"]["Exact"][:n_v])
        ax1.fill_between(
            ref_sum["nominal"],
            ref_sum["band_low"],
            ref_sum["band_high"],
            color="#e5e7eb",
            alpha=0.85,
            label=f"95% binomial band (M={n_v})",
        )
        ax1.plot([0, 1], [0, 1], "k--", lw=1.6, label="Ideal calibration")

        methods_show = ["Exact", "NPE", "NLE", "NRE", "ABC"] if m_sel == "ALL" else [m_sel]
        emp_at_nom = {}
        for mn in methods_show:
            col = PALETTE["exact"] if mn == "Exact" else PALETTE[mn.lower() if mn == "ABC" else mn]
            ranks = cov_suite["hpd_ranks"][mn][:n_v]
            cs = summarize_empirical_coverage(ranks)
            emp_val = float(np.mean(ranks <= nom))
            emp_at_nom[mn] = emp_val
            ax1.plot(cs["nominal"], cs["empirical"], color=col, lw=2.0, label=f"{mn} (@{int(round(nom*100))}% -> {int(round(emp_val*100))}%)")
            ax1.scatter([nom], [emp_val], color=col, s=55, zorder=6)

        ax1.axvline(nom, color=PALETTE["truth"], ls=":", lw=1.4)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Nominal Credibility Level $1 - \\alpha$")
        ax1.set_ylabel("Empirical Coverage Fraction")
        ax1.set_title("2D Credible-Region Coverage Across Repeated Simulations")
        ax1.legend(loc="upper left", fontsize=8.4)

        focus_m = "NPE" if m_sel == "ALL" else m_sel
        f_col = PALETTE["exact"] if focus_m == "Exact" else PALETTE[focus_m.lower() if focus_m == "ABC" else focus_m]
        n_bins = 10
        p_a = cov_suite["pit_alpha"][focus_m][:n_v]
        p_k = cov_suite["pit_kappa"][focus_m][:n_v]
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ha, _ = np.histogram(p_a, bins=edges, density=True)
        hk, _ = np.histogram(p_k, bins=edges, density=True)
        lo_u, hi_u = np.asarray(binom.ppf([0.025, 0.975], n=n_v, p=1.0 / n_bins)) / max(n_v / n_bins, 1e-6)
        ax2.axhspan(lo_u, hi_u, color="#e5e7eb", alpha=0.85, label="95% uniform band")
        ax2.axhline(1.0, color="black", ls="--", lw=1.4)
        ax2.bar(centers - 0.018, ha, width=0.034, color=f_col, alpha=0.82, label=f"{focus_m}: PIT rank $\\alpha$")
        ax2.bar(centers + 0.018, hk, width=0.034, color="#64748b", alpha=0.72, label=f"{focus_m}: PIT rank $\\kappa$")
        ax2.set_xlim(0.0, 1.0)
        ax2.set_xlabel("Marginal Posterior CDF Rank $F(\\theta^* \\mid x^*)$")
        ax2.set_ylabel("Normalized rank density")
        ax2.set_title(f"1D Simulation-Based Calibration Ranks ({focus_m})")
        ax2.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        return fig, {"empirical_at_nominal": emp_at_nom, "n_val": n_v}

    def reset() -> None:
        method_w.value = "ALL"
        nval_w.value = val_opts[-1]
        nom_w.value = 0.68

    controls = {"method": method_w, "n_validation": nval_w, "nominal_level": nom_w, "reset": reset_btn}
    ui_box = widgets.VBox([widgets.HBox([method_w, reset_btn]), widgets.HBox([nval_w, nom_w])])
    lw = LectureWidget("W15_Coverage_Calibration", controls, ui_box, render, reset)
    for w in (method_w, nval_w, nom_w):
        w.observe(lambda _c: lw.refresh(), names="value")
    reset_btn.on_click(lambda _b: lw.reset())
    return lw
