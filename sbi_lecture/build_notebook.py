"""Generate and execute `LECTURES/week10_simulation_based_inference.ipynb` from `SLIDES`.

Produces an executable ETH Zürich HS2026 lecture notebook formatted with the exact
`slideshow` and `remove-input` / `remove-cell` cell metadata conventions used by
`course_tools/make_slides.sh` across Weeks 1–9, with:
  - Live movable ipywidgets-style slider & training controls on every visualisation.
  - Accessible Python/PyTorch implementations of the Galton simulator, dynamic-programming
    benchmark, rejection ABC, NPE, NLE, NRE, PPC, and calibration utilities.
  - Clean lecturer voice and zero curriculum scaffolding on the slides.

Usage:
    python -m sbi_lecture.build_notebook
"""

from __future__ import annotations

import re
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from nbclient import NotebookClient

from .slide_deck import SLIDES


CODE_INSPECTION_CELLS: dict[str, str] = {
    "s07-state-dependent": """# Inspectable forward simulator for the Mystery Galton Board (R=12, N=100)
sim_demo = plinko.simulate_trajectories(alpha=0.60, kappa=1.20, n_balls=100, n_rows=12, seed=2026)
print("Observed Mystery Histogram x_obs (13 bins):", sim_demo["histogram"].tolist())""",

    "s15-plinko-abc": """# Rejection ABC on the cached 50,000-simulation Galton-Board bank (no re-simulation!)
abc_demo = abc.run_plinko_abc(
    bundle["abc_bank"],
    bundle["primary_obs"]["x_obs"],
    epsilon=0.35,
    distance_kind="wasserstein",
    summary_kind="full_hist",
)
print(
    f"Accepted {abc_demo['n_accepted']} / {abc_demo['n_total']} "
    f"({100 * abc_demo['acceptance_rate']:.2f}%) | "
    f"ABC posterior mean (alpha, kappa) = {abc_demo['accepted_thetas'].mean(axis=0).round(3)}"
)""",

    "s30-unified-compare": """# Quantitative comparison of NPE, NLE, and NRE against the Exact DP benchmark on x_obs
ref_grid = bundle["obs_posteriors"][0]["exact"]
for method_name in ("NPE", "NLE", "NRE"):
    comp = validation.compare_posterior_grids(ref_grid, bundle["obs_posteriors"][0][method_name])
    print(
        f"{method_name}: E[alpha|x]={comp['mean_alpha']:.3f}±{comp['std_alpha']:.3f}, "
        f"E[kappa|x]={comp['mean_kappa']:.3f}±{comp['std_kappa']:.3f} | "
        f"TV={comp['tv_distance']:.3f}, KL={comp['kl_divergence']:.3f}"
    )""",

    "s31-reveal-truth": """# Exact dynamic-programming validation check: kappa=0 matches Binomial(R, sigma(alpha))
p_dp = plinko.exact_bin_probabilities(alpha=0.5, kappa=0.0, n_rows=12)
p_bin = plinko.binomial_bin_probabilities(p_right=1.0 / (1.0 + np.exp(-0.5)), n_rows=12)
print("Max |DP(kappa=0) - Binomial|:", f"{np.max(np.abs(p_dp - p_bin)):.2e}")""",
}


def _clean_slide_markdown(raw_md: str, is_title_slide: bool = False) -> str:
    """Format slide markdown headings to match Weeks 1–9 nbconvert conventions."""
    md = raw_md.strip()
    if is_title_slide:
        md = re.sub(
            r"^##\s*Simulation-Based Inference:\s*",
            "### ",
            md,
            count=1,
        )
        return md
    # Convert `### Slide N — Heading` -> `## Heading` and `### Backup BX:` -> `## Backup BX:`
    md = re.sub(r"^###\s+Slide\s+\d+\s*[—–-]\s*", "## ", md, flags=re.MULTILINE)
    md = re.sub(r"^###\s+(Backup\s+B\d+:)", r"## \1", md, flags=re.MULTILINE)
    return md


def build_notebook_object() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }

    cells = []

    setup_code = """from pathlib import Path
import sys
import warnings

warnings.simplefilter("ignore")

repo_root = Path.cwd() if (Path.cwd() / "course_tools").is_dir() else Path.cwd().parent
for _p in (repo_root, repo_root / "course_tools" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import bayesian_stats_course_tools
from sbi_lecture import abc, density_estimation, interactive_widgets as iw, neural_sbi, plinko, precompute, validation

# Load precomputed Galton-board simulation bank, checkpoints, and trained NPE/NLE/NRE models (<1s, offline)
bundle = precompute.load_lecture_bundle()"""
    c_setup = new_code_cell(setup_code)
    c_setup.metadata["slideshow"] = {"slide_type": "skip"}
    c_setup.metadata["tags"] = ["remove-cell"]
    cells.append(c_setup)

    tex_code = """title = "# Simulation-Based Inference"
# Print title and setup TeX defs for both KaTeX and MathJax (matches Weeks 1–9)
bayesian_stats_course_tools.misc.display_markdown_and_setup_tex(title)

import matplotlib.style
matplotlib.style.use("bayesian_stats_course_tools.light")"""
    c_tex = new_code_cell(tex_code)
    c_tex.metadata["slideshow"] = {"slide_type": "slide"}
    c_tex.metadata["tags"] = ["remove-input"]
    cells.append(c_tex)

    for idx, slide in enumerate(SLIDES):
        sid = str(slide["id"])
        is_first = idx == 0
        md_cell = new_markdown_cell(_clean_slide_markdown(str(slide["nb_markdown"]), is_title_slide=is_first))
        md_cell.metadata["slideshow"] = {"slide_type": "-" if is_first else "slide"}
        cells.append(md_cell)

        if "figure" in slide:
            fname, fcap, fmw = slide["figure"]  # type: ignore[misc]
            fc = new_code_cell(f"iw.render_figure_card({fname!r}, {fcap!r}, max_width_px={int(fmw)})")
            fc.metadata["slideshow"] = {"slide_type": "-"}
            fc.metadata["tags"] = ["remove-input"]
            cells.append(fc)

        if "demo_id" in slide:
            did = str(slide["demo_id"])
            dc = new_code_cell(f"iw.demo({did!r})")
            dc.metadata["slideshow"] = {"slide_type": "-"}
            dc.metadata["tags"] = ["remove-input"]
            cells.append(dc)

        if "notes" in slide and slide["notes"]:
            nc = new_markdown_cell(str(slide["notes"]).strip())
            nc.metadata["slideshow"] = {"slide_type": "notes"}
            nc.metadata["tags"] = ["remove-cell"]
            cells.append(nc)

        if sid in CODE_INSPECTION_CELLS:
            sc = new_code_cell(CODE_INSPECTION_CELLS[sid])
            sc.metadata["slideshow"] = {"slide_type": "skip"}
            sc.metadata["tags"] = ["remove-cell"]
            cells.append(sc)

    nb.cells = cells
    return nb


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "LECTURES" / "week10_simulation_based_inference.ipynb"
    nb = build_notebook_object()

    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()

    nbformat.write(nb, nb_path)
    print(f"Executed and wrote {nb_path} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
