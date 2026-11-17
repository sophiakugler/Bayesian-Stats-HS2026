# Week 10 Lecture & Exercise Package — Simulation-Based Inference (`sbi_lecture`)

**Course:** Bayesian Statistics and Data Analysis, ETH Zürich, HS2026  
**Lecture Title:** *Simulation-Based Inference: Learning from Simulations When Likelihoods Are Inaccessible*  
**Duration:** 90 minutes (34 principal slides + 18 backup slides = 52 slides total)  
**Main Lecture Notebook:** [`LECTURES/week10_simulation_based_inference.ipynb`](../LECTURES/week10_simulation_based_inference.ipynb)  
**Continuous-Scroll Interactive Slides:** [`slides/week10_simulation_based_inference.slides.html`](../slides/week10_simulation_based_inference.slides.html)  
**Student Exercise Notebook:** [`EXERCISES/week10_simulation_based_inference.ipynb`](../EXERCISES/week10_simulation_based_inference.ipynb)  
**Instructor Solution Notebook:** [`exercise_solutions/week10_simulation_based_inference_solutions.ipynb`](../exercise_solutions/week10_simulation_based_inference_solutions.ipynb)

---

## 1. Narrative Spine & Interactive Design

1. **Central Scientific Problem — The Mystery Galton Board (`α, κ`):**
   - Begins with the analytically tractable $\kappa = 0$ Binomial Galton board ($R = 12$ rows, $N = 100$ balls) and introduces state-dependent restoring stiffness $\kappa > 0$, where $P(\mathrm{right}\mid s_t,\alpha,\kappa)=\sigma(\alpha-\kappa s_t/\sqrt{R})$ marginalizes over $2^{12} = 4096$ latent paths.
   - Solves the **same Mystery Galton Board** step-by-step using **Rejection ABC**, **Neural Posterior Estimation (NPE)**, **Neural Likelihood Estimation (NLE)** (with live prior swapping), and **Neural Ratio Estimation (NRE)** (two-tower bilinear-interaction joint-vs-marginal classifier with multi-negative balanced BCE), explicitly contrasting **when and why to choose each method**.
   - Unveils the exact dynamic-programming likelihood recursion to benchmark ABC, NPE, NLE, and NRE, followed by Posterior Predictive Checks (PPC) and 200-experiment HPD Coverage & Simulation-Based Calibration (SBC).

2. **24 Live Movable `ipywidgets`-Style Slider & Training Widgets:**
   - Includes **3 dedicated Galton-Board Neural Training Widgets** (`wt_npe_plinko`, `wt_nle_plinko`, `wt_nre_plinko`) with `▶ Train` / `⏸ Pause` / `↺ Reset` buttons, epoch sliders, semantic `<progress>` bars, held-out validation loss curves, and live posterior evolution across simulation budgets and negative-sampling regimes.
   - Paired directly with the exercises in `EXERCISES/week10_simulation_based_inference.ipynb` and `exercise_solutions/week10_simulation_based_inference_solutions.ipynb`, as well as Backup Slides `B16–B18` in the lecture deck.

3. **Week 1–9 `nbconvert` Slide Styling + Smooth Continuous Scroll:**
   - `slides/week10_simulation_based_inference.slides.html` is exported directly from `LECTURES/week10_simulation_based_inference.ipynb` via `nbconvert.SlidesExporter` using the exact configuration from `course_tools/make_slides.sh` (`JupyterLab Light` + Reveal.js `"white"` theme + `course_tools/nbconvert_templates` MathJax 3 `tex-svg.js` + `remove-cell` / `remove-input` tags), enhanced with continuous vertical scrolling (`scroll-behavior: smooth`, `height: auto`, `overflow-y: auto`), native browser zoom (`Cmd +/-`), a bottom-right Reveal slide counter, and `PageUp`/`PageDown` slide stepping.

---

## 2. Rebuilding Assets, Slides, Notebooks & Running Tests

```bash
# Export compact JSON payload (trains/loads NPE, NLE, NRE & epoch trajectories)
python -m sbi_lecture.payload_builder

# Rebuild smooth continuous-scroll interactive HTML slide deck
python -m sbi_lecture.export_slides

# Rebuild and execute synchronized Jupyter lecture notebook
python -m sbi_lecture.build_notebook

# Rebuild and execute student exercise + instructor solution notebooks
python -m sbi_lecture.build_exercises

# Run full verification test suite (36 tests)
pytest tests -v
```

---

## 3. Attribution & License

- Preserves ~85–90% of the mathematical derivations, pedagogical progression, and core curriculum of **Tilman Tröster**, *Bayesian Statistical Methods — Simulation-Based Inference* ([`tilmantroester/bayesian_statistical_methods`](https://github.com/tilmantroester/bayesian_statistical_methods), MIT License); see [`PRESERVATION_MATRIX.md`](./PRESERVATION_MATRIX.md).
- Interactive animation mechanics and normalizing-flow/classifier visual ideas inspired by **Christoph Weniger**, *A Crash-Course in SBI* ([`cweniger/teaching-2606-ICTP-SAIFR`](https://github.com/cweniger/teaching-2606-ICTP-SAIFR), MIT License).
