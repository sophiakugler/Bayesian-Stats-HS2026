# Assets for Week 10: Simulation-Based Inference

Offline precomputed simulation banks, PyTorch checkpoints, and interactive widget payloads supporting `LECTURES/week10_simulation_based_inference.ipynb`, `slides/week10_simulation_based_inference.slides.html`, and the `sbi_lecture` verification suite.

## Contents

- `interactive_payload.json` — Compact client-side numerical payload powering all 21 `ipywidgets`-style interactive slider widgets across the notebook and HTML slide deck (4 benchmark Galton-board experiments, $41 \times 41$ exact DP / NPE / NLE / NRE posteriors, raw NLE log-likelihoods and softmax bin-probability table for live prior-swapping, raw NRE classifier logits, 3,000-draw Galton ABC subbank, $\mathrm{Gamma}(3, 1/3)$ GMM checkpoints, skewed LogNormal $L_1$/$L_2$/MDN fits, and 200-experiment HPD coverage & 1D SBC PIT ranks).
- `plinko_abc_bank.npz` — 50,000 prior-predictive Galton/Plinko simulations ($R=12$, $N=100$, seed `1010`).
- `gmm_gamma_checkpoints.npz` — Optimization trajectories fitting $K \in \{1, 3, 5, 8\}$ Gaussian mixtures to $\mathrm{Gamma}(a=3, \text{scale}=1/3)$ samples across $N \in \{150, 1000, 4000\}$.
- `regression_mdn_demo.npz` — Trained $L_2$ MLP, $L_1$ MLP, and Conditional Mixture Density Network checkpoints.
- `neural_sbi_models.pt` — Reproducible PyTorch state dictionaries for `PlinkoNPEModel`, `PlinkoNLEModel`, and `PlinkoNREModel`.
- `validation_coverage.npz` — Independent empirical 2D HPD coverage and 1D Simulation-Based Calibration (SBC) ranks across 200 validation experiments.
- `figures/` — Supplementary conceptual diagrams and real-world scientific SBI illustrations.
- `plotly.min.js` — Local offline copy of Plotly.js v2.35.0.

## Provenance & License

- Mathematical curriculum and baseline derivations (~85–90%) preserved from **Tilman Tröster**, *Bayesian Statistical Methods — Simulation-Based Inference* (`tilmantroester/bayesian_statistical_methods`, MIT License).
- Interactive animation mechanics inspired by **Christoph Weniger**, *A Crash-Course in SBI* (`cweniger/teaching-2606-ICTP-SAIFR`, MIT License).
