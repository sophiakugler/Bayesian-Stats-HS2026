# Content Preservation Matrix — Week 10: Simulation-Based Inference

Maps every mandatory educational component of Tilman Tröster's original [`simulation_based_inference.ipynb`](https://github.com/tilmantroester/bayesian_statistical_methods/blob/main/lectures/simulation_based_inference.ipynb) (MIT License) and the Master Build Prompt storyboard to the Week 10 slides and interactive Galton-board widgets.

| Original Component | Treatment in Week 10 Lecture | Slide(s) & Interactive Widget |
|---|---|---|
| Definition of likelihood-intractable inference | Preserved & motivated via latent trajectory sum $p(x\mid\theta) = \sum_z p(x, z\mid\theta)$ on the Mystery Galton Board | Slides 1–3, 8 (`w01_forward_inverse`, `w02c_paths_challenge`) |
| Sampling from likelihood vs evaluating its density | Explicit 4-way distinction (evaluating vs sampling likelihoods & posteriors; MCMC/HMC requirements) | Slide 3 (`fig_three_paradigms.png`) |
| Historical arc of inverse probability, Monte Carlo, ABC & neural SBI | 1763 Bayes/Laplace $\to$ 1940s Ulam/von Neumann $\to$ 1984–2002 Rubin/Tavaré/Pritchard/Beaumont $\to$ 2016+ Neural SBI | Slides 4, 17, 34 |
| Mystery Galton / Plinko running example ($\alpha, \kappa$) | Central spine from Slide 2 to Slide 34: $\kappa=0$ Binomial, $\kappa>0$ state-dependent transitions, ABC, NPE, NLE, NRE, exact DP reveal, PPC & SBC | Slides 2, 5–9, 15–16, 25–33 (`w02_plinko_sim`, `w03_analytic_galton`, `w06`, `w07`, `w11`–`w15`) |
| Rejection ABC algorithm & $p_\varepsilon(\theta\mid x_{\mathrm{obs}})$ derivation | Preserved mathematically from first principles with indicator-kernel integral | Slide 10, Backup B2 |
| Original 1D $\chi^2(5)$ + $\mathcal{N}(\sqrt{\theta}, 1)$ joint scatter & slice | Preserved verbatim with interactive $\varepsilon$, $x_{\mathrm{obs}}$, and $N$ sliders on a fixed sample pool | Slides 11–13 (`w04_abc_1d`) |
| Tolerance clicker question & 3-source error decomposition | Preserved with click-to-reveal resolution separating tolerance bias, MC variance, and summary loss | Slide 13 |
| Correlated bivariate Gaussian ABC tolerance example | Preserved, corrected ($\mathbb{E}[\theta\mid x] = \mu_\theta + \rho\frac{\sigma_\theta}{\sigma_x}(x-\mu_x)$, $\mathrm{Var} = \sigma_\theta^2(1-\rho^2)$), dual-$\varepsilon$ comparison | Slide 14 (`w05_corr_gauss`), Backup B1 |
| Curse of dimensionality ($\propto \varepsilon^D$) & summary sufficiency | Demonstrated on Galton board comparing `Mean only` vs `Mean + Variance` vs `Full 13-bin histogram` | Slides 16–17 (`w07_plinko_summaries`), Backup B3 |
| KL divergence properties & forward-KL to NLL derivation | Complete mathematical derivation + mutual information connection | Slides 18–20 |
| 1D Gaussian mixture construction & Gamma(3, 1/3) fitting | Interactive 3-component GMM editor + saved training checkpoints across $N \in \{150,1000,4000\}, K \in \{1,3,5,8\}$ | Slides 21–22 (`w08_gmm_editor`, `w09_gmm_gamma`), Backup B4 |
| $L_1$ vs $L_2$ point regression vs conditional density (MDN) | Skewed $\mathrm{LogNormal}(s=\gamma x, \mathrm{loc}=2x)$ demonstration + Euler–Lagrange proofs | Slide 23 (`w10_loss_regression`), Backup B5–B6 |
| Conditional neural density estimation (MDNs & Normalising Flows) | Change-of-variables Jacobian + interactive 1D warping & live 4-layer 2D RealNVP coupling flow | Slide 24 (`w10b_flow_1d`), Backup B11–B12 (`wb_flow_2d`) |
| Amortized vs sequential inference | Prior-predictive bank $\mathcal{D} = \{(\theta_i, x_i)\}_{i=1}^N$ + zero-latency switching across 4 Galton boards | Slide 25 (`w11_amortized`), Backup B13 |
| Unified NPE, NLE, and NRE on Galton board + **Why choose which?** | Logit-MDN NPE, structured Multinomial NLE with live prior-swapping, classifier NRE with mandatory ratio proof, and decision matrix | Slides 26–30 (`w12a_npe_plinko`, `w12b_nle_prior`, `w13_nre_classifier`, `w12_unified_compare`), Backup B14 |
| Exact DP benchmark reveal, Posterior Predictive Checks & Calibration | Hidden $(\alpha_{\mathrm{true}}=0.6, \kappa_{\mathrm{true}}=1.2)$ reveal, PPC with misspecification toggle, and 200-run empirical HPD coverage & SBC | Slides 31–33 (`w14_ppc`, `w15_coverage`), Backup B15 |
| Neural-network layers, SGD/Adam, MLE vs MAP, epistemic ensembles | Condensed in main flow; preserved in full in Backup B7–B10 with live SGD trainer | Backup B7–B10 (`wb_mlp_sgd`) |
