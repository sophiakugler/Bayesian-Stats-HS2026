"""Master Build Lecture Storyboard — Part 2 (Slides 21–34 + Backup Slides B1–B15).

Covers:
  - Slides 21–25: GMMs, Gamma-to-GMM Fitting, L1/L2 vs Conditional Density, Flows & Amortization
  - Slides 26–30: Neural SBI on the Galton Board (NPE, NLE, NRE & Practical Decision Guide)
  - Slides 31–34: Exact DP Validation, Posterior Predictive Checks, Coverage/SBC & Conclusions
  - Backup B1–B15: Extended Mathematical Derivations, ML Foundations & Interactive Trainers
"""

from __future__ import annotations

SLIDES_PART2: list[dict[str, object]] = [
    # =========================================================================
    # Slide 21 — Gaussian Mixture Models (W08)
    # =========================================================================
    {
        "id": "s21-gmm-editor",
        "part": "density",
        "title": "Gaussian Mixture Models: Building Flexible Densities",
        "demo_id": "w08_gmm_editor",
        "html": r"""
<h2>Gaussian Mixture Models: Building Flexible Densities</h2>
<p>A single Gaussian is symmetric and unimodal. Weighted convex combinations of \(K\) Gaussians approximate arbitrary smooth densities while remaining analytically normalised and trivial to sample:</p>
<div class="box" style="text-align:center; padding:8px 14px;">
  \[ q_\phi(x) \;=\; \sum_{c=1}^K \pi_c\,\mathcal{N}(x;\, \mu_c, \sigma_c^2), \qquad \pi_c = \frac{e^{\eta_c}}{\sum_{j=1}^K e^{\eta_j}} > 0, \quad \sum_{c=1}^K \pi_c = 1, \quad \sigma_c = \exp(s_c) > 0 \]
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 21 — Gaussian Mixture Models: Building Flexible Densities

$$q_\phi(x) = \sum_{c=1}^K \pi_c\,\mathcal{N}(x;\, \mu_c, \sigma_c^2), \qquad \pi_c = \frac{e^{\eta_c}}{\sum_{j=1}^K e^{\eta_j}}, \quad \sigma_c = \exp(s_c) > 0$$

Adjust the component weights, means, and scales below to construct skewed and multimodal densities.""",
        "notes": "2.5 min | Move mu_1 and mu_2 apart to create bimodality, then push them closer with unequal weights to create skewness.",
    },

    # =========================================================================
    # Slide 22 — Learning a Density from Samples: Gamma(3, 1/3) (W09)
    # =========================================================================
    {
        "id": "s22-gmm-gamma",
        "part": "density",
        "title": "Learning a Non-Gaussian Density from Samples",
        "demo_id": "w09_gmm_gamma",
        "html": r"""
<h2>Learning a Non-Gaussian Density from Samples</h2>
<p>Draw \(x_1,\dots,x_N \sim \mathrm{Gamma}(\text{shape}=3, \text{scale}=\tfrac{1}{3})\) and fit a \(K\)-component Gaussian mixture by gradient descent on \(\mathcal{L}(\phi) = -\frac{1}{N}\sum_{i=1}^N \log q_\phi(x_i)\). Scrub through the optimisation checkpoints below:</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 22 — Learning a Non-Gaussian Density from Samples

Fitting a $K$-component GMM to $x_i \sim \mathrm{Gamma}(\text{shape}=3, \text{scale}=\frac{1}{3})$ by minimising empirical NLL. Compare $K = 1$ (underfits skewness) against $K = 5$, and $N = 150$ (finite-sample noise) against $N = 4000$.""",
        "notes": "3 min | Drag Checkpoint from iter 0 to iter 160, then switch K=1 vs K=5 and N=150 vs N=4000.",
    },

    # =========================================================================
    # Slide 23 — What Does a Machine-Learning Loss Actually Learn? (W10)
    # =========================================================================
    {
        "id": "s23-l1-l2-mdn",
        "part": "density",
        "title": "What Does a Machine-Learning Loss Actually Learn?",
        "demo_id": "w10_loss_regression",
        "html": r"""
<h2>What Does a Machine-Learning Loss Actually Learn?</h2>
<p>Why not simply train a standard neural regressor \(\hat{\theta}_\phi(x)\) on simulated pairs? Because point losses collapse the entire posterior distribution to a single summary number:</p>
<div style="display:flex; gap:10px; margin:6px 0 8px; flex-wrap:wrap;">
  <div class="box box-orange" style="flex:1 1 220px; margin:0; padding:8px 12px;">
    <strong>\(L_2\) Squared Error \(\mathbb{E}[(y - f(x))^2]\)</strong><br>
    <span style="font-size:0.88em;">Population optimum is the <strong>conditional mean</strong> \(f^\star_{L_2}(x) = \mathbb{E}[Y\mid x]\).</span>
  </div>
  <div class="box box-green" style="flex:1 1 220px; margin:0; padding:8px 12px;">
    <strong>\(L_1\) Absolute Error \(\mathbb{E}[|y - f(x)|]\)</strong><br>
    <span style="font-size:0.88em;">Population optimum is the <strong>conditional median</strong> \(f^\star_{L_1}(x) = \mathrm{median}(Y\mid x)\).</span>
  </div>
  <div class="box box-blue" style="flex:1 1 240px; margin:0; padding:8px 12px;">
    <strong>Conditional NLL \(-\mathbb{E}[\log q_\phi(y\mid x)]\)</strong><br>
    <span style="font-size:0.88em;">Learns the <strong>full conditional distribution</strong> \(p(y\mid x)\): tails, skewness, and modes!</span>
  </div>
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 23 — What Does a Machine-Learning Loss Actually Learn?

On the skewed conditional law $y \mid x \sim \mathrm{LogNormal}(s = \gamma x, \mathrm{loc} = 2x)$:
- **$L_2$ loss** learns only the conditional mean $\mathbb{E}[Y\mid x]$ (pulled upward by the heavy upper tail).
- **$L_1$ loss** learns only the conditional median $\mathrm{median}(Y\mid x)$.
- **Conditional NLL** learns the entire skewed conditional probability distribution $q_\phi(y\mid x) \approx p(y\mid x)$.""",
        "notes": "3 min | Drag Slice x from 0.2 to 1.4 and switch skewness gamma to 1.25 to make the separation of mean, median, and mode visually unmistakable.",
    },

    # =========================================================================
    # Slide 24 — Conditional Neural Density Estimation (W10b)
    # =========================================================================
    {
        "id": "s24-conditional-de",
        "part": "density",
        "title": "Conditional Neural Density Estimation",
        "demo_id": "w10b_flow_1d",
        "html": r"""
<h2>Conditional Neural Density Estimation</h2>
<p>Move from unconditional \(q_\phi(y)\) to conditional \(q_\phi(y \mid x)\) by letting a neural network map the conditioning input \(x\) to the parameters of a normalised density:</p>
<div style="display:flex; gap:14px; margin:6px 0 8px; flex-wrap:wrap;">
  <div class="box box-blue" style="flex:1 1 320px; margin:0;">
    <strong>1. Mixture Density Networks (Bishop 1994):</strong><br>
    <span style="font-size:0.9em;">Neural net maps \(x \mapsto \{\pi_c(x), \mu_c(x), \Sigma_c(x)\}_{c=1}^K\), ideal for low-dimensional parameter spaces like our 2D Galton board \((\alpha, \kappa)\).</span>
  </div>
  <div class="box box-green" style="flex:1 1 320px; margin:0;">
    <strong>2. Conditional Normalising Flows (Papamakarios et al. 2017):</strong><br>
    <span style="font-size:0.9em;">Push \(z \sim \mathcal{N}(0, I)\) through an \(x\)-conditioned bijection \(y = f_\phi(z; x)\), scaling cleanly to 10&ndash;100+ correlated parameters via \(q_\phi(y\mid x) = p_Z(f_\phi^{-1}(y; x))\,|\det J_{f_\phi^{-1}}|\).</span>
  </div>
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 24 — Conditional Neural Density Estimation

Two primary architectures map conditioning inputs $x$ to normalised conditional distributions $q_\phi(y\mid x)$:
1. **Mixture Density Networks (MDNs):** neural network outputs mixture logits, means, and Cholesky factors $\{\pi_c(x), \mu_c(x), L_c(x)\}_{c=1}^K$.
2. **Conditional Normalising Flows:** compose invertible $x$-conditioned transformations $y = f_\phi(z; x)$ on $z \sim \mathcal{N}(0, I)$ with exact change-of-variables density $q_\phi(y\mid x) = p_Z(f_\phi^{-1}(y; x))\left|\det \frac{\partial f_\phi^{-1}}{\partial y}\right|$.""",
        "notes": "2.5 min | Sweep Warp A(x) from 0 to 0.85 to show how a Gaussian morphs smoothly into a bimodal density, and past 1.0 to show why invertibility matters.",
    },

    # =========================================================================
    # Slide 25 — Turning Simulations into Reusable Amortized Inference (W11)
    # =========================================================================
    {
        "id": "s25-amortized",
        "part": "density",
        "title": "Turning Simulations into Reusable Amortized Inference",
        "demo_id": "w11_amortized",
        "html": r"""
<h2>Turning Simulations into Reusable Amortized Inference</h2>
<div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:6px;">
  <div class="box box-blue" style="flex:1 1 380px; margin:0; text-align:center;">
    <strong>Prior-predictive training bank:</strong>
    \[ \mathcal{D} = \bigl\{(\theta_i, x_i)\bigr\}_{i=1}^N, \qquad \theta_i \sim p(\theta), \quad x_i \sim p(x\mid\theta_i) \]
  </div>
  <div style="flex:1 1 360px; font-size:0.9em;">
    <strong>Amortized vs Sequential SBI:</strong> <em>Amortized</em> SBI pays simulation &amp; training cost once up front so any new \(x_{\mathrm{obs}}\) is inferred in milliseconds; <em>Sequential</em> SBI (SNPE/SNLE/SNRE) focuses simulations iteratively around a single high-value \(x_{\mathrm{obs}}\) when simulator calls are very expensive.
  </div>
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 25 — Turning Simulations into Reusable Amortized Inference

Generate a prior-predictive training dataset once:

$$\mathcal{D} = \bigl\{(\theta_i, x_i)\bigr\}_{i=1}^N, \qquad \theta_i = (\alpha_i, \kappa_i) \sim p(\theta), \quad x_i \sim p(x\mid\theta_i)$$

- **Amortized inference:** upfront simulation and neural training cost is reused across arbitrarily many future datasets $x_{\mathrm{obs}}$ (critical for survey catalogues, real-time gravitational-wave triggers, and 200-run calibration checks!).
- **Sequential SBI:** steers proposal rounds toward one specific $x_{\mathrm{obs}}$ when each simulation costs hours.""",
        "notes": "2.5 min | Switch across Experiments 1, 2, 3, and 4 in W11 to show zero-latency posterior evaluation without running a single new Galton-board simulation.",
    },

    # =========================================================================
    # Slide 26 — Neural Posterior Estimation (NPE)
    # =========================================================================
    {
        "id": "s26-npe-theory",
        "part": "neural",
        "title": "Neural Posterior Estimation: Direct Conditional Posteriors",
        "html": r"""
<h2>Neural Posterior Estimation: Direct Conditional Posteriors</h2>
<p>Bayes' theorem \(p(\theta\mid x) = \frac{p(x\mid\theta)p(\theta)}{p(x)}\) gives three neural targets. The most direct is <strong>Neural Posterior Estimation (NPE)</strong>, training \(q_\phi(\theta\mid x) \approx p(\theta\mid x)\) on \((\theta_i, x_i) \sim p(\theta)p(x\mid\theta)\):</p>
<div class="box box-blue" style="text-align:center;">
  \[ \mathcal{L}_{\mathrm{NPE}}(\phi) \;=\; -\frac{1}{N}\sum_{i=1}^N \log q_\phi(\theta_i \mid x_i) \;\approx\; \mathbb{E}_{p(x)}\!\Bigl[D_{\mathrm{KL}}\bigl(p(\theta\mid x) \;\big\|\; q_\phi(\theta\mid x)\bigr)\Bigr] + \mathrm{const} \]
</div>
<div style="display:flex; gap:14px; margin-top:10px; flex-wrap:wrap;">
  <div class="box" style="flex:1 1 300px; margin:0;">
    <strong>Respecting bounded prior support on our Galton board:</strong><br>
    <span style="font-size:0.9em;">Since \((\alpha, \kappa) \in [-1.5, 1.5]\times[0, 3]\), we fit a full-covariance 8-component MDN in unconstrained logit coordinates \(u = g(\theta)\) and include the exact Jacobian:
    \[ \log q_\phi(\theta\mid x) = \log q_\phi^{(U)}\!\bigl(g(\theta)\mid x\bigr) + \log\left|\det\frac{\partial g(\theta)}{\partial\theta}\right|. \]</span>
  </div>
  <div class="box box-green" style="flex:1 1 300px; margin:0;">
    <strong>When &amp; why to choose NPE:</strong>
    <ul style="font-size:0.89em; line-height:1.5; margin-top:4px;">
      <li><strong>Pros:</strong> Instant ancestral posterior sampling (\(\sim\mathrm{ms}\), no MCMC burn-in or mixing issues!); learns summary embeddings \(T_\phi(x)\) end-to-end on raw high-\(D\) data.</li>
      <li><strong>Tradeoff:</strong> Implicitly bakes in the training prior \(p(\theta)\); changing the prior later requires retraining or importance reweighting.</li>
    </ul>
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 26 — Neural Posterior Estimation (NPE)

NPE trains a conditional density estimator $q_\phi(\theta\mid x) \approx p(\theta\mid x)$ directly on joint draws $(\theta_i, x_i) \sim p(\theta)p(x\mid\theta)$:

$$\mathcal{L}_{\mathrm{NPE}}(\phi) = -\frac{1}{N}\sum_{i=1}^N \log q_\phi(\theta_i \mid x_i)$$

For bounded priors $(\alpha, \kappa) \in [-1.5, 1.5]\times[0, 3]$, transforming via logit $u = g(\theta)$ with exact Jacobian $\log|\det \partial g/\partial\theta|$ guarantees $100\%$ of posterior samples lie strictly inside the physical prior box.

- **Why choose NPE:** fastest possible inference at test time (direct i.i.d. posterior draws in milliseconds, no MCMC!) and natural end-to-end compression of high-dimensional $x$.
- **Limitation:** amortised for the fixed training prior $p(\theta)$.""",
        "notes": "3 min | Highlight both the logit-support Jacobian (a common practical pitfall!) and the core strength/weakness of NPE.",
    },

    # =========================================================================
    # Slide 27 — NPE on the Mystery Galton Board (W12a)
    # =========================================================================
    {
        "id": "s27-npe-plinko",
        "part": "neural",
        "title": "NPE Applied to the Mystery Galton Board",
        "demo_id": "w12a_npe_plinko",
        "html": r"""
<h2>NPE Applied to the Mystery Galton Board</h2>
<p>Condition our PyTorch <code>PlinkoNPEModel</code> (trained on \(24{,}000\) prior-predictive Galton-board simulations) on the Mystery Machine's observed histogram \(x_{\mathrm{obs}}\):</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 27 — NPE Applied to the Mystery Galton Board

Explore the learned 2D joint posterior $q_\phi(\alpha, \kappa \mid x_{\mathrm{obs}})$, ancestral posterior samples, and 1D marginals across all four Galton-board experiments.""",
        "notes": "2.5 min | Point out the positive correlation tilt between alpha and kappa captured cleanly by the full-covariance Cholesky mixture components.",
    },

    # =========================================================================
    # Slide 28 — Neural Likelihood Estimation (NLE) (W12b)
    # =========================================================================
    {
        "id": "s28-nle-plinko",
        "part": "neural",
        "title": "Neural Likelihood Estimation: Structured Surrogates and Prior Swapping",
        "demo_id": "w12b_nle_prior",
        "html": r"""
<h2>Neural Likelihood Estimation: Structured Surrogates and Prior Swapping</h2>
<div style="display:flex; gap:14px; align-items:stretch; flex-wrap:wrap; margin-bottom:6px;">
  <div style="flex:1 1 400px;">
    <p><strong>NLE</strong> reverses the conditioning to learn \(q_\psi(x\mid\theta) \approx p(x\mid\theta)\). For the Galton board, we exploit <strong>known Multinomial observation structure</strong>: a neural network predicts normalised bin probabilities \(\hat{\pi}_\psi(k\mid\theta)\) via softmax, giving an exact discrete log-likelihood:</p>
    <div class="box box-green" style="text-align:center; padding:6px 10px; margin:4px 0;">
      \[ \log q_\psi(x\mid\theta) = \log\frac{N!}{\prod_k n_k!} + \sum_{k=0}^R n_k \log \hat{\pi}_\psi(k\mid\theta) \]
    </div>
  </div>
  <div class="box" style="flex:1 1 320px; margin:0; font-size:0.88em;">
    <strong>When &amp; why to choose NLE:</strong><br>
    1. <strong>Change priors \(p'(\theta)\) freely</strong> after training without re-simulating!<br>
    2. <strong>Combine \(M\) independent datasets</strong> via \(\sum_{m=1}^M \log q_\psi(x^{(m)}\mid\theta) + \log p(\theta)\).<br>
    <em>Tradeoff:</em> requires MCMC / HMC / grid at inference time, and density estimation in \(x\)-space gets hard if \(\dim(x)\) is huge and unstructured.
  </div>
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 28 — Neural Likelihood Estimation (NLE)

NLE learns a normalised surrogate likelihood $q_\psi(x\mid\theta) \approx p(x\mid\theta)$, then obtains the posterior via MCMC/HMC or grid normalisation on $q_\psi(x_{\mathrm{obs}}\mid\theta)p(\theta)$.

On the Galton board, predicting softmax landing probabilities $\hat{\pi}_\psi(0\mid\theta),\dots,\hat{\pi}_\psi(R\mid\theta)$ inside the known Multinomial count law gives a **structured neural likelihood estimator**:

$$\log q_\psi(x\mid\theta) = \log N! - \sum_{k=0}^R \log(n_k!) + \sum_{k=0}^R n_k \log \hat{\pi}_\psi(k\mid\theta)$$

**Switch the prior dropdown below** to see NLE adapt to a new prior $p'(\alpha, \kappa)$ instantaneously without retraining!""",
        "notes": "3.5 min | Switch the Prior dropdown from Uniform to 'Gaussian calibration prior on alpha' and drag mu_alpha to show live why NLE beats NPE whenever priors vary across downstream studies!",
    },

    # =========================================================================
    # Slide 29 — Neural Ratio Estimation (NRE) (W13)
    # =========================================================================
    {
        "id": "s29-nre-plinko",
        "part": "neural",
        "title": "Neural Ratio Estimation: Inference via Binary Classification",
        "demo_id": "w13_nre_classifier",
        "html": r"""
<h2>Neural Ratio Estimation: Inference via Binary Classification</h2>
<div style="display:flex; gap:14px; align-items:stretch; flex-wrap:wrap; margin-bottom:6px;">
  <div style="flex:1 1 420px;">
    <p><strong>NRE</strong> trains a classifier \(D_\omega(\theta, x) = \sigma(f_\omega(\theta, x))\) on balanced pairs: \(y=1\) for genuine joint draws \((\theta, x) \sim p(\theta, x)\) vs \(y=0\) for shuffled independent pairs \((\theta, x') \sim p(\theta)p(x)\). At the binary cross-entropy optimum:</p>
    <div class="box box-purple" style="text-align:center; padding:6px 10px; margin:4px 0;">
      \[ D^\star(\theta, x) = \frac{p(\theta, x)}{p(\theta, x) + p(\theta)p(x)} \;\Longrightarrow\; \underbrace{\frac{D^\star(\theta, x)}{1 - D^\star(\theta, x)}}_{\exp(f^\star(\theta, x))} = \frac{p(\theta, x)}{p(\theta)p(x)} = \frac{p(x\mid\theta)}{p(x)} \equiv r(\theta, x) \]
    </div>
  </div>
  <div class="box" style="flex:1 1 300px; margin:0; font-size:0.88em;">
    <strong>When &amp; why to choose NRE:</strong><br>
    &bull; Needs <strong>no normalising flow or tractable Jacobian</strong>: any standard ResNet/Transformer/GNN classifier works!<br>
    &bull; Like NLE, supports <strong>prior swapping</strong> \(p(\theta\mid x_{\mathrm{obs}}) = p(\theta)r(\theta, x_{\mathrm{obs}})\), pooling i.i.d. observations, and frequentist likelihood-ratio tests, even when \(x\) is high-dimensional!
  </div>
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 29 — Neural Ratio Estimation (NRE)

Construct balanced training pairs:
- **Class $y = 1$ (joint):** $(\theta_i, x_i) \sim p(\theta, x) = p(\theta)p(x\mid\theta)$.
- **Class $y = 0$ (independent marginal product):** $(\theta_i, x_{\pi(i)}) \sim p(\theta)p(x)$.

Minimising binary cross-entropy yields Bayes-optimal probability $D^\star(\theta, x) = \frac{p(\theta,x)}{p(\theta,x) + p(\theta)p(x)}$, so the classifier logit $f^\star(\theta, x) = \log\frac{D^\star}{1-D^\star}$ directly estimates the log likelihood-to-evidence ratio:

$$\frac{D^\star(\theta, x)}{1 - D^\star(\theta, x)} = \frac{p(\theta, x)}{p(\theta)p(x)} = \frac{p(x\mid\theta)}{p(x)} = r(\theta, x) \qquad\Longrightarrow\qquad p(\theta\mid x_{\mathrm{obs}}) = p(\theta)\,r(\theta, x_{\mathrm{obs}})$$""",
        "notes": "3.5 min | Walk through the 2-line algebra of D*/(1-D*) carefully, then drag Probe alpha and Probe kappa on W13 to watch D_omega and r_omega respond live.",
    },

    # =========================================================================
    # Slide 30 — Three Neural Paradigms + ABC: Why Choose Which? (W12)
    # =========================================================================
    {
        "id": "s30-unified-compare",
        "part": "neural",
        "title": "ABC, NPE, NLE, and NRE: Why Choose Which Method?",
        "demo_id": "w12_unified_compare",
        "html": r"""
<h2>ABC, NPE, NLE, and NRE: Why Choose Which Method?</h2>
<table class="sbi-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>Learned Quantity</th>
      <th>Test-Time Inference</th>
      <th>Change Prior \(p'(\theta)\)?</th>
      <th>Pool \(M\) Runs \(x^{(1:M)}\)?</th>
      <th>Best Suited When&hellip;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong style="color:#d97706;">ABC</strong></td>
      <td>None (nonparametric)</td>
      <td>Filter / SMC-ABC</td>
      <td>Reweight / rerun</td>
      <td>Via summary \(S\)</td>
      <td>\(\dim S(x) \le 3\), fast simulator, zero-training sanity check</td>
    </tr>
    <tr>
      <td><strong style="color:#2563eb;">NPE</strong></td>
      <td>Posterior \(q_\phi(\theta\mid x)\)</td>
      <td><strong>Direct draw (\(\sim\mathrm{ms}\))</strong></td>
      <td>Locked to \(p(\theta)\)</td>
      <td>Needs set encoder</td>
      <td>Fast amortised sampling on 1000s of events or high-\(D\) raw \(x\)</td>
    </tr>
    <tr>
      <td><strong style="color:#059669;">NLE</strong></td>
      <td>Likelihood \(q_\psi(x\mid\theta)\)</td>
      <td>MCMC / HMC / VI</td>
      <td><strong>Yes (instant!)</strong></td>
      <td><strong>Yes (\(\sum_m \log q_\psi\))</strong></td>
      <td>Low/moderate \(\dim x\) (or structured counts) + varying priors</td>
    </tr>
    <tr>
      <td><strong style="color:#7c3aed;">NRE</strong></td>
      <td>Ratio \(r_\omega(\theta, x) = \frac{p(x\mid\theta)}{p(x)}\)</td>
      <td>MCMC / Nested / Grid</td>
      <td><strong>Yes (instant!)</strong></td>
      <td><strong>Yes (\(\sum_m \log r_\omega\))</strong></td>
      <td>High-\(D\) raw \(x\) <em>and</em> prior swapping / LR tests without flows</td>
    </tr>
  </tbody>
</table>
__DEMO__
""",
        "nb_markdown": r"""### Slide 30 — ABC, NPE, NLE, and NRE: Why Choose Which Method?

| Method | Learned Target | Test-Time Sampling | Swap Prior $p'(\theta)$? | Combine $M$ i.i.d. Datasets? | Primary Strength |
|---|---|---|---|---|---|
| **ABC** | None | Distance filter | Reweight/rerun | Via summary | Zero neural training; transparent baseline in $\le 3$D |
| **NPE** | $q_\phi(\theta\mid x)$ | **Direct i.i.d. ($\sim\mathrm{ms}$)** | Tied to $p(\theta)$ | Requires set net | Instant amortised sampling; no MCMC at test time |
| **NLE** | $q_\psi(x\mid\theta)$ | MCMC / HMC | **Yes** | **Yes ($\prod_m q_\psi$)** | Exact prior independence + structured data likelihoods |
| **NRE** | $r_\omega(\theta,x)=\frac{p(x\mid\theta)}{p(x)}$ | MCMC / Grid | **Yes** | **Yes ($\prod_m r_\omega$)** | Unconstrained classifier on high-$D$ $x$ + prior flexibility |

Toggle any subset of **NPE**, **NLE**, **NRE**, **ABC**, and **Exact** on the shared Mystery Galton Board below:""",
        "notes": "4 min | Key synthesis slide! Walk through when to pick NPE vs NLE vs NRE in real scientific workflows, then toggle contours on W12.",
    },

    # =========================================================================
    # Slide 31 — Did We Recover the Mystery Plinko Settings?
    # =========================================================================
    {
        "id": "s31-reveal-truth",
        "part": "validation",
        "title": "Did We Recover the Mystery Galton Board Settings?",
        "html": r"""
<h2>Did We Recover the Mystery Galton Board Settings?</h2>
<div style="display:flex; gap:16px; flex-wrap:wrap;">
  <div style="flex:1 1 400px;">
    <div class="box box-blue" style="margin-top:0;">
      <strong>Unveiling the hidden machine &amp; exact benchmark:</strong><br>
      Hidden parameters: <strong style="color:#dc2626;">\(\alpha_{\mathrm{true}} = 0.60,\; \kappa_{\mathrm{true}} = 1.20\)</strong> (red star in Slide 30).<br>
      Independent exact validation via 12-step dynamic programming (\(\pi_0(0)=1\)):
      \[ \pi_{t+1}(k) = \pi_t(k)\bigl[1 - p_R(t,k;\theta)\bigr] + \pi_t(k-1)\,p_R(t,k-1;\theta) \]
      \[ p(x\mid\theta) = \frac{N!}{\prod_{k=0}^R n_k!}\prod_{k=0}^R \pi_R(k\mid\theta)^{n_k} \]
    </div>
  </div>
  <div style="flex:1 1 360px;">
    <div class="box box-orange" style="margin-top:0;">
      <strong>Why doesn't the posterior peak sit dead-on \(\theta_{\mathrm{true}}\)?</strong>
      <ul style="font-size:0.9em; line-height:1.52; margin-top:6px;">
        <li><strong>Finite-data randomness (\(N=100\)):</strong> even the <em>exact</em> dynamic-programming posterior peaks slightly away from \((0.60, 1.20)\) because \(x_{\mathrm{obs}}\) is one noisy 100-ball draw!</li>
        <li><strong>Approximation accuracy:</strong> NPE, NLE, and NRE match the <em>exact posterior</em> closely (\(\mathrm{TV} \approx 0.06\text{--}0.09\)), including its positive \((\alpha, \kappa)\) degeneracy tilt, while fixed-\(\varepsilon\) ABC is visibly broader.</li>
      </ul>
    </div>
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 31 — Did We Recover the Mystery Galton Board Settings?

- **Hidden truth revealed:** $\alpha_{\mathrm{true}} = 0.60$, $\kappa_{\mathrm{true}} = 1.20$.
- **Exact dynamic-programming benchmark:** propagating $\pi_{t+1}(k) = \pi_t(k)[1 - p_R(t,k;\theta)] + \pi_t(k-1)p_R(t,k-1;\theta)$ from $\pi_0(0)=1$ yields exact Multinomial likelihoods for independent validation.
- **Crucial statistical distinction:** $\theta_{\mathrm{true}}$ lies well inside the $68\%$ credible region, *not* at the exact posterior mode—because $N = 100$ balls carry irreducible multinomial sampling uncertainty! An inference algorithm's job is to match $p(\theta\mid x_{\mathrm{obs}})$, never to magically guess $\theta_{\mathrm{true}}$ without uncertainty.""",
        "notes": "2.5 min | Emphasize that in real SBI we do NOT have a dynamic-programming oracle—which motivates Slides 32 and 33!",
    },

    # =========================================================================
    # Slide 32 — Posterior Predictive Checks (W14)
    # =========================================================================
    {
        "id": "s32-ppc",
        "part": "validation",
        "title": "Posterior Predictive Checks: Testing Model Adequacy",
        "demo_id": "w14_ppc",
        "html": r"""
<h2>Posterior Predictive Checks: Testing Model Adequacy</h2>
<p>When no exact likelihood exists, how do we validate inference? First, test whether simulated data from inferred parameters actually look like \(x_{\mathrm{obs}}\) via the <strong>posterior predictive distribution</strong> \(p(\tilde{x}\mid x_{\mathrm{obs}}) = \int p(\tilde{x}\mid\theta)\,q(\theta\mid x_{\mathrm{obs}})\,d\theta\):</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 32 — Posterior Predictive Checks (PPC)

Sample $\theta^{(s)} \sim q(\theta\mid x_{\mathrm{obs}})$, forward-simulate replicated 100-ball histograms $\tilde{x}^{(s)} \sim p(x\mid\theta^{(s)})$, and compare discrepancy statistics $T(\tilde{x}^{(s)})$ against $T(x_{\mathrm{obs}})$.

- **Parameter recovery** asks where $\theta$ sits *assuming the simulator family is right*.
- **Posterior predictive checks** detect **simulator misspecification** (check the *Corrupt $x_{\mathrm{obs}}$* box below!).""",
        "notes": "2.5 min | Toggle 'Corrupt x_obs (sticky-peg misspecification)' to show how PPC immediately flags a broken simulator!",
    },

    # =========================================================================
    # Slide 33 — Calibration Across Repeated Experiments (W15)
    # =========================================================================
    {
        "id": "s33-calibration",
        "part": "validation",
        "title": "Coverage and Simulation-Based Calibration",
        "demo_id": "w15_coverage",
        "html": r"""
<h2>Coverage and Simulation-Based Calibration</h2>
<p>Posterior predictive checks test one \(x_{\mathrm{obs}}\); they cannot guarantee that a \(68\%\) credible region contains \(\theta^\star\) \(68\%\) of the time! Draw independent validation pairs \((\theta^\star_j, x_j) \sim p(\theta)p(x\mid\theta)\) (disjoint from training) and check <strong>empirical HPD coverage</strong> and <strong>Simulation-Based Calibration (SBC)</strong> rank uniformity (Talts et al. 2018):</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 33 — Coverage and Simulation-Based Calibration

Across $J = 200$ strictly independent validation experiments $(\theta^\star_j, x_j) \sim p(\theta)p(x\mid\theta)$:
1. **Empirical HPD Coverage:** a calibrated $(1-\alpha)$ credible region must contain $\theta^\star_j$ in $(1-\alpha)$ of experiments (diagonal line).
2. **Simulation-Based Calibration (SBC):** 1D marginal PIT / rank statistics $u_j = Q_\phi(\theta^\star_j \mid x_j)$ must be $\mathcal{U}(0, 1)$ ($\cup$-shape = overconfident; $\cap$-shape = conservative/overdispersed, as seen in finite-$\varepsilon$ ABC!).""",
        "notes": "3 min | Show NPE/NLE/NRE hugging the diagonal while fixed-epsilon ABC sits above the diagonal (conservative), then switch to 'Synthetic miscalibration explorer' to show U-shaped overconfidence.",
    },

    # =========================================================================
    # Slide 34 — What Did We Learn? Conclusions & References
    # =========================================================================
    {
        "id": "s34-conclusions",
        "part": "validation",
        "title": "What Did We Learn? Conclusions and Frontiers",
        "figure": ("dingo_fig3.png", "DINGO (Dax et al. 2021): conditional normalising-flow NPE inferring all 15 binary black hole merger parameters for GW150914 in seconds.", 430),
        "html": r"""
<h2>What Did We Learn? Conclusions and Frontiers</h2>
<div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
  <div style="flex:1 1 430px;">
    <div class="box box-green" style="margin-top:0;">
      <strong>Answer to our opening question:</strong> <em>Yes.</em> Even when \(p(x\mid\theta) = \int p(x, z\mid\theta)\,dz\) cannot be evaluated, forward simulations \((\theta_i, x_i) \sim p(\theta)p(x\mid\theta)\) encode the exact joint distribution needed for rigorous Bayesian inference.
    </div>
    <ul style="font-size:0.89em; line-height:1.52;">
      <li><strong>Rejection ABC</strong> slices \(p(\theta, x)\) locally, trading tolerance bias against \(\varepsilon^D\) sample starvation and summary loss.</li>
      <li><strong>Forward KL \(\Longleftrightarrow\) Empirical NLL</strong> lets neural networks learn normalised distributions from samples without evaluating target densities.</li>
      <li><strong>NPE, NLE, and NRE</strong> target the posterior, likelihood, or ratio; choose among them by weighing test-time speed against prior flexibility and data dimension.</li>
      <li><strong>PPCs, Coverage &amp; SBC</strong> are mandatory before trusting any learned surrogate in science.</li>
    </ul>
    <p style="font-size:0.78em; color:#475569; margin-top:8px;">
      <strong>Key references:</strong> Rubin (1984); Tavar&eacute; et al. (1997); Pritchard et al. (1999); Beaumont et al. (2002); Papamakarios &amp; Murray (2016); Dinh et al. (2017); Talts et al. (2018); Papamakarios et al. (2019); Hermans et al. (2020); Cranmer, Brehmer &amp; Louppe (2020, <em>PNAS</em>); Dax et al. (2021).
    </p>
  </div>
  <div style="flex:0 0 380px;">
    __FIGURE__
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 34 — What Did We Learn? Conclusions and References

1. **Answer to our opening question:** A faithful forward simulator defines $p(\theta, x) = p(\theta)p(x\mid\theta)$ through samples, supporting full Bayesian parameter inference without explicit likelihood evaluations.
2. **ABC $\to$ Neural Density Estimation:** Minimising empirical NLL minimises forward KL divergence, utilizing *all* simulations rather than discarding $99\%$.
3. **NPE, NLE, NRE:** Choose **NPE** for millisecond amortised posterior sampling, **NLE** for structured likelihoods and changing priors, and **NRE** for classifier-based ratio inference on high-dimensional data with prior flexibility.
4. **Validation is non-negotiable:** Combine Posterior Predictive Checks (model adequacy) with empirical coverage and SBC (statistical calibration).

**Foundational Literature:**
- Tavaré et al. (1997), *Genetics* 145; Pritchard et al. (1999), *Mol. Biol. Evol.* 16; Beaumont et al. (2002), *Genetics* 162; Sisson, Fan & Beaumont (2018), *Handbook of ABC*.
- Papamakarios & Murray (2016), NeurIPS; Papamakarios et al. (2019), AISTATS; Hermans, Begy & Louppe (2020), ICML.
- Cranmer, Brehmer & Louppe (2020), *The Frontier of Simulation-Based Inference*, *PNAS* 117(48).
- Talts et al. (2018), *Validating Bayesian Inference Algorithms with SBC*, arXiv:1804.06788; Dax et al. (2021), *Phys. Rev. Lett.* 127.""",
        "notes": "2.5 min | Close the loop back to Slide 1 and point to DINGO gravitational-wave inference and cosmological large-scale structure.",
    },

    # =========================================================================
    # BACKUP SLIDES B1 – B15 (Preserving Extended Source Material, Section 10)
    # =========================================================================
    {
        "id": "b01-cond-gauss",
        "part": "backup",
        "title": "Backup B1: Conditional Bivariate Gaussian Derivation",
        "html": r"""
<h2>Backup B1: Conditional Bivariate Gaussian Derivation</h2>
<p>Let \(\begin{pmatrix}\theta \\ x\end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix}\mu_\theta \\ \mu_x\end{pmatrix}, \begin{pmatrix}\sigma_\theta^2 & \rho\sigma_\theta\sigma_x \\ \rho\sigma_\theta\sigma_x & \sigma_x^2\end{pmatrix}\right)\). Completing the square in the joint quadratic form:</p>
<div class="box" style="text-align:center;">
  \[ \frac{1}{1-\rho^2}\!\left[\tilde{\theta}^2 - 2\rho\tilde{\theta}\tilde{x} + \tilde{x}^2\right] \;=\; \frac{(\tilde{\theta} - \rho\tilde{x})^2}{1 - \rho^2} + \tilde{x}^2, \qquad \tilde{\theta} = \frac{\theta - \mu_\theta}{\sigma_\theta},\;\; \tilde{x} = \frac{x - \mu_x}{\sigma_x} \]
</div>
<p>Factoring \(p(\theta, x) = p(\theta\mid x)\,p(x)\) immediately yields \(\theta \mid x \sim \mathcal{N}\!\left(\mu_\theta + \rho\frac{\sigma_\theta}{\sigma_x}(x - \mu_x),\; \sigma_\theta^2(1 - \rho^2)\right)\).</p>
""",
        "nb_markdown": r"""---
## Backup Material (Extended Derivations & Machine Learning Foundations)

### Backup B1: Conditional Bivariate Gaussian Derivation

Completing the square on standardized $\tilde{\theta} = (\theta-\mu_\theta)/\sigma_\theta$ and $\tilde{x} = (x-\mu_x)/\sigma_x$:

$$\frac{\tilde{\theta}^2 - 2\rho\tilde{\theta}\tilde{x} + \tilde{x}^2}{1-\rho^2} = \frac{(\tilde{\theta} - \rho\tilde{x})^2}{1-\rho^2} + \tilde{x}^2 \implies \theta \mid x \sim \mathcal{N}\!\left(\mu_\theta + \rho\frac{\sigma_\theta}{\sigma_x}(x - \mu_x),\; \sigma_\theta^2(1 - \rho^2)\right)$$""",
    },

    {
        "id": "b02-abc-taylor",
        "part": "backup",
        "title": "Backup B2: ABC Small-Tolerance Asymptotics",
        "html": r"""
<h2>Backup B2: ABC Small-Tolerance Asymptotics</h2>
<p>Expanding \(p(s \mid \theta)\) to second order around \(s_{\mathrm{obs}} = S(x_{\mathrm{obs}})\) over a symmetric ball \(B_\varepsilon(s_{\mathrm{obs}})\):</p>
<div class="box" style="text-align:center;">
  \[ \mathbb{P}\bigl(\|S(X) - s_{\mathrm{obs}}\| < \varepsilon \mid \theta\bigr) \;=\; \int_{B_\varepsilon(s_{\mathrm{obs}})} p(s\mid\theta)\,ds \;=\; V_D \varepsilon^D \left[p(s_{\mathrm{obs}}\mid\theta) + \frac{\varepsilon^2}{2(D+2)}\Delta_s p(s_{\mathrm{obs}}\mid\theta) + \mathcal{O}(\varepsilon^4)\right] \]
</div>
<p>Thus leading ABC posterior bias scales as \(\mathcal{O}(\varepsilon^2)\), while acceptance rate scales as \(\mathcal{O}(\varepsilon^D)\): halving bias requires shrinking \(\varepsilon\) by \(\sqrt{2}\), which reduces acceptance by \(2^{D/2}\).</p>
""",
        "nb_markdown": r"""### Backup B2: ABC Small-Tolerance Asymptotics

Taylor-expanding $p(s\mid\theta)$ around $s_{\mathrm{obs}}$ shows $\mathbb{P}(\|S(X)-s_{\mathrm{obs}}\|<\varepsilon\mid\theta) = V_D\varepsilon^D\bigl[p(s_{\mathrm{obs}}\mid\theta) + \mathcal{O}(\varepsilon^2)\bigr]$: leading bias is $\mathcal{O}(\varepsilon^2)$ while acceptance probability drops as $\mathcal{O}(\varepsilon^D)$.""",
    },

    {
        "id": "b03-distances-sufficiency",
        "part": "backup",
        "title": "Backup B3: Distance Metrics and Fisher-Neyman Sufficiency",
        "html": r"""
<h2>Backup B3: Distance Metrics and Fisher-Neyman Sufficiency</h2>
<ul style="line-height:1.6;">
  <li><strong>1D Wasserstein (\(W_1\) / Earth Mover's Distance):</strong> \(W_1(\hat{p}, \hat{q}) = \sum_{k=0}^{R-1} |\hat{F}_p(k) - \hat{F}_q(k)|\) respects horizontal bin ordering on the Galton board, unlike binwise \(L_2\) or Total Variation \(\frac{1}{2}\sum_k |\hat{p}_k - \hat{q}_k|\).</li>
  <li><strong>Fisher&ndash;Neyman Factorisation:</strong> \(S(x)\) is sufficient iff \(p(x\mid\theta) = h(x)\,g_\theta(S(x))\). By the Pitman&ndash;Koopman&ndash;Darmois theorem, fixed-dimensional exact sufficient summaries exist generically only for exponential families; neural SBI learns <em>approximate Bayes-sufficient</em> summaries by maximising \(I(\Theta; T_\phi(X))\).</li>
</ul>
""",
        "nb_markdown": r"""### Backup B3: Distance Metrics and Sufficiency

- **1D Wasserstein distance** $W_1(\hat{p},\hat{q}) = \sum_{k=0}^{R-1} |\hat{F}_p(k) - \hat{F}_q(k)|$ respects bin geometry (shifting mass from bin 6 to 7 costs less than 6 to 12).
- **Fisher–Neyman sufficiency:** $p(x\mid\theta) = h(x)g_\theta(S(x)) \iff p(\theta\mid x) = p(\theta\mid S(x))$ for every prior $p(\theta)$.""",
    },

    {
        "id": "b04-gmm-param",
        "part": "backup",
        "title": "Backup B4: Numerically Stable GMM Parameterisation",
        "html": r"""
<h2>Backup B4: Numerically Stable GMM Parameterisation</h2>
<p>Unconstrained gradient descent optimizes raw logits \(\eta_c \in \mathbb{R}\), means \(\mu_c \in \mathbb{R}^d\), and Cholesky factors \(L_c\) (positive diagonal via \(\operatorname{softplus}\)), evaluating log-densities via the <strong>log-sum-exp identity</strong> to prevent underflow:</p>
<div class="box" style="text-align:center;">
  \[ \log q_\phi(\theta) \;=\; m + \log\sum_{c=1}^K \exp\bigl(\log\pi_c + \log\mathcal{N}(\theta;\mu_c, L_c L_c^\top) - m\bigr), \qquad m = \max_c \bigl[\log\pi_c + \log\mathcal{N}(\theta;\mu_c, L_c L_c^\top)\bigr] \]
</div>
""",
        "nb_markdown": r"""### Backup B4: Numerically Stable GMM Parameterisation

Using softmax logits $\eta_c$, positive Cholesky diagonals $L_{c,ii} = \mathrm{softplus}(\tilde{\ell}_{c,ii}) + \epsilon$, and `logsumexp` prevents numerical underflow in low-density tails during NLL training.""",
    },

    {
        "id": "b05-erm-regression",
        "part": "backup",
        "title": "Backup B5: Supervised Learning and Empirical Risk Minimisation",
        "html": r"""
<h2>Backup B5: Supervised Learning and Empirical Risk Minimisation</h2>
<p>Given i.i.d. pairs \((x_i, y_i) \sim p(x, y)\) and pointwise loss \(\ell(y, \hat{y})\), population risk \(\mathcal{R}(f) = \mathbb{E}_{p(x,y)}[\ell(y, f(x))]\) is approximated by empirical risk \(\hat{\mathcal{R}}_N(f) = \frac{1}{N}\sum_{i=1}^N \ell(y_i, f_\phi(x_i))\). In SBI, the simulator supplies an inexhaustible stream of fresh pairs \((\theta_i, x_i)\), mitigating classical small-\(N\) overfitting when simulated online.</p>
""",
        "nb_markdown": r"""### Backup B5: Supervised Learning and Empirical Risk Minimisation

Population risk $\mathcal{R}(f) = \mathbb{E}_{p(x,y)}[\ell(y, f(x))]$ is estimated by empirical risk $\hat{\mathcal{R}}_N(\phi) = \frac{1}{N}\sum_{i=1}^N \ell(y_i, f_\phi(x_i))$. Because $\theta_i \sim p(\theta), x_i \sim p(x\mid\theta_i)$ are synthetic, fresh simulation batches prevent memorisation.""",
    },

    {
        "id": "b06-l1-l2-proofs",
        "part": "backup",
        "title": "Backup B6: Variational Proofs of L2 Mean and L1 Median",
        "html": r"""
<h2>Backup B6: Variational Proofs of L2 Mean and L1 Median</h2>
<div style="display:flex; gap:14px; flex-wrap:wrap;">
  <div class="box box-orange" style="flex:1 1 300px; margin:0;">
    <strong>Squared Error (\(L_2\)):</strong>
    \[ \frac{\partial}{\partial \hat{y}}\int (y - \hat{y})^2 p(y\mid x)\,dy = -2\!\int (y - \hat{y})p(y\mid x)\,dy = 0 \]
    \[ \Longrightarrow \quad \hat{y}^\star_{L_2}(x) = \int y\,p(y\mid x)\,dy = \mathbb{E}[Y\mid x] \]
  </div>
  <div class="box box-green" style="flex:1 1 300px; margin:0;">
    <strong>Absolute Error (\(L_1\)):</strong>
    \[ \frac{\partial}{\partial \hat{y}}\!\left[\int_{-\infty}^{\hat{y}}(\hat{y}-y)p(y\mid x)dy + \int_{\hat{y}}^{\infty}(y-\hat{y})p(y\mid x)dy\right] = 0 \]
    \[ \Longrightarrow \quad \int_{-\infty}^{\hat{y}^\star} p(y\mid x)\,dy = \frac{1}{2} \;\Longleftrightarrow\; \hat{y}^\star_{L_1}(x) = \mathrm{median}(Y\mid x) \]
  </div>
</div>
""",
        "nb_markdown": r"""### Backup B6: Variational Proofs of $L_2$ Mean and $L_1$ Median

- **Squared loss:** $\frac{\partial}{\partial\hat{y}}\mathbb{E}[(Y-\hat{y})^2\mid x] = -2(\mathbb{E}[Y\mid x] - \hat{y}) = 0 \implies \hat{y}^\star = \mathbb{E}[Y\mid x]$.
- **Absolute loss:** Leibniz's rule gives $F_{Y\mid X}(\hat{y}^\star\mid x) - (1 - F_{Y\mid X}(\hat{y}^\star\mid x)) = 0 \implies F_{Y\mid X}(\hat{y}^\star\mid x) = \frac{1}{2}$.""",
    },

    {
        "id": "b07-mlp-layers",
        "part": "backup",
        "title": "Backup B7: Neural Network Layers and Activations",
        "html": r"""
<h2>Backup B7: Neural Network Layers and Activations</h2>
<p>A Multi-Layer Perceptron composes affine maps with smooth nonlinearities \(h^{(\ell)} = \psi(W^{(\ell)}h^{(\ell-1)} + b^{(\ell)})\). Common activations: \(\operatorname{ReLU}(u) = \max(0, u)\), \(\operatorname{SiLU}(u) = u\,\sigma(u)\) (smooth derivatives, preferred in density networks), and \(\tanh(u)\).</p>
""",
        "nb_markdown": r"""### Backup B7: Neural Network Layers and Activations

Composing affine layers $h^{(\ell)} = \psi(W^{(\ell)}h^{(\ell-1)} + b^{(\ell)})$ with nonlinear activations ($\mathrm{ReLU}$, $\mathrm{SiLU}$, $\tanh$) provides a universal function approximator mapping data embeddings to distribution parameters.""",
    },

    {
        "id": "b08-sgd-trainer",
        "part": "backup",
        "title": "Backup B8: Gradient-Based Optimisation Live",
        "demo_id": "wb_mlp_sgd",
        "html": r"""
<h2>Backup B8: Gradient-Based Optimisation Live</h2>
<p>Mini-batch stochastic gradient descent / Adam updates \(\phi_{t+1} = \phi_t - \eta\,\hat{g}_t\) using automatic differentiation (backpropagation). Press <strong>&#9654; Train</strong> to watch \(\phi = (W, b)\) learn both mean and heteroscedastic spread simultaneously:</p>
__DEMO__
""",
        "nb_markdown": r"""### Backup B8: Gradient-Based Optimisation Live

Interactive stochastic gradient descent trainer minimising Gaussian negative log-likelihood on heteroscedastic data.""",
    },

    {
        "id": "b09-mle-map",
        "part": "backup",
        "title": "Backup B9: Maximum Likelihood vs MAP and Regularisation",
        "html": r"""
<h2>Backup B9: Maximum Likelihood vs MAP and Regularisation</h2>
<p>Placing a prior \(p(\phi)\) over network weights turns NLL minimisation into <strong>Maximum A Posteriori (MAP)</strong> estimation:</p>
<div class="box" style="text-align:center;">
  \[ \phi_{\mathrm{MAP}} = \arg\min_\phi \Bigl\{-\sum_{i=1}^N \log q_\phi(\theta_i\mid x_i) - \log p(\phi)\Bigr\} \]
</div>
<ul style="font-size:0.92em;">
  <li>Gaussian prior \(\phi \sim \mathcal{N}(0, \tau^2 I)\) \(\Longleftrightarrow\) <strong>\(L_2\) weight decay</strong> \(\frac{1}{2\tau^2}\|\phi\|_2^2\).</li>
  <li>Laplace prior \(p(\phi) \propto \exp(-\lambda\|\phi\|_1)\) \(\Longleftrightarrow\) <strong>\(L_1\) sparsity penalty</strong> \(\lambda\|\phi\|_1\).</li>
</ul>
""",
        "nb_markdown": r"""### Backup B9: Maximum Likelihood vs MAP and Regularisation

Adding $-\log p(\phi)$ to the empirical NLL yields MAP estimation: an isotropic Gaussian prior on weights corresponds to $L_2$ weight decay, while a Laplace prior corresponds to $L_1$ regularisation.""",
    },

    {
        "id": "b10-epistemic-weights",
        "part": "backup",
        "title": "Backup B10: Epistemic Network Uncertainty and Ensembles",
        "html": r"""
<h2>Backup B10: Epistemic Network Uncertainty and Ensembles</h2>
<p>With a finite simulation budget \(\mathcal{D}\), network weights \(\phi\) carry epistemic uncertainty \(p(\phi\mid\mathcal{D})\). Averaging over a deep ensemble \(\{\phi^{(m)}\}_{m=1}^M\) approximates the full predictive posterior:</p>
<div class="box" style="text-align:center;">
  \[ \bar{q}(\theta \mid x_{\mathrm{obs}}, \mathcal{D}) \;=\; \int q_\phi(\theta\mid x_{\mathrm{obs}})\,p(\phi\mid\mathcal{D})\,d\phi \;\approx\; \frac{1}{M}\sum_{m=1}^M q_{\phi^{(m)}}(\theta\mid x_{\mathrm{obs}}) \]
</div>
<p>Ensembling prevents overconfident posteriors in sparsely simulated regions (Hermans et al. 2022).</p>
""",
        "nb_markdown": r"""### Backup B10: Epistemic Network Uncertainty and Deep Ensembles

Marginalising over finite-budget weight uncertainty via deep ensembles $\bar{q}(\theta\mid x_{\mathrm{obs}}) = \frac{1}{M}\sum_{m=1}^M q_{\phi^{(m)}}(\theta\mid x_{\mathrm{obs}})$ improves calibration against overconfidence (Hermans et al. 2022).""",
    },

    {
        "id": "b11-mdn-details",
        "part": "backup",
        "title": "Backup B11: Conditional Mixture Density Network Architecture",
        "html": r"""
<h2>Backup B11: Conditional Mixture Density Network Architecture</h2>
<p>In <code>PlinkoNPEModel</code>, raw 13-bin histograms \(x\) are converted to normalised bin frequencies plus standardised first/second moments \((x/N, \tilde{\mu}_x, \tilde{\sigma}_x^2) \in \mathbb{R}^{15}\), encoded by a 3-layer \(\operatorname{SiLU}\) MLP (\(H=128\)), and projected to \(K=8\) 2D full-covariance Gaussian components (\(8\) logits, \(16\) means, \(24\) Cholesky entries).</p>
""",
        "nb_markdown": r"""### Backup B11: Conditional Mixture Density Network Architecture

`PlinkoNPEModel` maps 15-dimensional histogram features through a 3-layer `SiLU` encoder to $K=8$ 2D Cholesky-parameterised Gaussian components in logit-prior space.""",
    },

    {
        "id": "b12-flows-live",
        "part": "backup",
        "title": "Backup B12: Affine-Coupling Normalising Flows Live",
        "demo_id": "wb_flow_2d",
        "html": r"""
<h2>Backup B12: Affine-Coupling Normalising Flows Live</h2>
<p>RealNVP coupling layers split \(z = (z_A, z_B)\), keeping \(z'_A = z_A\) and applying \(z'_B = z_B \odot e^{s_\phi(z_A)} + t_\phi(z_A)\) so \(\log|\det J| = \sum s_\phi(z_A)\) costs \(\mathcal{O}(d)\). Press <strong>&#9654; Train flow</strong> to morph a 2D Gaussian into a curved banana density:</p>
__DEMO__
""",
        "nb_markdown": r"""### Backup B12: Affine-Coupling Normalising Flows Live

Four alternating RealNVP affine-coupling layers trained live in the browser via exact negative log-likelihood on a curved 2D banana distribution.""",
    },

    {
        "id": "b13-sequential-sbi",
        "part": "backup",
        "title": "Backup B13: Sequential vs Amortized SBI",
        "html": r"""
<h2>Backup B13: Sequential vs Amortized SBI</h2>
<p>When inferring parameters for <em>one</em> expensive target \(x_{\mathrm{obs}}\), drawing \(\theta_i \sim \tilde{p}_r(\theta)\) from a narrower proposal concentrator around \(p(\theta\mid x_{\mathrm{obs}})\) boosts sample efficiency by orders of magnitude, requiring proposal correction:</p>
<div class="box" style="text-align:center;">
  \[ \tilde{p}(\theta\mid x) \;=\; p(\theta\mid x)\,\frac{\tilde{p}_r(\theta)}{p(\theta)}\,\frac{p(x)}{\tilde{p}_r(x)} \]
</div>
<p><strong>SNLE</strong> and <strong>SNRE</strong> need no importance-weight correction because \(p(x\mid\theta)\) is independent of the parameter proposal distribution!</p>
""",
        "nb_markdown": r"""### Backup B13: Sequential vs Amortized SBI

Sequential rounds draw $\theta \sim \tilde{p}_r(\theta)$ near the current posterior for a single $x_{\mathrm{obs}}$. **SNLE** learns $p(x\mid\theta)$ directly on any proposal support without atomic/truncation corrections, whereas **SNPE** must adjust for $\tilde{p}_r(\theta) \neq p(\theta)$.""",
    },

    {
        "id": "b14-nre-priors",
        "part": "backup",
        "title": "Backup B14: NRE with Arbitrary Class Priors",
        "html": r"""
<h2>Backup B14: NRE with Arbitrary Class Priors</h2>
<p>If each positive joint pair (\(y=1\), prior \(\pi_1 = \frac{1}{1+K}\)) is contrasted against \(K\) independent negative pairs (\(y=0\), prior \(\pi_0 = \frac{K}{1+K}\)), Bayes' rule gives:</p>
<div class="box" style="text-align:center;">
  \[ \frac{D^\star(\theta, x)}{1 - D^\star(\theta, x)} \;=\; \frac{\pi_1}{\pi_0}\,\frac{p(\theta, x)}{p(\theta)p(x)} \;=\; \frac{1}{K}\,r(\theta, x) \quad\Longrightarrow\quad \log r(\theta, x) \;=\; \operatorname{logit}\!\bigl(D^\star(\theta, x)\bigr) + \log K \]
</div>
""",
        "nb_markdown": r"""### Backup B14: NRE with Arbitrary Class Priors

With $K$ negative marginal draws per positive joint draw ($\pi_1/\pi_0 = 1/K$), the optimal classifier logit shifts by $-\log K$, so $\log r(\theta, x) = f^\star(\theta, x) + \log K$.""",
    },

    {
        "id": "b15-open-questions",
        "part": "backup",
        "title": "Backup B15: Calibration Blind Spots, Misspecification, and Exercises",
        "html": r"""
<h2>Backup B15: Calibration Blind Spots, Misspecification, and Exercises</h2>
<ul style="line-height:1.6;">
  <li><strong>Necessary vs sufficient calibration:</strong> outputting the prior \(q(\theta\mid x) = p(\theta)\) or a lossy-summary posterior \(p(\theta\mid S(x))\) yields <em>perfectly flat</em> global SBC rank histograms! Combine SBC with HPD coverage, TARP / random projections (Lemos et al. 2023), and sharp posterior-contraction checks.</li>
  <li><strong>Simulator misspecification:</strong> when \(x_{\mathrm{obs}}\) lies out-of-distribution relative to simulated summaries \(T_\phi(x)\), neural extrapolations can become silently overconfident; always pair SBI with PPCs and summary-space anomaly checks.</li>
  <li><strong>Hands-on Exercise Notebook:</strong> in <code>EXERCISES/week10_simulation_based_inference.ipynb</code>, solve the Mystery Galton Board step-by-step using <strong>ABC, NPE, NLE, and NRE</strong>, and inspect their live training trajectories in Slides B16&ndash;B18 below.</li>
</ul>
""",
        "nb_markdown": r"""### Backup B15: Calibration Blind Spots, Misspecification, and Exercise Session

- **Calibration is necessary, not sufficient:** $q(\theta\mid x) = p(\theta)$ passes global SBC trivially! Pair SBC with TARP / Cramér–Wold projections, local stratification, and posterior shrinkage checks.
- **Model misspecification:** always audit out-of-distribution summary statistics and Posterior Predictive Checks before scientific interpretation.
- **Companion Exercise Notebook:** see `EXERCISES/week10_simulation_based_inference.ipynb` (and `exercise_solutions/week10_simulation_based_inference_solutions.ipynb`) to implement ABC, NPE, NLE, and NRE on the Mystery Galton Board.""",
    },

    {
        "id": "b16-train-npe-plinko",
        "part": "backup",
        "title": "Backup B16: Live NPE Training on the Galton Board",
        "demo_id": "wt_npe_plinko",
        "html": r"""
<h2>Backup B16: Live NPE Training on the Galton Board</h2>
<p>Press <strong>&#9654; Train</strong> to watch the conditional posterior \(q_\phi(\alpha, \kappa \mid x_{\mathrm{obs}})\) evolve across epochs under forward-KL NLL, and switch simulation budgets (\(M = 36{,}000\) vs \(800\)) to observe overfitting on held-out validation pairs:</p>
__DEMO__
""",
        "nb_markdown": r"""### Backup B16: Live NPE Training on the Galton Board

Interactive epoch-by-epoch training explorer for `PlinkoNPEModel` across simulation budgets $M \in \{800, 4{,}000, 36{,}000\}$ and mixture capacities $K \in \{1, 8\}$.""",
    },

    {
        "id": "b17-train-nle-plinko",
        "part": "backup",
        "title": "Backup B17: Live NLE Training & Prior Swapping on the Galton Board",
        "demo_id": "wt_nle_plinko",
        "html": r"""
<h2>Backup B17: Live NLE Training &amp; Prior Swapping on the Galton Board</h2>
<p>Press <strong>&#9654; Train</strong> to watch the 13-bin softmax likelihood \(q_\psi(x \mid \alpha, \kappa)\) converge from uniform \(1/13\) probabilities onto the true simulator distribution while swapping priors \(p'(\theta)\) at any epoch:</p>
__DEMO__
""",
        "nb_markdown": r"""### Backup B17: Live NLE Training & Prior Swapping on the Galton Board

Interactive epoch-by-epoch training explorer for `PlinkoNLEModel` showing learned 13-bin probabilities, induced posterior contours under live prior swapping, and multinomial NLL convergence.""",
    },

    {
        "id": "b18-train-nre-plinko",
        "part": "backup",
        "title": "Backup B18: Live NRE Classifier Training on the Galton Board",
        "demo_id": "wt_nre_plinko",
        "html": r"""
<h2>Backup B18: Live NRE Classifier Training on the Galton Board</h2>
<p>Press <strong>&#9654; Train</strong> to watch binary cross-entropy classification drop from \(\ln 2 \approx 0.693\) and compare multi-negative contrastive training (\(K_{\mathrm{neg}}=24\)) against single-negative (\(K_{\mathrm{neg}}=1\)) convergence:</p>
__DEMO__
""",
        "nb_markdown": r"""### Backup B18: Live NRE Classifier Training on the Galton Board

Interactive epoch-by-epoch training explorer for `PlinkoNREModel` comparing multi-negative ($K_{\mathrm{neg}}=24$), single-negative ($K_{\mathrm{neg}}=1$), and small-sample ($M=1{,}200$) binary classification regimes.""",
    },
]
