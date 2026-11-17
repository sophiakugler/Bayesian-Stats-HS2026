"""Master Build Lecture Storyboard — Part 1 (Slides 1–20).

Follows the exact 34-slide progression of the Master Build Prompt in clean lecturer voice:
  - Slides 1–4: Revisiting Bayesian Inverse Problems & Historical Arc
  - Slides 5–9: The Mystery Galton Board / Plinko Machine
  - Slides 10–17: Approximate Bayesian Computation (1D Chi2, Correlated Gaussian, Galton ABC & Summaries)
  - Slides 18–20: Density Estimation, KL Divergence & Empirical NLL Derivation
"""

from __future__ import annotations

SLIDES_PART1: list[dict[str, object]] = [
    # =========================================================================
    # Slide 1 — Title Slide
    # =========================================================================
    {
        "id": "s01-title",
        "part": "intro",
        "title": "Simulation-Based Inference",
        "html": r"""
<div style="text-align:center; padding:26px 12px;">
  <p style="font-size:0.86em; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#2563eb; margin:0 0 10px;">
    ETH Z&uuml;rich &middot; Bayesian Statistics and Data Analysis &middot; HS2026 &middot; Week 10
  </p>
  <h1 style="font-size:2.05em; margin:0 0 12px; line-height:1.22;">
    Simulation-Based Inference:<br>When We Can Simulate Nature, but Cannot Evaluate Its Likelihood
  </h1>
  <div style="width:180px; height:4px; background:linear-gradient(90deg,#2563eb,#dc2626); margin:16px auto;"></div>
  <div class="box box-blue" style="max-width:740px; margin:18px auto 12px; text-align:center; font-size:1.05em;">
    <strong>Central question for today:</strong><br>
    <em>If we can simulate an experiment faithfully on a computer, can we infer the hidden parameters of that experiment from observed data alone?</em>
  </div>
  <p style="font-size:0.80em; color:#64748b; margin-top:14px;">
    Based on Tilman Tr&ouml;ster's ETH Z&uuml;rich SBI lecture (MIT License), with visual extensions inspired by Christoph Weniger (ICTP-SAIFR).
  </p>
</div>
""",
        "nb_markdown": r"""## Simulation-Based Inference: When We Can Simulate Nature, but Cannot Evaluate Its Likelihood

**ETH Zürich · Bayesian Statistics and Data Analysis · HS2026 (Week 10)**

> **Central question for today:** *If we can simulate an experiment faithfully on a computer, can we infer the hidden parameters of that experiment from observed data alone?*

*Based on Tilman Tröster's ETH Zürich SBI lecture (`bayesian_statistical_methods`, MIT License), with interactive extensions inspired by Christoph Weniger (`teaching-2606-ICTP-SAIFR`, MIT License).*""",
        "notes": "2 min | Pose the opening question without answering it yet. Remind students that Weeks 5-9 relied on evaluating L(theta) = p(x_obs | theta); today asks what happens when we only have forward simulation code.",
    },

    # =========================================================================
    # Slide 2 — Scientific Inference Is an Inverse Problem (W01)
    # =========================================================================
    {
        "id": "s02-inverse-problem",
        "part": "intro",
        "title": "Scientific Inference Is an Inverse Problem",
        "demo_id": "w01_forward_inverse",
        "html": r"""
<h2>Scientific Inference Is an Inverse Problem</h2>
<div class="box" style="text-align:center; padding:10px 14px;">
  \[ \underbrace{\theta \;\xrightarrow{\;\text{Forward simulator}\;}\; x \sim p(x\mid\theta)}_{\textbf{Forward prediction: }\text{known parameters }\to\text{ synthetic outcomes}}
     \qquad\Longleftrightarrow\qquad
     \underbrace{x_{\mathrm{obs}} \;\xrightarrow{\;\text{Bayesian inference}\;}\; p(\theta\mid x_{\mathrm{obs}})}_{\textbf{Inverse problem: }\text{one noisy dataset }\to\text{ plausible parameters}} \]
</div>
<p>Switch between <strong>Forward</strong> and <strong>Inverse</strong> modes below to distinguish generating synthetic data from inferring unknown causes:</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 2 — Scientific Inference Is an Inverse Problem

$$\underbrace{\theta \;\xrightarrow{\;\text{Forward simulator}\;}\; x \sim p(x\mid\theta)}_{\textbf{Forward prediction}} \qquad\Longleftrightarrow\qquad \underbrace{x_{\mathrm{obs}} \;\xrightarrow{\;\text{Bayesian inference}\;}\; p(\theta\mid x_{\mathrm{obs}})}_{\textbf{Bayesian inverse problem}}$$

- **Forward prediction ($\theta \to x$):** specify physical parameters $\theta$ and execute the mechanistic model to generate synthetic data $x$.
- **Parameter inference ($x_{\mathrm{obs}} \to \theta$):** given a single noisy experimental realization $x_{\mathrm{obs}}$, determine the full posterior distribution over all parameter configurations $\theta$ compatible with $x_{\mathrm{obs}}$.""",
        "notes": "3 min | Demonstrate W01 in Forward mode first, then flip to Inverse mode and drag alpha and kappa to try matching the dark Mystery Histogram.",
    },

    # =========================================================================
    # Slide 3 — What Does Bayesian Inference Require?
    # =========================================================================
    {
        "id": "s03-bayes-requires",
        "part": "intro",
        "title": "What Does Bayesian Inference Require?",
        "figure": ("fig_three_paradigms.png", "Symbolic conjugate updates (Weeks 1–4), explicit likelihood evaluation via MCMC/HMC (Weeks 5–9), and implicit simulation-based inference (Week 10).", 720),
        "html": r"""
<h2>What Does Bayesian Inference Require?</h2>
<div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
  <div style="flex:1 1 420px;">
    <div class="box box-blue" style="text-align:center; margin-top:0;">
      \[ p(\theta\mid x_{\mathrm{obs}}) \;=\; \frac{p(x_{\mathrm{obs}}\mid\theta)\,p(\theta)}{p(x_{\mathrm{obs}})} \]
    </div>
    <p>Keep four distinct mathematical operations strictly separate:</p>
    <ul style="font-size:0.92em; line-height:1.55;">
      <li><strong>Evaluating a likelihood \(p(x_{\mathrm{obs}}\mid\theta)\):</strong> computing a scalar density number at fixed \(x_{\mathrm{obs}}\).</li>
      <li><strong>Sampling from a likelihood \(x \sim p(x\mid\theta)\):</strong> running forward simulation code with random draws.</li>
      <li><strong>Evaluating an unnormalised posterior \(\tilde{p}(\theta\mid x_{\mathrm{obs}}) = p(x_{\mathrm{obs}}\mid\theta)p(\theta)\):</strong> what Grid, Metropolis&ndash;Hastings, HMC (plus \(\nabla_\theta \log L\)), and Nested Sampling query at every step!</li>
      <li><strong>Sampling from a posterior \(\theta^{(s)} \sim p(\theta\mid x_{\mathrm{obs}})\):</strong> what expectations and credible regions actually need.</li>
    </ul>
  </div>
  <div style="flex:1 1 380px;">
    __FIGURE__
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 3 — What Does Bayesian Inference Require?

$$p(\theta\mid x_{\mathrm{obs}}) = \frac{p(x_{\mathrm{obs}}\mid\theta)\,p(\theta)}{p(x_{\mathrm{obs}})}$$

Distinguish four computational operations:
1. **Evaluating a likelihood density** $p(x_{\mathrm{obs}}\mid\theta)$ at a proposed $\theta$.
2. **Sampling from a likelihood** $x \sim p(x\mid\theta)$ by running a stochastic simulator.
3. **Evaluating an unnormalised posterior** $p(x_{\mathrm{obs}}\mid\theta)p(\theta)$ — required at *every* iteration of Grid evaluation, Metropolis–Hastings, Hamiltonian Monte Carlo (along with $\nabla_\theta\log p(x_{\mathrm{obs}}\mid\theta)$), and Nested Sampling.
4. **Sampling from the posterior** $\theta^{(s)} \sim p(\theta\mid x_{\mathrm{obs}})$.

When $p(x_{\mathrm{obs}}\mid\theta)$ cannot be evaluated in closed form or tractable quadrature, every conventional likelihood-based sampler stalls.""",
        "notes": "3 min | Emphasize the crucial distinction: having code that samples x ~ p(x|theta) is NOT the same as having a formula for the scalar density p(x_obs|theta).",
    },

    # =========================================================================
    # Slide 4 — A Short History of Inference and Simulation
    # =========================================================================
    {
        "id": "s04-history",
        "part": "intro",
        "title": "A Short History of Inverse Probability and Simulation",
        "html": r"""
<h2>A Short History of Inverse Probability and Simulation</h2>
<div style="display:flex; gap:10px; margin:14px 0; flex-wrap:wrap;">
  <div class="box box-blue" style="flex:1 1 200px; margin:0;">
    <strong style="color:#2563eb;">1763 &ndash; 1812: Inverse Probability</strong><br>
    <span style="font-size:0.88em;"><strong>Bayes &amp; Laplace</strong> invert analytical binomial and celestial-mechanics likelihoods via Bayes' theorem.</span>
  </div>
  <div class="box box-orange" style="flex:1 1 200px; margin:0;">
    <strong style="color:#d97706;">1940s: Monte Carlo Forward Sim</strong><br>
    <span style="font-size:0.88em;"><strong>Ulam &amp; von Neumann</strong> simulate stochastic neutron cascades forward on ENIAC when analytic PDEs fail.</span>
  </div>
  <div class="box box-green" style="flex:1 1 200px; margin:0;">
    <strong style="color:#059669;">1984 &ndash; 2002: Classical ABC</strong><br>
    <span style="font-size:0.88em;"><strong>Rubin (1984), Tavar&eacute; (1997), Pritchard (1999), Beaumont (2002)</strong> infer coalescent genetics via accept&ndash;reject simulations.</span>
  </div>
  <div class="box box-purple" style="flex:1 1 200px; margin:0;">
    <strong style="color:#7c3aed;">2016 &ndash; Present: Neural SBI</strong><br>
    <span style="font-size:0.88em;"><strong>Papamakarios, Murray, Cranmer, Hermans et al.</strong> learn conditional posteriors, likelihoods, and ratios via deep neural networks.</span>
  </div>
</div>
<div class="box" style="text-align:center; margin-top:14px; font-size:1.02em;">
  <strong>Key historical lesson:</strong> A computer model that predicts experimental outcomes forward in time does <em>not</em> automatically supply an evaluable probability density \(p(x_{\mathrm{obs}}\mid\theta)\) for solving the inverse problem.
</div>
""",
        "nb_markdown": r"""### Slide 4 — A Short History of Inverse Probability and Simulation

- **1763–1812 (Bayes & Laplace):** Inverse probability for tractable analytical likelihoods.
- **1940s (Ulam & von Neumann):** Monte Carlo forward simulation of complex stochastic branching processes.
- **1984–2002 (Rubin, Tavaré, Pritchard, Beaumont):** Approximate Bayesian Computation (ABC) arises in population genetics where coalescent genealogies make $p(x\mid\theta)$ intractable.
- **2016–Present (Papamakarios, Murray, Cranmer, Hermans et al.):** Neural density and ratio estimation turn simulation banks into reusable amortised Bayesian inference engines.

> **Central point:** A model that can predict experimental outcomes does *not* automatically supply an efficiently evaluable probability density for those outcomes.""",
        "notes": "2 min | Keep brisk: highlight the 70-year gap between forward Monte Carlo (1940s) and scalable neural inverse inference (2010s).",
    },

    # =========================================================================
    # Slide 5 — Our Scientific Experiment: The Mystery Galton Board (W02)
    # =========================================================================
    {
        "id": "s05-mystery-plinko",
        "part": "plinko",
        "title": "Our Scientific Experiment: The Mystery Galton Board",
        "demo_id": "w02_plinko_sim",
        "html": r"""
<h2>Our Scientific Experiment: The Mystery Galton Board</h2>
<p>A ball drops through \(R = 12\) rows of pegs, bouncing left or right at each peg before landing in one of \(13\) bottom bins. An experiment drops \(N = 100\) balls and records only the final bin counts \(x = (n_0, n_1, \dots, n_{12})\). Before writing any equations, explore how horizontal <strong>bias</strong> and <strong>centering force</strong> shape the histogram:</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 5 — Our Scientific Experiment: The Mystery Galton Board

A ball drops through $R = 12$ rows of pegs, bouncing left or right until landing in one of $13$ bins. Dropping $N = 100$ balls yields an observed landing histogram $x = (n_0, n_1, \dots, n_{12})$ with $\sum_{k=0}^{12} n_k = 100$.

**Question:** *Can we determine how the hidden machine is configured by observing only where the 100 balls land?*""",
        "notes": "3 min | Click 'Drop 1 ball' twice to show stochasticity, then sweep Bias and Centering sliders to build physical intuition before showing formulas.",
    },

    # =========================================================================
    # Slide 6 — A Simple Galton Board Has a Simple Likelihood (W03)
    # =========================================================================
    {
        "id": "s06-simple-galton",
        "part": "plinko",
        "title": "A Simple Galton Board Has a Simple Binomial Likelihood",
        "demo_id": "w03_analytic_galton",
        "html": r"""
<h2>A Simple Galton Board Has a Simple Binomial Likelihood</h2>
<p>In the textbook Galton board (\(\kappa = 0\)), every peg has the same position-independent rightward probability \(p = \sigma(\alpha) = \frac{1}{1 + e^{-\alpha}}\), yielding an elementary closed-form Binomial likelihood:</p>
<div class="box box-blue" style="text-align:center; padding:8px 14px;">
  \[ P(K = k \mid \alpha, \kappa = 0) \;=\; \binom{R}{k}\,p^k\,(1 - p)^{R - k}, \qquad k \in \{0, 1, \dots, R\} \]
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 6 — A Simple Galton Board Has a Simple Binomial Likelihood

When there is no position-dependent restoring tendency ($\kappa = 0$), every bounce is an independent Bernoulli trial with $p = \sigma(\alpha)$:

$$P(K = k \mid \alpha, \kappa = 0) = \binom{R}{k}\,p^k\,(1 - p)^{R - k}, \qquad k \in \{0, \dots, R\}$$

Increase $N$ in the slider below to watch the empirical frequencies converge to the exact Binomial distribution, then switch the dropdown to the state-dependent board ($\kappa > 0$).""",
        "notes": "2.5 min | Show N=25 vs N=2000 in Binomial mode, then toggle to State-dependent mode to show how kappa > 0 narrows the distribution below Binomial.",
    },

    # =========================================================================
    # Slide 7 — A Slightly More Realistic Machine
    # =========================================================================
    {
        "id": "s07-state-dependent",
        "part": "plinko",
        "title": "State-Dependent Transitions Break the Binomial Formula",
        "html": r"""
<h2>State-Dependent Transitions Break the Binomial Formula</h2>
<div style="display:flex; gap:16px; align-items:stretch; flex-wrap:wrap;">
  <div style="flex:1 1 420px;">
    <p>Let \(k_t \in \{0,\dots,t\}\) count rightward moves after \(t\) rows, so horizontal displacement from the board centre is \(s_t = 2k_t - t\). Introduce two unknown machine parameters \(\theta = (\alpha, \kappa)\):</p>
    <div class="box box-blue" style="text-align:center;">
      \[ P(\mathrm{right} \mid s_t, \alpha, \kappa) \;=\; \sigma\!\left(\alpha - \kappa\frac{s_t}{\sqrt{R}}\right), \qquad \sigma(u) = \frac{1}{1 + e^{-u}} \]
    </div>
    <ul style="line-height:1.6;">
      <li><strong>Horizontal bias \(\alpha \in [-1.5, 1.5]\):</strong> tilts bounces rightward (\(\alpha > 0\)) or leftward (\(\alpha < 0\)).</li>
      <li><strong>Restoring tendency \(\kappa \in [0, 3.0]\):</strong> pulls straying balls back toward \(s_t = 0\), compressing landing spread.</li>
      <li>When \(\kappa > 0\), step probabilities depend on <em>where the ball currently sits</em>: paths with identical final bin \(k_R\) no longer share equal probability!</li>
    </ul>
  </div>
  <div style="flex:1 1 340px;">
    <div class="box box-orange" style="margin-top:0;">
      <strong>Experimental setup (default benchmark):</strong>
      <ul style="font-size:0.92em; line-height:1.55; margin-top:6px;">
        <li>Board depth: \(R = 12\) rows (\(13\) bins).</li>
        <li>Sample size: \(N = 100\) independent balls per experiment.</li>
        <li>Uniform prior: \(p(\alpha, \kappa) = \mathcal{U}(-1.5, 1.5) \times \mathcal{U}(0, 3)\).</li>
        <li>Observed data: \(x_{\mathrm{obs}} = (n_0, \dots, n_{12})\) generated from hidden true settings \(\theta^\star = (\alpha^\star, \kappa^\star)\).</li>
      </ul>
    </div>
    <p style="font-size:0.86em; color:#64748b; margin-top:8px;">
      Note: this is a clean pedagogical probabilistic simulator rather than a rigid-body mechanics model of steel spheres.
    </p>
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 7 — State-Dependent Transitions Break the Binomial Formula

After $t$ rows with $k_t$ rightward bounces, horizontal position is $s_t = 2k_t - t$. Define the state-dependent transition law with parameters $\theta = (\alpha, \kappa)$:

$$P(\mathrm{right} \mid s_t, \alpha, \kappa) = \sigma\!\left(\alpha - \kappa\frac{s_t}{\sqrt{R}}\right), \qquad \alpha \in [-1.5, 1.5], \quad \kappa \in [0, 3]$$

- **$\alpha$ (horizontal bias):** shifts the landing distribution left or right.
- **$\kappa$ (centering tendency):** pulls displaced balls back toward the middle ($s_t = 0$), narrowing the variance.
- Once $\kappa > 0$, transition probabilities depend on the intermediate path $s_1, \dots, s_{R-1}$, breaking the closed-form Binomial formula.""",
        "notes": "2.5 min | Point out why Binomial(R, p) fails: p^k (1-p)^(R-k) assumed every path with k rights had identical probability.",
    },

    # =========================================================================
    # Slide 8 — Simulating Is Easy; Evaluating the Likelihood Sums Over Paths
    # =========================================================================
    {
        "id": "s08-path-sum",
        "part": "plinko",
        "title": "Simulating One Path vs Summing Over All Latent Paths",
        "demo_id": "w02c_paths_challenge",
        "html": r"""
<h2>Simulating One Path vs Summing Over All Latent Paths</h2>
<div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:6px;">
  <div style="flex:1 1 420px;">
    <p>To evaluate \(p(x\mid\theta)\), probability theory requires marginalising over all unobserved intermediate trajectories \(z = (s_1,\dots,s_{R-1})\):</p>
    <div class="box box-purple" style="text-align:center; padding:8px 12px; margin:6px 0;">
      \[ p(x \mid \theta) \;=\; \sum_{z \in \mathcal{Z}} p(x, z \mid \theta), \qquad \theta = (\alpha, \kappa) \]
    </div>
  </div>
  <div class="box" style="flex:1 1 360px; margin:0; font-size:0.88em;">
    <strong>Important scientific distinction:</strong> Following one random path takes \(12\) coin flips. Naive path enumeration spans \(2^{12} = 4{,}096\) paths per ball (\((2^{12})^{100}\) for 100 balls!). <em>(For this Markovian toy board, dynamic programming can compute bin probabilities exactly; we withhold that solver until Slide 31 as an honest ground-truth benchmark!)</em>
  </div>
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 8 — Simulating One Path vs Summing Over All Latent Paths

Evaluating the likelihood requires marginalising over every compatible latent trajectory $z$:

$$p(x \mid \theta) = \sum_{z \in \mathcal{Z}} p(x, z \mid \theta), \qquad \theta = (\alpha, \kappa)$$

**Following one random path is easy; accounting for the probabilities of all compatible paths is a fundamentally harder computational task.**

*(Scientific caveat: because this discrete pedagogical board is Markovian on 13 states, it admits an exact dynamic-programming recursion—which we deliberately hold back until Slide 31 as an independent validation benchmark. In realistic simulators with continuous/non-Markovian nuisance states, the integral $\int p(x, z\mid\theta)\,dz$ is genuinely intractable.)*""",
        "notes": "3 min | Drag the 'Inspect target bin k' slider on the left panel to show 924 paths hitting bin 6 vs 1 path hitting bin 12.",
    },

    # =========================================================================
    # Slide 9 — The Challenge
    # =========================================================================
    {
        "id": "s09-challenge",
        "part": "plinko",
        "title": "The Mystery Challenge: Which Settings Generated the Data?",
        "html": r"""
<h2>The Mystery Challenge: Which Settings Generated the Data?</h2>
<p>In the right panel above, compare the three candidate configurations against the Mystery Machine's observed 100-ball histogram \(x_{\mathrm{obs}}\):</p>
<div style="display:flex; gap:12px; margin:12px 0; flex-wrap:wrap;">
  <div class="box" style="flex:1 1 220px; margin:0;">
    <strong>Candidate A: \((\alpha = 0.25, \kappa = 0.25)\)</strong><br>
    <span style="font-size:0.9em;">Mild rightward tilt, almost no centering force (broad near-Binomial spread).</span>
  </div>
  <div class="box" style="flex:1 1 220px; margin:0;">
    <strong>Candidate B: \((\alpha = 0.55, \kappa = 1.15)\)</strong><br>
    <span style="font-size:0.9em;">Moderate rightward bias balanced by moderate restoring pull toward the centre.</span>
  </div>
  <div class="box" style="flex:1 1 220px; margin:0;">
    <strong>Candidate C: \((\alpha = 0.95, \kappa = 2.45)\)</strong><br>
    <span style="font-size:0.9em;">Strong rightward push offset by strong centering compression.</span>
  </div>
</div>
<details class="quiz-box">
  <summary>Audience discussion: Why can Candidate B and Candidate C produce surprisingly similar means, and how do we tell them apart?</summary>
  <div class="quiz-ans">
    Increasing both \(\alpha\) and \(\kappa\) keeps the <em>mean</em> landing bin near \(k \approx 7.1\) (creating a curved parameter correlation ridge!), while Candidate C produces a noticeably narrower <em>spread</em> than Candidate A, and finite \(N=100\) multinomial noise blurs single realizations. We need a principled Bayesian posterior over \((\alpha, \kappa)\), not eyeball matching!
  </div>
</details>
""",
        "nb_markdown": r"""### Slide 9 — The Mystery Challenge

Comparing Candidate A $(0.25, 0.25)$, Candidate B $(0.55, 1.15)$, and Candidate C $(0.95, 2.45)$ in the widget above:
- Raising both $\alpha$ and $\kappa$ shifts the mean right while pulling balls back toward the centre, creating a partial **tradeoff ridge** in $(\alpha, \kappa)$.
- With only $N = 100$ balls, random realization noise means no single simulated histogram matches $x_{\mathrm{obs}}$ bin-for-bin. We need a formal likelihood-free Bayesian procedure.""",
        "notes": "2 min | Poll the audience on Candidates A, B, C using the right-hand panel of W02c, then click 'New 100-ball realization' twice to show finite-sample variability.",
    },

    # =========================================================================
    # Slide 10 — Rejection ABC from First Principles
    # =========================================================================
    {
        "id": "s10-abc-intro",
        "part": "abc",
        "title": "A Simple Solution: Simulate and Compare (Rejection ABC)",
        "html": r"""
<h2>A Simple Solution: Simulate and Compare (Rejection ABC)</h2>
<p>If we can sample \(\theta_i \sim p(\theta)\) and forward-simulate \(x_i \sim p(x\mid\theta_i)\), we already generate exact draws from the <strong>joint distribution</strong> \(p(\theta, x) = p(\theta)\,p(x\mid\theta)\):</p>
<div class="box box-orange">
  <strong>Rejection ABC Algorithm:</strong>
  <ol style="margin:6px 0 0; line-height:1.6;">
    <li>Sample parameter proposal \(\theta_i \sim p(\theta)\) for \(i = 1, \dots, N\).</li>
    <li>Run the forward simulator once to draw synthetic data \(x_i \sim p(x\mid\theta_i)\).</li>
    <li>Accept \(\theta_i\) if \(d\bigl(S(x_i), S(x_{\mathrm{obs}})\bigr) < \varepsilon\), where \(S(\cdot)\) is a summary statistic and \(d(\cdot,\cdot)\) is a distance metric.</li>
  </ol>
</div>
<div class="box box-blue" style="text-align:center; margin-top:10px;">
  <strong>What distribution do accepted samples follow?</strong>
  \[ p_\varepsilon(\theta \mid x_{\mathrm{obs}}) \;\propto\; p(\theta)\,\underbrace{\int \mathbb{1}\!\Bigl(d\bigl(S(x), S(x_{\mathrm{obs}})\bigr) < \varepsilon\Bigr)\,p(x\mid\theta)\,dx}_{\mathbb{P}\bigl[d(S(X), S(x_{\mathrm{obs}})) < \varepsilon \;\big|\; \theta\bigr] \;\approx\; \text{smoothed likelihood}} \;\xrightarrow{\;\varepsilon \to 0\;}\; p\bigl(\theta \;\big|\; S(x_{\mathrm{obs}})\bigr) \]
</div>
""",
        "nb_markdown": r"""### Slide 10 — A Simple Solution: Simulate and Compare (Rejection ABC)

For $i = 1, \dots, N$:
$$\theta_i \sim p(\theta), \qquad x_i \sim p(x\mid\theta_i), \qquad \text{accept } \theta_i \text{ if } d\bigl(S(x_i), S(x_{\mathrm{obs}})\bigr) < \varepsilon$$

Accepted draws follow the exact conditional distribution given acceptance:

$$p_\varepsilon(\theta\mid x_{\mathrm{obs}}) \propto p(\theta)\,\mathbb{P}\!\left[d\bigl(S(X), S(x_{\mathrm{obs}})\bigr) < \varepsilon \;\big|\; \theta\right]$$

where the acceptance probability acts as a kernel-smoothed surrogate likelihood, converging to $p(\theta\mid S(x_{\mathrm{obs}}))$ as $\varepsilon \to 0$.""",
        "notes": "2.5 min | Walk carefully through the integral: indicator function times p(x|theta) integrates to the probability of landing within epsilon of x_obs.",
    },

    # =========================================================================
    # Slide 11 — Understanding ABC in One Dimension (Original Chi2 Model)
    # =========================================================================
    {
        "id": "s11-abc-1d-setup",
        "part": "abc",
        "title": "Understanding ABC Geometry in One Dimension",
        "html": r"""
<h2>Understanding ABC Geometry in One Dimension</h2>
<p>Before tackling the 13-bin Galton board, inspect the exact geometry of joint-space conditioning on our course's canonical one-dimensional toy model:</p>
<div class="box box-blue" style="text-align:center; max-width:680px; margin:12px auto;">
  \[ \theta \sim \chi^2(k = 5), \qquad x \mid \theta \sim \mathcal{N}\!\bigl(\sqrt{\theta},\, 1\bigr) \]
</div>
<ul style="line-height:1.65;">
  <li>Sampling \(\theta_i \sim \chi^2(5)\) followed by \(x_i \sim \mathcal{N}(\sqrt{\theta_i}, 1)\) populates the 2D plane \((\theta, x)\) according to \(p(\theta, x) = p(\theta)p(x\mid\theta)\).</li>
  <li>Bayes' theorem \(p(\theta\mid x_{\mathrm{obs}}) = \frac{p(\theta, x_{\mathrm{obs}})}{p(x_{\mathrm{obs}})}\) simply asks for the <strong>normalised horizontal slice</strong> through \(p(\theta, x)\) along the line \(x = x_{\mathrm{obs}}\)!</li>
  <li>Because continuous simulations satisfy \(\mathbb{P}(X = x_{\mathrm{obs}}) = 0\), Rejection ABC thickens that infinitesimally thin line into a horizontal slab \([x_{\mathrm{obs}} - \varepsilon,\, x_{\mathrm{obs}} + \varepsilon]\).</li>
</ul>
""",
        "nb_markdown": r"""### Slide 11 — Understanding ABC Geometry in One Dimension

Consider the canonical 1D nonlinear model from the original lecture:

$$\theta \sim \chi^2(k = 5), \qquad x \mid \theta \sim \mathcal{N}\!\bigl(\sqrt{\theta},\, 1\bigr)$$

- Ancestral pairs $(\theta_i, x_i)$ populate the joint distribution $p(\theta, x)$.
- Conditioning on $x = x_{\mathrm{obs}}$ is geometrically a **horizontal slice** through $p(\theta, x)$.
- Since $\mathbb{P}(X = x_{\mathrm{obs}}) = 0$ for continuous $X$, ABC replaces the zero-thickness line with a slab $|x - x_{\mathrm{obs}}| < \varepsilon$.""",
        "notes": "2 min | Set up the geometric picture of conditioning as slicing the joint distribution p(theta, x) horizontally.",
    },

    # =========================================================================
    # Slide 12 — ABC as a Slice Through the Joint Distribution (W04)
    # =========================================================================
    {
        "id": "s12-abc-slice",
        "part": "abc",
        "title": "ABC as a Slice Through the Joint Distribution",
        "demo_id": "w04_abc_1d",
        "html": r"""
<h2>ABC as a Slice Through the Joint Distribution</h2>
<p>Vary the tolerance \(\varepsilon\), observed value \(x_{\mathrm{obs}}\), and simulation pool size \(N\) below:</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 12 — ABC as a Slice Through the Joint Distribution

Adjust $\varepsilon$, $x_{\mathrm{obs}}$, and $N$ in `W04` below. Notice that the simulation pool stays fixed while $\varepsilon$ varies: large $\varepsilon$ leaks back toward the $\chi^2(5)$ prior, whereas tiny $\varepsilon$ leaves only a handful of noisy samples.""",
        "notes": "3 min | Drag epsilon from 1.80 down to 0.06 live: show prior leakage at 1.80, sweet spot near 0.30, and severe Monte Carlo sparsity at 0.06.",
    },

    # =========================================================================
    # Slide 13 — The Tolerance Problem (Clicker Question)
    # =========================================================================
    {
        "id": "s13-tolerance-clicker",
        "part": "abc",
        "title": "The Tolerance Problem: Audience Question",
        "html": r"""
<h2>The Tolerance Problem: Audience Question</h2>
<div class="box box-orange">
  <strong>Question:</strong> Suppose we run Rejection ABC with a fixed budget of \(N = 5{,}000\) simulations and choose a tiny tolerance \(\varepsilon = 10^{-4}\) so bias is negligible. Will this guarantee a highly accurate posterior estimate?
  <ul style="margin:8px 0 0; line-height:1.55;">
    <li><strong>A.</strong> Yes: \(p_\varepsilon(\theta\mid x_{\mathrm{obs}}) \to p(\theta\mid x_{\mathrm{obs}})\) monotonically as \(\varepsilon \to 0\).</li>
    <li><strong>B.</strong> No: almost zero simulations will be accepted, so Monte Carlo variance explodes.</li>
    <li><strong>C.</strong> Only if we also use a low-dimensional summary statistic \(S(x)\).</li>
    <li><strong>D.</strong> Both Monte Carlo starvation (few/no accepted draws) <em>and</em> potential summary-statistic information loss limit accuracy.</li>
  </ul>
</div>
<details class="quiz-box">
  <summary>Reveal answer and three-part error decomposition</summary>
  <div class="quiz-ans">
    <strong>Answer: D (with B as the immediate finite-\(N\) failure).</strong> Total ABC error decomposes into three distinct sources:
    <ol style="margin:6px 0 0; line-height:1.5;">
      <li><strong>Tolerance bias (\(\varepsilon > 0\)):</strong> \(p_\varepsilon(\theta\mid S(x_{\mathrm{obs}})) \neq p(\theta\mid S(x_{\mathrm{obs}}))\); shrinks as \(\varepsilon \to 0\).</li>
      <li><strong>Monte Carlo error (\(M_{\mathrm{acc}} < \infty\)):</strong> scales as \(1/\sqrt{M_{\mathrm{acc}}}\); explodes when \(\varepsilon\) is too small for fixed \(N\)!</li>
      <li><strong>Summary-statistic loss (\(S(x)\) insufficient):</strong> \(p(\theta\mid S(x_{\mathrm{obs}})) \neq p(\theta\mid x_{\mathrm{obs}})\); <em>never</em> vanishes even if \(N \to \infty\) and \(\varepsilon \to 0\)!</li>
    </ol>
  </div>
</details>
""",
        "nb_markdown": r"""### Slide 13 — The Tolerance Problem: Audience Question

**Question:** Suppose we run Rejection ABC with a fixed budget of $N = 5{,}000$ simulations and set a tiny tolerance $\varepsilon = 10^{-4}$ so bias is negligible. Will this guarantee a highly accurate posterior estimate?

- **A.** Yes: $p_\varepsilon(\theta\mid x_{\mathrm{obs}}) \to p(\theta\mid x_{\mathrm{obs}})$ monotonically as $\varepsilon \to 0$.
- **B.** No: almost zero simulations will be accepted, so Monte Carlo variance explodes.
- **C.** Only if $\theta$ is one-dimensional.
- **D.** Both Monte Carlo starvation (few/no accepted draws) *and* potential summary-statistic information loss limit accuracy.

<details>
<summary><strong>Click to reveal answer &amp; three-part error decomposition</strong></summary>

**Answer: D (with B as the immediate finite-$N$ failure).**

1. **Finite-tolerance bias ($\varepsilon > 0$):** smooths the posterior toward the prior ($\mathcal{O}(\varepsilon^2)$).
2. **Monte Carlo variance ($M_{\mathrm{acc}} \ll N$):** at $\varepsilon = 10^{-4}$, $M_{\mathrm{acc}} \approx 0$, causing estimator variance $\propto 1/M_{\mathrm{acc}}$ to blow up.
3. **Insufficient summary bias ($S(x)$ lossy):** even at $N = \infty$ and $\varepsilon \to 0$, ABC converges to $p(\theta\mid S(x_{\mathrm{obs}}))$, never recovering discarded information.
</details>""",
        "notes": "2.5 min | Have students vote A/B/C/D before clicking Reveal. Stress the 3-way error decomposition.",
    },

    # =========================================================================
    # Slide 14 — ABC and Correlated Variables (W05)
    # =========================================================================
    {
        "id": "s14-correlated-gauss",
        "part": "abc",
        "title": "ABC and Correlated Variables",
        "demo_id": "w05_corr_gauss",
        "html": r"""
<h2>ABC and Correlated Variables</h2>
<p>For a bivariate Gaussian \((\theta, x)\) with correlation \(\rho\), exact conditioning shifts the mean proportionally to \(\rho\) and shrinks variance by \(1 - \rho^2\), whereas finite-\(\varepsilon\) ABC inflates variance by \(\frac{1}{3}\rho^2\sigma_\theta^2(\varepsilon/\sigma_x)^2\):</p>
<div class="box" style="text-align:center; padding:6px 14px; margin:4px auto 8px; max-width:740px;">
  \[ \mathbb{E}[\theta \mid x] \;=\; \mu_\theta + \rho\frac{\sigma_\theta}{\sigma_x}(x - \mu_x), \qquad \operatorname{Var}(\theta \mid x) \;=\; \sigma_\theta^2(1 - \rho^2) \]
</div>
__DEMO__
""",
        "nb_markdown": r"""### Slide 14 — ABC and Correlated Variables

For jointly Gaussian $(\theta, x)$ with correlation $\rho$:

$$\mathbb{E}[\theta \mid x] = \mu_\theta + \rho\frac{\sigma_\theta}{\sigma_x}(x - \mu_x), \qquad \operatorname{Var}(\theta \mid x) = \sigma_\theta^2(1 - \rho^2)$$

Explore how correlation $\rho$ concentrates the posterior and how wide tolerance $\varepsilon_2$ both inflates posterior variance and biases the posterior mean toward $\mu_\theta$.""",
        "notes": "2.5 min | Highlight the factor of rho in the conditional mean and (1 - rho^2) in the conditional variance; sweep rho from 0 to 0.9.",
    },

    # =========================================================================
    # Slide 15 — ABC Solves Our Plinko Problem (W06)
    # =========================================================================
    {
        "id": "s15-plinko-abc",
        "part": "abc",
        "title": "Rejection ABC Solves Our Mystery Galton Board",
        "demo_id": "w06_plinko_abc",
        "html": r"""
<h2>Rejection ABC Solves Our Mystery Galton Board</h2>
<p>Return to our Mystery Galton Board! Draw \((\alpha_i, \kappa_i) \sim \mathcal{U}(-1.5, 1.5)\times\mathcal{U}(0, 3)\), simulate 100-ball histograms \(x_i\), and accept parameters whose histogram lies within tolerance \(\varepsilon\) of \(x_{\mathrm{obs}}\):</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 15 — Rejection ABC Solves Our Mystery Galton Board

Applying Rejection ABC to our cached Galton-Board prior-predictive bank:
- Each point in the right panel is a simulated machine $(\alpha_i, \kappa_i)$.
- Accepted machines (orange) cluster along the correlated $(\alpha, \kappa)$ region that reproduces both the location and width of $x_{\mathrm{obs}}$!""",
        "notes": "3 min | Sweep epsilon on W06, switch between Wasserstein and Total Variation distances, and briefly toggle 'Show exact reference' to show ABC captures the right region but is noisy at small epsilon.",
    },

    # =========================================================================
    # Slide 16 — Choosing a Distance and Summary Statistics (W07)
    # =========================================================================
    {
        "id": "s16-summaries",
        "part": "abc",
        "title": "Choosing Summary Statistics and Distances on the Galton Board",
        "demo_id": "w07_plinko_summaries",
        "html": r"""
<h2>Choosing Summary Statistics and Distances on the Galton Board</h2>
<p>Compare three summary choices on the <em>exact same</em> Mystery Galton Board observation \(x_{\mathrm{obs}}\): (1) mean bin \(\bar{k}\) alone, (2) mean and variance \((\bar{k}, s^2)\), and (3) the full 13-bin histogram:</p>
__DEMO__
""",
        "nb_markdown": r"""### Slide 16 — Choosing Summary Statistics and Distances on the Galton Board

Compare three summaries on the same observation $x_{\mathrm{obs}}$:
1. **Mean landing position $\bar{k}$ only:** 1 summary cannot constrain 2 parameters $(\alpha, \kappa)$! Even at tiny tolerance, $\kappa$ spans nearly the entire prior $[0, 3]$.
2. **Mean and variance $(\bar{k}, s^2)$:** $\bar{k}$ constrains horizontal bias $\alpha$ while $s^2$ constrains restoring strength $\kappa$, recovering almost the full-histogram posterior!
3. **Full 13-bin histogram:** avoids summary loss, but in higher-dimensional data spaces raw vector distances suffer the $\varepsilon^D$ acceptance cliff.""",
        "notes": "3 min | Start on '1. Mean bin only' and lower M to 25 to prove shrinking epsilon NEVER recovers discarded variance info; then switch to '2. Mean & variance'.",
    },

    # =========================================================================
    # Slide 17 — Why ABC Is Not Always Enough
    # =========================================================================
    {
        "id": "s17-abc-limits",
        "part": "abc",
        "title": "Why Rejection ABC Is Not Always Enough",
        "html": r"""
<h2>Why Rejection ABC Is Not Always Enough</h2>
<div style="display:flex; gap:16px; flex-wrap:wrap;">
  <div style="flex:1 1 380px;">
    <ul style="line-height:1.65;">
      <li><strong>Curse of dimensionality:</strong> accepting inside an \(\varepsilon\)-ball in \(D = \dim S(x)\) dimensions has probability \(\propto \varepsilon^D\), crashing exponentially for spectra, images, or time series.</li>
      <li><strong>Severe simulation waste:</strong> rejecting \(98\%\text{--}99.9\%\) of simulations is untenable when each cosmology or detector run costs CPU-hours.</li>
      <li><strong>Summary engineering bottleneck:</strong> hand-designing sufficient statistics for complex simulators is rarely possible a priori.</li>
      <li><strong>Pointwise rejection loops:</strong> sequential / MCMC-ABC variants focus on a single \(x_{\mathrm{obs}}\) and must restart from scratch for every new dataset.</li>
    </ul>
  </div>
  <div style="flex:1 1 320px;">
    <div class="box box-green" style="margin-top:0; font-size:0.9em;">
      <strong>Historical context (1997&ndash;2009):</strong><br>
      ABC revolutionised coalescent population genetics (Tavar&eacute; et al. 1997; Pritchard et al. 1999; Beaumont, Zhang &amp; Balding 2002), inspiring local linear regression adjustments and SMC-ABC (Sisson et al. 2007; Toni et al. 2009).
    </div>
    <div class="box box-blue" style="text-align:center; font-size:0.98em;">
      <em>We simulated thousands of Galton boards and threw almost all of them away. What if we trained a flexible probability model on <strong>every single simulation</strong> instead?</em>
    </div>
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 17 — Why Rejection ABC Is Not Always Enough

- **Curse of dimensionality:** acceptance probability scales as $\varepsilon^{\dim S(x)}$.
- **Wasted simulations:** discarding $99\%$ of simulator runs throws away global smooth structure across $(\theta, x)$-space.
- **Hand-crafted summaries:** rarely sufficient on high-dimensional scientific observables.

> *We simulated thousands of experiments and discarded most of them. What if we could learn a smooth probability distribution from **all** of them instead?*""",
        "notes": "2 min | Use the closing question to motivate parametric density estimation in Slide 18.",
    },

    # =========================================================================
    # Slide 18 — From Simulation Samples to Density Estimation
    # =========================================================================
    {
        "id": "s18-density-intro",
        "part": "density",
        "title": "From Simulation Samples to Density Estimation",
        "html": r"""
<h2>From Simulation Samples to Density Estimation</h2>
<p>Abstract the core statistical problem: nature (or our simulator) gives i.i.d. samples \(x_1, x_2, \dots, x_N \sim p(x)\), where the true density \(p(x)\) <strong>cannot be evaluated</strong>.</p>
<div style="display:flex; gap:16px; margin-top:12px; flex-wrap:wrap;">
  <div class="box" style="flex:1 1 300px; margin:0;">
    <strong>True Generative Distribution \(p(x)\)</strong>
    <ul style="font-size:0.92em; line-height:1.55;">
      <li>Defined implicitly by the simulator.</li>
      <li>Can draw samples \(x_i \sim p(x)\).</li>
      <li>Cannot evaluate \(p(x)\) or \(\nabla_x \log p(x)\).</li>
    </ul>
  </div>
  <div class="box box-blue" style="flex:1 1 300px; margin:0;">
    <strong>Parametric Approximation \(q_\phi(x)\)</strong>
    <ul style="font-size:0.92em; line-height:1.55;">
      <li>Explicit tractable family indexed by parameters \(\phi\) (e.g. Gaussian mixture or normalising flow).</li>
      <li>Strictly normalised: \(q_\phi(x) \ge 0\) and \(\int q_\phi(x)\,dx = 1\) for all \(\phi\).</li>
      <li>Both \(q_\phi(x)\) and \(\nabla_\phi \log q_\phi(x)\) are cheap to evaluate!</li>
    </ul>
  </div>
</div>
<p style="margin-top:12px;"><strong>Goal:</strong> Find \(\phi^\star\) so that \(q_{\phi^\star}(x) \approx p(x)\) using <em>only</em> the samples \(\{x_i\}_{i=1}^N\).</p>
""",
        "nb_markdown": r"""### Slide 18 — From Simulation Samples to Density Estimation

Suppose we observe samples $x_1, \dots, x_N \sim p(x)$ from an implicit distribution $p(x)$ whose density cannot be evaluated. We introduce a tractable, normalised parametric family $q_\phi(x)$ ($\int q_\phi(x)\,dx = 1$) whose log-density $\log q_\phi(x)$ and gradients $\nabla_\phi \log q_\phi(x)$ are analytically computable.""",
        "notes": "2 min | Keep p(x) (implicit truth) and q_phi(x) (normalised tractable model) visually distinct.",
    },

    # =========================================================================
    # Slide 19 — KL Divergence
    # =========================================================================
    {
        "id": "s19-kl-divergence",
        "part": "density",
        "title": "Measuring Distribution Discrepancy: KL Divergence",
        "html": r"""
<h2>Measuring Distribution Discrepancy: KL Divergence</h2>
<p>We measure how far \(q_\phi\) departs from \(p\) using the <strong>forward Kullback&ndash;Leibler (KL) divergence</strong> (relative entropy):</p>
<div class="box box-blue" style="text-align:center;">
  \[ D_{\mathrm{KL}}(p \,\|\, q_\phi) \;=\; \int p(x)\,\log\frac{p(x)}{q_\phi(x)}\,dx \;=\; \mathbb{E}_{x \sim p(x)}\!\left[\log p(x) - \log q_\phi(x)\right] \]
</div>
<div style="display:flex; gap:14px; margin-top:10px; flex-wrap:wrap;">
  <div class="box" style="flex:1 1 280px; margin:0;">
    <strong>Fundamental properties:</strong>
    <ul style="font-size:0.9em; line-height:1.55;">
      <li><strong>Gibbs' inequality:</strong> \(D_{\mathrm{KL}}(p \,\|\, q_\phi) \ge 0\), with equality iff \(p(x) = q_\phi(x)\) almost everywhere (by Jensen's inequality on \(-\log\)).</li>
      <li><strong>Asymmetry:</strong> \(D_{\mathrm{KL}}(p \,\|\, q_\phi) \neq D_{\mathrm{KL}}(q_\phi \,\|\, p)\). Forward KL is <em>mass-covering</em> (penalises \(q_\phi \to 0\) wherever \(p > 0\)), avoiding silent under-coverage!</li>
    </ul>
  </div>
  <div class="box box-purple" style="flex:1 1 280px; margin:0;">
    <strong>Information-theoretic link:</strong><br>
    <span style="font-size:0.9em; line-height:1.55;">
      Expected coding overhead (extra nats) when encoding draws from \(p\) using a code optimal for \(q_\phi\). Applied to joint vs product distributions, KL defines <strong>mutual information</strong>:
      \[ I(\Theta; X) = D_{\mathrm{KL}}\!\bigl(p(\theta, x) \;\big\|\; p(\theta)p(x)\bigr). \]
    </span>
  </div>
</div>
""",
        "nb_markdown": r"""### Slide 19 — Measuring Distribution Discrepancy: KL Divergence

$$D_{\mathrm{KL}}(p \,\|\, q_\phi) = \int p(x)\,\log\frac{p(x)}{q_\phi(x)}\,dx = \mathbb{E}_{p(x)}\!\bigl[\log p(x) - \log q_\phi(x)\bigr]$$

- **Non-negativity:** $D_{\mathrm{KL}}(p \,\|\, q_\phi) \ge 0$, with $D_{\mathrm{KL}}(p \,\|\, q_\phi) = 0 \iff p = q_\phi$ a.e.
- **Asymmetry (mass-covering):** wherever $p(x) > 0$ and $q_\phi(x) \to 0$, the integrand blows up to $+\infty$, forcing $q_\phi$ to cover all support of $p$.
- **Mutual information:** $I(\Theta; X) = D_{\mathrm{KL}}\bigl(p(\theta, x) \,\|\, p(\theta)p(x)\bigr)$ quantifies how much $X$ informs $\Theta$.""",
        "notes": "2.5 min | Emphasize why forward KL's mass-covering property is ideal for conservative scientific credible regions.",
    },

    # =========================================================================
    # Slide 20 — From KL Divergence to Maximum Likelihood
    # =========================================================================
    {
        "id": "s20-kl-to-nll",
        "part": "density",
        "title": "From Forward KL Divergence to Maximum Likelihood",
        "html": r"""
<h2>From Forward KL Divergence to Maximum Likelihood</h2>
<p>Split the logarithm inside \(D_{\mathrm{KL}}(p \,\|\, q_\phi)\):</p>
<div class="box" style="text-align:center;">
  \[ D_{\mathrm{KL}}(p \,\|\, q_\phi) \;=\; \mathbb{E}_{x \sim p(x)}\!\bigl[-\log q_\phi(x)\bigr] \;+\; \underbrace{\mathbb{E}_{x \sim p(x)}\!\bigl[\log p(x)\bigr]}_{-H(p),\;\text{constant wrt }\phi} \]
</div>
<p>Because the unknown entropy term \(\mathbb{E}_p[\log p(x)]\) has <strong>zero gradient with respect to \(\phi\)</strong>, minimising forward KL is mathematically identical to minimising expected negative log-likelihood, estimated by Monte Carlo on \(x_i \sim p(x)\):</p>
<div class="box box-green" style="text-align:center;">
  \[ \phi^\star \;=\; \arg\min_\phi \mathbb{E}_{x \sim p(x)}\!\bigl[-\log q_\phi(x)\bigr] \;\approx\; \arg\min_\phi \underbrace{\left\{-\frac{1}{N}\sum_{i=1}^N \log q_\phi(x_i)\right\}}_{\mathcal{L}(\phi):\;\text{empirical NLL}} \]
</div>
<div class="box box-blue" style="text-align:center; margin-top:10px;">
  <strong>The Mathematical Bridge to Neural SBI:</strong> We can train \(q_\phi\) to approximate \(p\) without <em>ever</em> evaluating the intractable density \(p(x)\) that generated the training samples!
</div>
""",
        "nb_markdown": r"""### Slide 20 — From Forward KL Divergence to Maximum Likelihood

$$D_{\mathrm{KL}}(p \,\|\, q_\phi) = \mathbb{E}_{p(x)}\!\bigl[-\log q_\phi(x)\bigr] + \underbrace{\mathbb{E}_{p(x)}\!\bigl[\log p(x)\bigr]}_{\text{independent of }\phi}$$

Dropping the $\phi$-independent constant and replacing the expectation over $p(x)$ by the sample average over $x_1,\dots,x_N \sim p(x)$:

$$\phi^\star = \arg\min_\phi \mathbb{E}_{p(x)}\!\bigl[-\log q_\phi(x)\bigr] \qquad\Longrightarrow\qquad \mathcal{L}(\phi) = -\frac{1}{N}\sum_{i=1}^N \log q_\phi(x_i)$$

> **Core result:** Training a parametric density estimator by maximum likelihood requires *samples* from $p$, never density evaluations of $p$!""",
        "notes": "3 min | Pause on this derivation: every single neural density estimator in SBI (NPE and NLE) rests on these three equations.",
    },
]
