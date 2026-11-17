"""Interactive SBI Lecture Widgets — Part 2 (W10 through W15 + Backup Animations).

Implements all conditional-density, amortized Galton-board NPE/NLE/NRE, posterior
predictive check, and coverage/SBC slider widgets from Section 5 of the Master Build Prompt.
"""

from __future__ import annotations

WIDGETS_PART2: dict[str, dict[str, object]] = {
    # =========================================================================
    # W10: Skewed LogNormal L1 vs L2 vs Conditional MDN Explorer (Slide 23)
    # =========================================================================
    "w10_loss_regression": {
        "title": "W10: L2 Mean vs L1 Median vs Conditional Density (MDN)",
        "height": 435,
        "data_keys": ["reg_mdn"],
        "html": """
<div class="iw-widget" id="w10-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w10-x">Slice x</label>
      <input type="range" id="w10-x" min="0" max="39" step="1" value="28">
      <span class="iw-val" id="w10-x-val">1.11</span>
    </div>
    <div class="iw-ctrl">
      <label for="w10-sk">Skewness &gamma;</label>
      <select class="iw-select" id="w10-sk">
        <option value="0.60">&gamma; = 0.60 (moderate)</option>
        <option value="1.00" selected>&gamma; = 1.00 (course default)</option>
        <option value="1.25">&gamma; = 1.25 (heavy tail)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w10-l2" checked>
      <label for="w10-l2">L<sub>2</sub> (mean)</label>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w10-l1" checked>
      <label for="w10-l1">L<sub>1</sub> (median)</label>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w10-mdn" checked>
      <label for="w10-mdn">MDN q<sub>&phi;</sub>(y | x)</label>
    </div>
    <button type="button" class="iw-btn" id="w10-reset">&#x21ba; Reset</button>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w10-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w10-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    y | x &sim; LogNormal(s = &gamma;x, loc = 2x): L<sub>2</sub> regression tracks E[Y | x] = 2x + exp(&gamma;<sup>2</sup>x<sup>2</sup>/2), L<sub>1</sub> tracks median(Y | x) = 2x + 1, while NLL-trained MDN recovers the full skewed distribution.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w10-root');
  if (!root || !window.SBI_DATA) return;
  var R = window.SBI_DATA.reg_mdn;
  var xEv = R.x_eval, yEv = R.y_eval;
  var slX = document.getElementById('w10-x');
  var selSk = document.getElementById('w10-sk');
  var chkL2 = document.getElementById('w10-l2');
  var chkL1 = document.getElementById('w10-l1');
  var chkMdn = document.getElementById('w10-mdn');
  slX.max = String(xEv.length - 1);

  function logNormPdf(y, x, sk) {
    var z = y - 2.0 * x;
    if (z <= 1e-5) return 0.0;
    var s = Math.max(1e-4, sk * x);
    var ly = Math.log(z);
    return Math.exp(-0.5 * (ly * ly) / (s * s)) / (z * s * 2.5066282746);
  }

  function draw() {
    var xi = parseInt(slX.value, 10);
    var xv = xEv[xi];
    var skKey = selSk.value, sk = parseFloat(skKey);
    var dat = R.by_skew[skKey];
    document.getElementById('w10-x-val').textContent = xv.toFixed(2);

    var theoMean = xEv.map(function(x) { return 2 * x + Math.exp(0.5 * sk * sk * x * x); });
    var theoMed = xEv.map(function(x) { return 2 * x + 1.0; });

    var sL = SBI.prepCanvas(document.getElementById('w10-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: 'Skewed regression data & learned predictors',
      xRange: [0.08, 1.52], yRange: [0, 11.5],
      xLabel: 'Input x', yLabel: 'Target y'
    });
    if (chkMdn.checked) axL.heatmapAK(xEv, yEv, dat.pdf, '37,99,235', 0.36);
    axL.scatter(dat.xs, dat.ys, 'rgba(71,85,105,0.40)', 1.9);
    if (chkL2.checked) {
      axL.line(xEv, theoMean, '#d97706', 1.6, [5, 4]);
      axL.line(xEv, dat.l2, '#d97706', 2.6, []);
    }
    if (chkL1.checked) {
      axL.line(xEv, theoMed, '#059669', 1.6, [5, 4]);
      axL.line(xEv, dat.l1, '#059669', 2.6, []);
    }
    axL.vline(xv, '#dc2626', 1.7, [3, 3]);
    axL.legend([
      { type: 'line', color: '#d97706', label: 'L₂ MLP ≈ E[Y | x]' },
      { type: 'line', color: '#059669', label: 'L₁ MLP ≈ median(Y | x)' },
      { type: 'bar', color: 'rgba(37,99,235,0.45)', label: 'MDN density q_φ(y | x)' }
    ], 'tl');

    var trueSlice = yEv.map(function(y) { return logNormPdf(y, xv, sk); });
    var mdnSlice = dat.pdf[xi];
    var yMax = Math.max(0.45, Math.max.apply(null, trueSlice) * 1.18, Math.max.apply(null, mdnSlice) * 1.18);

    var sR = SBI.prepCanvas(document.getElementById('w10-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: 'Conditional slice p(y | x = ' + xv.toFixed(2) + ')',
      xRange: [0, 11.5], yRange: [0, yMax],
      xLabel: 'y', yLabel: 'Conditional density'
    });
    if (chkMdn.checked) {
      axR.fillUnder(yEv, mdnSlice, 'rgba(37,99,235,0.18)', 0);
      axR.line(yEv, mdnSlice, '#2563eb', 2.8, []);
    }
    axR.line(yEv, trueSlice, '#111827', 2.2, [5, 4]);
    if (chkL2.checked) axR.vline(dat.l2[xi], '#d97706', 2.2, []);
    if (chkL1.checked) axR.vline(dat.l1[xi], '#059669', 2.2, []);
    axR.legend([
      { type: 'line', color: '#111827', dash: [5, 4], label: 'True LogNormal p(y | x)' },
      { type: 'line', color: '#2563eb', label: 'MDN q_φ(y | x)' },
      { type: 'line', color: '#d97706', label: 'L₂ mean' },
      { type: 'line', color: '#059669', label: 'L₁ median' }
    ], 'tr');
  }

  slX.addEventListener('input', draw);
  selSk.addEventListener('change', draw);
  [chkL2, chkL1, chkMdn].forEach(function(el) { el.addEventListener('change', draw); });
  document.getElementById('w10-reset').addEventListener('click', function() {
    slX.value = '28'; selSk.value = '1.00'; chkL2.checked = true; chkL1.checked = true; chkMdn.checked = true; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W10b: 1D Normalising Flow Warping Intuition (Slide 24)
    # =========================================================================
    "w10b_flow_1d": {
        "title": "Conditional Normalising Flow Change-of-Variables Intuition",
        "height": 395,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="w10b-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w10b-a">Bijection warp A(x)</label>
      <input type="range" id="w10b-a" min="0.00" max="1.10" step="0.02" value="0.74">
      <span class="iw-val" id="w10b-a-val">0.74</span>
    </div>
    <div class="iw-ctrl">
      <label for="w10b-sh">Shift b(x)</label>
      <input type="range" id="w10b-sh" min="-1.50" max="1.50" step="0.05" value="0.30">
      <span class="iw-val" id="w10b-sh-val">0.30</span>
    </div>
    <button type="button" class="iw-btn" id="w10b-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w10b-mono"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w10b-map" data-height="260"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w10b-base" data-height="260"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w10b-warp" data-height="260"></canvas></div>
  </div>
  <p class="iw-caption">
    Normalising flows push z &sim; N(0, 1) through an x-conditioned bijection &theta; = f<sub>&phi;</sub>(z; x) = z + b(x) + A(x) sin(z): exact density evaluation follows via the Jacobian factor |&part;f<sub>&phi;</sub>/&part;z|<sup>&minus;1</sup>.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w10b-root');
  if (!root) return;
  var slA = document.getElementById('w10b-a');
  var slSh = document.getElementById('w10b-sh');
  var mono = document.getElementById('w10b-mono');
  var zs = SBI.linspace(-6.0, 6.0, 420);

  function draw() {
    var A = parseFloat(slA.value), sh = parseFloat(slSh.value);
    document.getElementById('w10b-a-val').textContent = A.toFixed(2);
    document.getElementById('w10b-sh-val').textContent = sh.toFixed(2);
    mono.textContent = A < 1.0 ? 'Invertible bijection ✓ (|∂f/∂z| > 0)' : 'Non-invertible ✗ (folds where A ≥ 1)';
    mono.style.color = A < 1.0 ? '#059669' : '#dc2626';

    var pz = zs.map(function(z) { return SBI.normPdf(z, 0, 1.45); });
    var th = zs.map(function(z) { return z + sh + A * Math.sin(z); });
    var pt = zs.map(function(z, i) { return Math.min(0.85, pz[i] / Math.max(Math.abs(1 + A * Math.cos(z)), 0.03)); });

    var sM = SBI.prepCanvas(document.getElementById('w10b-map'), 260);
    var axM = SBI.createAxes(sM, { title: 'Bijection θ = f_φ(z; x)', xRange: [-6, 6], yRange: [-6, 6], xLabel: 'z', yLabel: 'θ' });
    for (var k = 0; k < 16; k++) {
      var zk = -5.5 + 11.0 * (k + 0.5) / 16, tk = zk + sh + A * Math.sin(zk);
      axM.line([zk, zk], [-6, tk], 'rgba(5,150,105,0.28)', 1.0, []);
      axM.line([-6, zk], [tk, tk], 'rgba(5,150,105,0.28)', 1.0, []);
    }
    axM.line(zs, th, '#7c3aed', 2.6, []);

    var sB = SBI.prepCanvas(document.getElementById('w10b-base'), 260);
    var axB = SBI.createAxes(sB, { title: 'Base p_Z(z)', xRange: [-6, 6], yRange: [0, 0.56], xLabel: 'z', yLabel: 'Density' });
    axB.fillUnder(zs, pz, 'rgba(37,99,235,0.20)', 0);
    axB.line(zs, pz, '#2563eb', 2.4, []);

    var sW = SBI.prepCanvas(document.getElementById('w10b-warp'), 260);
    var axW = SBI.createAxes(sW, { title: 'Warped q_φ(θ | x)', xRange: [-6, 6], yRange: [0, 0.56], xLabel: 'θ', yLabel: '' });
    if (A < 1.0) axW.fillUnder(th, pt, 'rgba(5,150,105,0.24)', 0);
    axW.line(th, pt, A < 1.0 ? '#059669' : '#dc2626', 2.6, []);
  }

  slA.addEventListener('input', draw);
  slSh.addEventListener('input', draw);
  document.getElementById('w10b-reset').addEventListener('click', function() { slA.value = '0.74'; slSh.value = '0.30'; draw(); });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W11: Amortized Galton-Board Pipeline Explorer (Slide 25)
    # =========================================================================
    "w11_amortized": {
        "title": "W11: Amortized Simulation-to-Inference Pipeline on Galton Boards",
        "height": 430,
        "data_keys": ["grid", "observations"],
        "html": """
<div class="iw-widget" id="w11-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w11-obs">New observed experiment</label>
      <select class="iw-select" id="w11-obs">
        <option value="0" selected>1. Mystery Machine (Observed)</option>
        <option value="1">2. Experiment B (Left Bias)</option>
        <option value="2">3. Experiment C (Weak Restoring)</option>
        <option value="3">4. Experiment D (Strong Centering)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w11-ref">
      <label for="w11-ref">Compare exact posterior</label>
    </div>
    <span class="iw-status" id="w11-stat">0 new simulations needed at test time!</span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w11-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w11-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Trained once on D = {(&theta;<sub>i</sub>, x<sub>i</sub>)}<sub>i=1</sub><sup>24,000</sup>, the amortised neural estimator maps <em>any</em> new 100-ball histogram x<sub>obs</sub> directly to q<sub>&phi;</sub>(&alpha;, &kappa; | x<sub>obs</sub>) in a single neural forward pass.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w11-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, gr = D.grid;
  var selO = document.getElementById('w11-obs');
  var chkR = document.getElementById('w11-ref');
  var bins = SBI.linspace(0, 12, 13);

  function draw() {
    var idx = parseInt(selO.value, 10);
    var obs = D.observations[idx];
    var npe = obs.post.NPE, ex = obs.post.Exact;
    var m = obs.metrics.NPE;
    document.getElementById('w11-stat').textContent =
      'NPE E[α|x]=' + m.mean_a.toFixed(2) + '±' + m.std_a.toFixed(2) + ',  E[κ|x]=' + m.mean_k.toFixed(2) + '±' + m.std_k.toFixed(2);

    var sL = SBI.prepCanvas(document.getElementById('w11-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: obs.label + ' (k̄=' + obs.mean_bin.toFixed(2) + ', SD=' + obs.std_bin.toFixed(2) + ')',
      xRange: [-0.7, 12.7], yRange: [0, 38],
      xLabel: 'Landing bin k', yLabel: 'Count (100 balls)'
    });
    axL.bars(bins, obs.x_obs, 0.74, 'rgba(31,41,55,0.82)', '#111827');

    var sR = SBI.prepCanvas(document.getElementById('w11-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: 'Instant amortised posterior q_φ(α, κ | x_obs)',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axR.heatmapAK(gr.alpha, gr.kappa, npe, '37,99,235', 0.48);
    axR.contourAK(gr.alpha, gr.kappa, npe, SBI.hpdLevel(npe, gr.da, gr.dk, 0.68), '#2563eb', 2.5, []);
    axR.contourAK(gr.alpha, gr.kappa, npe, SBI.hpdLevel(npe, gr.da, gr.dk, 0.95), '#2563eb', 1.6, [5, 4]);
    if (chkR.checked) {
      axR.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.68), '#111827', 2.0, []);
      axR.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.95), '#111827', 1.4, [4, 4]);
    }
    if (idx > 0) axR.star(obs.alpha_true, obs.kappa_true, '#dc2626', 7.2);
    var leg = [{ type: 'line', color: '#2563eb', label: 'Amortised NPE (68%/95% HPD)' }];
    if (chkR.checked) leg.push({ type: 'line', color: '#111827', label: 'Exact DP reference' });
    if (idx > 0) leg.push({ type: 'star', color: '#dc2626', label: 'True (α*, κ*)' });
    axR.legend(leg, 'tl');
  }

  selO.addEventListener('change', draw);
  chkR.addEventListener('change', draw);
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W12a: NPE on the Mystery Galton Board (Slide 27)
    # =========================================================================
    "w12a_npe_plinko": {
        "title": "W12 (NPE): Neural Posterior Estimation on the Galton Board",
        "height": 435,
        "data_keys": ["grid", "observations"],
        "html": """
<div class="iw-widget" id="w12a-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w12a-obs">Observation</label>
      <select class="iw-select" id="w12a-obs">
        <option value="0" selected>Mystery Machine (Observed)</option>
        <option value="1">Experiment B (Left Bias)</option>
        <option value="2">Experiment C (Weak Restoring)</option>
        <option value="3">Experiment D (Strong Centering)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w12a-ns">Posterior draws</label>
      <input type="range" id="w12a-ns" min="0" max="600" step="50" value="250">
      <span class="iw-val" id="w12a-ns-val">250</span>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w12a-ref" checked>
      <label for="w12a-ref">Show reference posterior</label>
    </div>
    <span class="iw-status" id="w12a-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w12a-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w12a-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Logit-transformed Mixture Density Network q<sub>&phi;</sub>(&alpha;, &kappa; | x): respects bounded prior support [&minus;1.5, 1.5] &times; [0, 3] via exact change-of-variables Jacobian and yields direct ancestral posterior draws.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w12a-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, gr = D.grid;
  var selO = document.getElementById('w12a-obs');
  var slNs = document.getElementById('w12a-ns');
  var chkR = document.getElementById('w12a-ref');

  function sampleGrid(postAK, count, seed) {
    var na = gr.alpha.length, nk = gr.kappa.length, tot = 0;
    var cdf = new Float64Array(na * nk);
    for (var i = 0; i < na; i++) for (var j = 0; j < nk; j++) { tot += postAK[i][j]; cdf[i * nk + j] = tot; }
    var rng = SBI.createRNG(seed), sa = [], sk = [];
    for (var s = 0; s < count; s++) {
      var u = rng() * tot, lo = 0, hi = cdf.length - 1;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      var ia = Math.floor(lo / nk), ik = lo % nk;
      sa.push(gr.alpha[ia] + (rng() - 0.5) * gr.da);
      sk.push(gr.kappa[ik] + (rng() - 0.5) * gr.dk);
    }
    return { a: sa, k: sk };
  }

  function draw() {
    var idx = parseInt(selO.value, 10), nS = parseInt(slNs.value, 10);
    document.getElementById('w12a-ns-val').textContent = String(nS);
    var obs = D.observations[idx], npe = obs.post.NPE, ex = obs.post.Exact;
    document.getElementById('w12a-stat').textContent =
      'TV(NPE, Exact) = ' + obs.metrics.NPE.tv.toFixed(3);

    var sL = SBI.prepCanvas(document.getElementById('w12a-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: '2D NPE joint posterior q_φ(α, κ | x_obs)',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axL.heatmapAK(gr.alpha, gr.kappa, npe, '37,99,235', 0.45);
    if (nS > 0) {
      var pts = sampleGrid(npe, nS, 77 + idx);
      axL.scatter(pts.a, pts.k, 'rgba(37,99,235,0.55)', 2.1);
    }
    axL.contourAK(gr.alpha, gr.kappa, npe, SBI.hpdLevel(npe, gr.da, gr.dk, 0.68), '#2563eb', 2.5, []);
    axL.contourAK(gr.alpha, gr.kappa, npe, SBI.hpdLevel(npe, gr.da, gr.dk, 0.95), '#2563eb', 1.5, [5, 4]);
    if (chkR.checked) {
      axL.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.68), '#111827', 2.0, []);
    }

    var mNpe = SBI.marginalsAK(npe, gr.da, gr.dk);
    var mEx = SBI.marginalsAK(ex, gr.da, gr.dk);
    var yMax = Math.max(Math.max.apply(null, mNpe.alpha), Math.max.apply(null, mNpe.kappa)) * 1.25 + 0.1;
    var sR = SBI.prepCanvas(document.getElementById('w12a-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: '1D Marginals: p(α | x_obs) & p(κ − 1.5 | x_obs)',
      xRange: [-1.5, 1.5], yRange: [0, yMax],
      xLabel: 'α (blue)  /  shifted κ − 1.5 (teal)', yLabel: 'Marginal density'
    });
    var kShift = gr.kappa.map(function(v) { return v - 1.5; });
    axR.fillUnder(gr.alpha, mNpe.alpha, 'rgba(37,99,235,0.16)', 0);
    axR.line(gr.alpha, mNpe.alpha, '#2563eb', 2.7, []);
    axR.line(kShift, mNpe.kappa, '#0d9488', 2.7, []);
    if (chkR.checked) {
      axR.line(gr.alpha, mEx.alpha, '#111827', 2.0, [5, 4]);
      axR.line(kShift, mEx.kappa, '#475569', 2.0, [3, 3]);
    }
    axR.legend([
      { type: 'line', color: '#2563eb', label: 'NPE marginal α' },
      { type: 'line', color: '#0d9488', label: 'NPE marginal κ (shifted −1.5)' },
      { type: 'line', color: '#111827', dash: [5, 4], label: 'Exact DP marginals' }
    ], 'tr');
  }

  selO.addEventListener('change', draw);
  slNs.addEventListener('input', draw);
  chkR.addEventListener('change', draw);
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W12b: NLE Surrogate Likelihood & Live Prior-Swap Explorer (Slide 28)
    # =========================================================================
    "w12b_nle_prior": {
        "title": "W12 (NLE): Structured Surrogate Likelihood & Live Prior Swapping",
        "height": 440,
        "data_keys": ["grid", "observations", "nle_bin_grid"],
        "html": """
<div class="iw-widget" id="w12b-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w12b-prior">Prior p&prime;(&alpha;, &kappa;) after training</label>
      <select class="iw-select" id="w12b-prior">
        <option value="flat" selected>Original Uniform [&minus;1.5, 1.5] &times; [0, 3]</option>
        <option value="gauss_a">New calibration prior: &alpha; &sim; N(&mu;<sub>&alpha;</sub>, 0.25&sup2;)</option>
        <option value="low_k">Physical prior favouring weak centering &kappa; &sim; Exp(1.2)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w12b-mua">Prior mean &mu;<sub>&alpha;</sub></label>
      <input type="range" id="w12b-mua" min="-1.00" max="1.00" step="0.05" value="0.10">
      <span class="iw-val" id="w12b-mua-val">0.10</span>
    </div>
    <div class="iw-ctrl">
      <label for="w12b-ka">Probe (&alpha;, &kappa;) bin probs</label>
      <input type="range" id="w12b-ka" min="0" max="14" step="1" value="10" style="width:85px;">
    </div>
    <button type="button" class="iw-btn" id="w12b-reset">&#x21ba; Reset</button>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w12b-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w12b-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    <strong>Why choose NLE?</strong> Because q<sub>&psi;</sub>(x | &theta;) approximates the <em>likelihood</em> independently of the prior, you can swap in a new prior p&prime;(&theta;) or multiply independent experiments <em>without retraining the neural network</em>!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w12b-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, gr = D.grid, obs = D.observations[0], bg = D.nle_bin_grid;
  var selP = document.getElementById('w12b-prior');
  var slMua = document.getElementById('w12b-mua');
  var slPr = document.getElementById('w12b-ka');
  var bins = SBI.linspace(0, 12, 13);

  function draw() {
    var pKind = selP.value, mua = parseFloat(slMua.value), pIdx = parseInt(slPr.value, 10);
    document.getElementById('w12b-mua-val').textContent = mua.toFixed(2);
    slMua.disabled = (pKind !== 'gauss_a');

    /* Left: NLE predicted bin probabilities vs exact DP at probed (alpha, kappa) */
    var aProbe = bg.alpha[pIdx], kProbe = bg.kappa[7];
    var nleBins = bg.probs[pIdx][7];
    var dpBins = SBI.plinkoProbs(aProbe, kProbe, 12);
    var sL = SBI.prepCanvas(document.getElementById('w12b-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: 'Structured NLE π̂_ψ(k | α=' + aProbe.toFixed(2) + ', κ=' + kProbe.toFixed(2) + ')',
      xRange: [-0.7, 12.7], yRange: [0, 0.34],
      xLabel: 'Landing bin k', yLabel: 'Bin probability'
    });
    axL.bars(bins, nleBins, 0.72, 'rgba(5,150,105,0.55)', '#047857');
    axL.line(bins, dpBins, '#111827', 2.3, [5, 4]);
    axL.legend([
      { type: 'bar', color: 'rgba(5,150,105,0.65)', label: 'Learned NLE softmax π̂_ψ(k | θ)' },
      { type: 'line', color: '#111827', dash: [5, 4], label: 'True simulator probability' }
    ], 'tl');

    /* Right: Multiply cached NLE log-likelihood surface by chosen new prior p'(theta)! */
    var na = gr.alpha.length, nk = gr.kappa.length;
    var postNew = new Array(na), sum = 0.0;
    for (var i = 0; i < na; i++) {
      postNew[i] = new Float64Array(nk);
      var av = gr.alpha[i];
      for (var j = 0; j < nk; j++) {
        var kv = gr.kappa[j];
        var logPrior = 0.0;
        if (pKind === 'gauss_a') logPrior = -0.5 * Math.pow((av - mua) / 0.25, 2);
        else if (pKind === 'low_k') logPrior = -1.2 * kv;
        var v = Math.exp(obs.nle_loglike[i][j] + logPrior);
        postNew[i][j] = v;
        sum += v;
      }
    }
    var norm = sum * gr.da * gr.dk;
    for (var i2 = 0; i2 < na; i2++) for (var j2 = 0; j2 < nk; j2++) postNew[i2][j2] /= norm;

    var sR = SBI.prepCanvas(document.getElementById('w12b-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: 'Posterior p′(θ | x_obs) ∝ q_ψ(x_obs | θ) p′(θ) (0 retraining!)',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axR.heatmapAK(gr.alpha, gr.kappa, postNew, '5,150,105', 0.48);
    axR.contourAK(gr.alpha, gr.kappa, obs.post.NLE, SBI.hpdLevel(obs.post.NLE, gr.da, gr.dk, 0.68), '#94a3b8', 1.8, [4, 4]);
    axR.contourAK(gr.alpha, gr.kappa, postNew, SBI.hpdLevel(postNew, gr.da, gr.dk, 0.68), '#059669', 2.6, []);
    axR.contourAK(gr.alpha, gr.kappa, postNew, SBI.hpdLevel(postNew, gr.da, gr.dk, 0.95), '#059669', 1.6, [5, 4]);
    axR.legend([
      { type: 'line', color: '#059669', label: 'NLE × active prior p′(θ)' },
      { type: 'line', color: '#94a3b8', dash: [4, 4], label: 'NLE × flat prior (reference)' }
    ], 'tl');
  }

  selP.addEventListener('change', draw);
  slMua.addEventListener('input', draw);
  slPr.addEventListener('input', draw);
  document.getElementById('w12b-reset').addEventListener('click', function() {
    selP.value = 'flat'; slMua.value = '0.10'; slPr.value = '10'; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W13: NRE Joint-vs-Independent Classifier Explorer (Slide 29)
    # =========================================================================
    "w13_nre_classifier": {
        "title": "W13: Neural Ratio Estimation via Joint-vs-Marginal Classification",
        "height": 440,
        "data_keys": ["grid", "observations", "abc_bank"],
        "html": """
<div class="iw-widget" id="w13-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w13-ai">Probe &alpha;</label>
      <input type="range" id="w13-ai" min="0" max="40" step="1" value="28">
      <span class="iw-val" id="w13-a-val">0.60</span>
    </div>
    <div class="iw-ctrl">
      <label for="w13-ki">Probe &kappa;</label>
      <input type="range" id="w13-ki" min="0" max="40" step="1" value="16">
      <span class="iw-val" id="w13-k-val">1.20</span>
    </div>
    <div class="iw-ctrl">
      <label for="w13-surf">Surface</label>
      <select class="iw-select" id="w13-surf">
        <option value="prob" selected>Classifier D<sub>&omega;</sub>(&theta;, x<sub>obs</sub>) &isin; (0, 1)</option>
        <option value="post">Ratio posterior p(&theta;) r<sub>&omega;</sub>(&theta;, x<sub>obs</sub>)</option>
      </select>
    </div>
    <button type="button" class="iw-btn" id="w13-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w13-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w13-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w13-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Trained on balanced joint pairs (y=1) vs shuffled independent pairs (y=0), the classifier logit f<sub>&omega;</sub>(&theta;, x<sub>obs</sub>) = log[D<sub>&omega;</sub>/(1 &minus; D<sub>&omega;</sub>)] directly estimates log[p(x<sub>obs</sub> | &theta;) / p(x<sub>obs</sub>)] without normalizing any density!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w13-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, gr = D.grid, obs = D.observations[0];
  var slA = document.getElementById('w13-ai');
  var slK = document.getElementById('w13-ki');
  var selS = document.getElementById('w13-surf');
  var bins = SBI.linspace(0, 12, 13);

  function draw() {
    var ia = parseInt(slA.value, 10), ik = parseInt(slK.value, 10);
    var av = gr.alpha[ia], kv = gr.kappa[ik];
    document.getElementById('w13-a-val').textContent = av.toFixed(2);
    document.getElementById('w13-k-val').textContent = kv.toFixed(2);

    var logit = obs.nre_logit[ia][ik];
    var probD = SBI.sigmoid(logit);
    var ratio = Math.exp(Math.max(-20, Math.min(20, logit)));
    document.getElementById('w13-stat').textContent =
      'D_ω = ' + probD.toFixed(3) + '  |  log r_ω = ' + logit.toFixed(2) + '  |  r_ω = ' + ratio.toFixed(2);

    /* Left: Observed x_obs vs expected histogram at probed theta — visualising compatible vs mismatched pair */
    var rng = SBI.createRNG(101);
    var sim = SBI.simPlinko(av, kv, 100, 12, rng, 0);
    var sL = SBI.prepCanvas(document.getElementById('w13-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: probD > 0.5 ? 'Compatible pair (D_ω > 0.5 → r_ω > 1)' : 'Mismatched pair (D_ω < 0.5 → r_ω < 1)',
      xRange: [-0.7, 12.7], yRange: [0, 38],
      xLabel: 'Landing bin k', yLabel: 'Count (100 balls)'
    });
    axL.bars(bins, obs.x_obs, 0.38, 'rgba(31,41,55,0.84)', '#111827', -0.20);
    axL.bars(bins, sim.counts, 0.38, probD > 0.5 ? 'rgba(124,58,237,0.68)' : 'rgba(220,38,38,0.52)', '#6d28d9', 0.20);
    axL.legend([
      { type: 'bar', color: 'rgba(31,41,55,0.85)', label: 'Observed x_obs' },
      { type: 'bar', color: probD > 0.5 ? 'rgba(124,58,237,0.75)' : 'rgba(220,38,38,0.65)', label: 'Sim at probed θ (' + av.toFixed(2) + ', ' + kv.toFixed(2) + ')' }
    ], 'tl');

    /* Right: 2D NRE classifier probability or normalized ratio posterior */
    var surf2D = obs.post.NRE;
    if (selS.value === 'prob') {
      surf2D = obs.nre_logit.map(function(row) { return row.map(SBI.sigmoid); });
    }
    var sR = SBI.prepCanvas(document.getElementById('w13-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: selS.value === 'prob' ? 'Classifier score D_ω(α, κ, x_obs)' : 'NRE posterior p(θ) r_ω(θ, x_obs)',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axR.heatmapAK(gr.alpha, gr.kappa, surf2D, '124,58,237', 0.52);
    axR.contourAK(gr.alpha, gr.kappa, obs.post.NRE, SBI.hpdLevel(obs.post.NRE, gr.da, gr.dk, 0.68), '#7c3aed', 2.4, []);
    axR.contourAK(gr.alpha, gr.kappa, obs.post.Exact, SBI.hpdLevel(obs.post.Exact, gr.da, gr.dk, 0.68), '#111827', 1.8, [4, 4]);
    axR.cross(av, kv, '#dc2626', 6.5, 2.8);
    axR.legend([
      { type: 'line', color: '#7c3aed', label: 'NRE 68% HPD' },
      { type: 'line', color: '#111827', dash: [4, 4], label: 'Exact DP 68% HPD' }
    ], 'tl');
  }

  [slA, slK].forEach(function(el) { el.addEventListener('input', draw); });
  selS.addEventListener('change', draw);
  document.getElementById('w13-reset').addEventListener('click', function() {
    slA.value = '28'; slK.value = '16'; selS.value = 'prob'; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W12: Unified ABC vs NPE vs NLE vs NRE vs Exact Comparison (Slides 30 & 31)
    # =========================================================================
    "w12_unified_compare": {
        "title": "W12: Unified Galton-Board Comparison (ABC vs NPE vs NLE vs NRE vs Exact)",
        "height": 445,
        "data_keys": ["grid", "observations", "abc_bank"],
        "html": """
<div class="iw-widget" id="w12-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w12-obs">Experiment</label>
      <select class="iw-select" id="w12-obs">
        <option value="0" selected>Mystery Machine (Observed)</option>
        <option value="1">Experiment B (Left Bias)</option>
        <option value="2">Experiment C (Weak Restoring)</option>
        <option value="3">Experiment D (Strong Centering)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w12-cred">Credible level</label>
      <input type="range" id="w12-cred" min="0.50" max="0.95" step="0.05" value="0.68" style="width:92px;">
      <span class="iw-val" id="w12-cred-val">68% HPD</span>
    </div>
    <div class="iw-ctrl"><input type="checkbox" id="w12-npe" checked><label for="w12-npe" style="color:#2563eb;">NPE</label></div>
    <div class="iw-ctrl"><input type="checkbox" id="w12-nle" checked><label for="w12-nle" style="color:#059669;">NLE</label></div>
    <div class="iw-ctrl"><input type="checkbox" id="w12-nre" checked><label for="w12-nre" style="color:#7c3aed;">NRE</label></div>
    <div class="iw-ctrl"><input type="checkbox" id="w12-abc" checked><label for="w12-abc" style="color:#d97706;">ABC</label></div>
    <div class="iw-ctrl"><input type="checkbox" id="w12-ex" checked><label for="w12-ex" style="color:#111827;">Exact</label></div>
    <div class="iw-ctrl"><input type="checkbox" id="w12-tr" checked><label for="w12-tr" style="color:#dc2626;">Truth &theta;<sup>&starf;</sup></label></div>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w12-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w12-marga" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w12-margk" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption" id="w12-cap"></p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w12-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, gr = D.grid, bank = D.abc_bank;
  var selO = document.getElementById('w12-obs');
  var slC = document.getElementById('w12-cred');
  var chks = {
    NPE: document.getElementById('w12-npe'),
    NLE: document.getElementById('w12-nle'),
    NRE: document.getElementById('w12-nre'),
    ABC: document.getElementById('w12-abc'),
    Exact: document.getElementById('w12-ex'),
    truth: document.getElementById('w12-tr')
  };

  function computeAbcKde(xObs) {
    var accA = [], accK = [];
    for (var i = 0; i < bank.alphas.length; i++) {
      if (SBI.histDistance(bank.hists[i], xObs, 'wasserstein') < 0.36) {
        accA.push(bank.alphas[i]); accK.push(bank.kappas[i]);
      }
    }
    return SBI.kde2DAK(accA, accK, gr.alpha, gr.kappa, gr.da, gr.dk);
  }

  function draw() {
    var idx = parseInt(selO.value, 10), cred = parseFloat(slC.value);
    document.getElementById('w12-cred-val').textContent = Math.round(cred * 100) + '% HPD';
    var obs = D.observations[idx];
    var abcGrid = computeAbcKde(obs.x_obs);

    var methods = [
      { key: 'Exact', color: '#111827', dash: [5, 4], lw: 2.3, post: obs.post.Exact },
      { key: 'NPE',   color: '#2563eb', dash: [],     lw: 2.5, post: obs.post.NPE },
      { key: 'NLE',   color: '#059669', dash: [],     lw: 2.5, post: obs.post.NLE },
      { key: 'NRE',   color: '#7c3aed', dash: [],     lw: 2.5, post: obs.post.NRE },
      { key: 'ABC',   color: '#d97706', dash: [3, 3], lw: 2.2, post: abcGrid }
    ];

    var s2 = SBI.prepCanvas(document.getElementById('w12-left'), 305);
    var ax2 = SBI.createAxes(s2, {
      title: 'Joint (' + Math.round(cred * 100) + '% HPD) on identical axes',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    if (chks.Exact.checked) ax2.heatmapAK(gr.alpha, gr.kappa, obs.post.Exact, '15,23,42', 0.16);
    var leg = [];
    methods.forEach(function(m) {
      if (!chks[m.key].checked) return;
      var lv = SBI.hpdLevel(m.post, gr.da, gr.dk, cred);
      ax2.contourAK(gr.alpha, gr.kappa, m.post, lv, m.color, m.lw, m.dash);
      leg.push({ type: 'line', color: m.color, dash: m.dash, label: m.key });
    });
    if (chks.truth.checked) {
      ax2.star(obs.alpha_true, obs.kappa_true, '#dc2626', 7.5);
      leg.push({ type: 'star', color: '#dc2626', label: 'θ*=(' + obs.alpha_true.toFixed(2) + ',' + obs.kappa_true.toFixed(2) + ')' });
    }
    ax2.legend(leg, 'tl');

    /* 1D Marginals */
    var margs = {};
    var maxA = 0.5, maxK = 0.5;
    methods.forEach(function(m) {
      margs[m.key] = SBI.marginalsAK(m.post, gr.da, gr.dk);
      if (chks[m.key].checked) {
        maxA = Math.max(maxA, Math.max.apply(null, margs[m.key].alpha));
        maxK = Math.max(maxK, Math.max.apply(null, margs[m.key].kappa));
      }
    });

    var sA = SBI.prepCanvas(document.getElementById('w12-marga'), 305);
    var axA = SBI.createAxes(sA, {
      title: 'Marginal p(α | x_obs)', xRange: [gr.alpha_min, gr.alpha_max], yRange: [0, maxA * 1.2],
      xLabel: 'Bias α', yLabel: 'Density'
    });
    var sK = SBI.prepCanvas(document.getElementById('w12-margk'), 305);
    var axK = SBI.createAxes(sK, {
      title: 'Marginal p(κ | x_obs)', xRange: [gr.kappa_min, gr.kappa_max], yRange: [0, maxK * 1.2],
      xLabel: 'Restoring κ', yLabel: 'Density'
    });
    methods.forEach(function(m) {
      if (!chks[m.key].checked) return;
      axA.line(gr.alpha, margs[m.key].alpha, m.color, m.lw, m.dash);
      axK.line(gr.kappa, margs[m.key].kappa, m.color, m.lw, m.dash);
    });
    if (chks.truth.checked) {
      axA.vline(obs.alpha_true, '#dc2626', 1.8, [3, 3]);
      axK.vline(obs.kappa_true, '#dc2626', 1.8, [3, 3]);
    }

    var mt = obs.metrics;
    document.getElementById('w12-cap').textContent =
      'Total Variation to Exact DP — NPE: ' + mt.NPE.tv.toFixed(3) + '  |  NLE: ' + mt.NLE.tv.toFixed(3) +
      '  |  NRE: ' + mt.NRE.tv.toFixed(3) + '. True θ* lies comfortably inside the 68% credible contour; finite N=100 sample noise shifts the posterior mode slightly from θ*.';
  }

  selO.addEventListener('change', draw);
  slC.addEventListener('input', draw);
  Object.keys(chks).forEach(function(k) { chks[k].addEventListener('change', draw); });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W14: Posterior Predictive Checks Explorer (Slide 32)
    # =========================================================================
    "w14_ppc": {
        "title": "W14: Galton-Board Posterior Predictive Checks (PPC)",
        "height": 435,
        "data_keys": ["grid", "observations"],
        "html": """
<div class="iw-widget" id="w14-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w14-meth">Posterior</label>
      <select class="iw-select" id="w14-meth">
        <option value="NPE" selected>NPE posterior</option>
        <option value="NLE">NLE posterior</option>
        <option value="NRE">NRE posterior</option>
        <option value="Exact">Exact DP posterior</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w14-stat-sel">Test statistic T(x)</label>
      <select class="iw-select" id="w14-stat-sel">
        <option value="mean">Mean landing bin k&#x0304;</option>
        <option value="std" selected>Bin spread SD(k)</option>
        <option value="max">Peak bin count max<sub>k</sub> n<sub>k</sub></option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w14-nrep">Predictive draws</label>
      <input type="range" id="w14-nrep" min="50" max="400" step="50" value="200">
      <span class="iw-val" id="w14-nrep-val">200</span>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w14-mis">
      <label for="w14-mis">Corrupt x<sub>obs</sub> (sticky-peg misspecification)</label>
    </div>
    <span class="iw-status" id="w14-pval"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w14-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w14-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Draw &theta;<sup>(s)</sup> &sim; q(&theta; | x<sub>obs</sub>) and re-simulate x&#x0303;<sup>(s)</sup> &sim; p(x | &theta;<sup>(s)</sup>). Toggle <strong>Corrupt x<sub>obs</sub></strong> to see how PPCs detect model misspecification even when parameter inference looks confident!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w14-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, gr = D.grid, baseObs = D.observations[0];
  var selM = document.getElementById('w14-meth');
  var selT = document.getElementById('w14-stat-sel');
  var slN = document.getElementById('w14-nrep');
  var chkMis = document.getElementById('w14-mis');
  var bins = SBI.linspace(0, 12, 13);

  function calcStat(h, kind) {
    if (kind === 'max') return Math.max.apply(null, h);
    var st = SBI.histStats(h);
    return kind === 'mean' ? st.mean : st.std;
  }

  function draw() {
    var mKey = selM.value, tKind = selT.value, nRep = parseInt(slN.value, 10);
    document.getElementById('w14-nrep-val').textContent = String(nRep);

    /* Optional corrupted observation: bimodal split from a broken/sticky center splitter */
    var xObs = chkMis.checked ? [2,6,14,15,6,2,1,2,8,18,16,8,2] : baseObs.x_obs;
    var postAK = baseObs.post[mKey];

    var na = gr.alpha.length, nk = gr.kappa.length, tot = 0;
    var cdf = new Float64Array(na * nk);
    for (var i = 0; i < na; i++) for (var j = 0; j < nk; j++) { tot += postAK[i][j]; cdf[i * nk + j] = tot; }

    var rng = SBI.createRNG(202605);
    var reps = [], tVals = new Float64Array(nRep);
    for (var r = 0; r < nRep; r++) {
      var u = rng() * tot, lo = 0, hi = cdf.length - 1;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      var ia = Math.floor(lo / nk), ik = lo % nk;
      var sim = SBI.simPlinko(gr.alpha[ia], gr.kappa[ik], 100, 12, rng, 0);
      reps.push(sim.counts);
      tVals[r] = calcStat(sim.counts, tKind);
    }

    var q05 = new Array(13), q95 = new Array(13), med = new Array(13);
    for (var b = 0; b < 13; b++) {
      var col = reps.map(function(rr) { return rr[b]; }).sort(function(x, y) { return x - y; });
      q05[b] = col[Math.floor(0.05 * (nRep - 1))];
      med[b] = col[Math.floor(0.50 * (nRep - 1))];
      q95[b] = col[Math.floor(0.95 * (nRep - 1))];
    }

    var tObs = calcStat(xObs, tKind);
    var ge = 0;
    for (var r2 = 0; r2 < nRep; r2++) if (tVals[r2] >= tObs) ge++;
    var pPred = ge / nRep;
    document.getElementById('w14-pval').textContent =
      'P(T(x̃) ≥ T(x_obs)) = ' + pPred.toFixed(2) + (pPred < 0.02 || pPred > 0.98 ? '  [MISSPECIFIED ✗]' : '  [compatible ✓]');

    var sL = SBI.prepCanvas(document.getElementById('w14-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: '90% posterior predictive envelope vs x_obs',
      xRange: [-0.6, 12.6], yRange: [0, 38],
      xLabel: 'Landing bin k', yLabel: 'Ball count'
    });
    axL.band(bins, q05, q95, 'rgba(37,99,235,0.22)');
    for (var j2 = 0; j2 < Math.min(6, nRep); j2++) axL.line(bins, reps[j2], 'rgba(37,99,235,0.28)', 1.1, []);
    axL.line(bins, med, '#2563eb', 2.6, []);
    axL.line(bins, xObs, '#111827', 2.4, [4, 3]);
    axL.scatter(bins, xObs, chkMis.checked ? '#dc2626' : '#111827', 3.8);
    axL.legend([
      { type: 'bar', color: 'rgba(37,99,235,0.32)', label: '90% PPC band (' + mKey + ')' },
      { type: 'dot', color: chkMis.checked ? '#dc2626' : '#111827', label: 'Observed x_obs' }
    ], 'tl');

    var tMin = Math.min(tObs, Math.min.apply(null, tVals)) * 0.90;
    var tMax = Math.max(tObs, Math.max.apply(null, tVals)) * 1.10 + 0.2;
    var h = SBI.histogramDensity(tVals, tMin, tMax, 22);
    var sR = SBI.prepCanvas(document.getElementById('w14-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: 'Discrepancy distribution T(x̃) vs T(x_obs)',
      xRange: [tMin, tMax], yRange: [0, Math.max.apply(null, h.ys) * 1.25 + 0.1],
      xLabel: 'Test statistic T(x)', yLabel: 'Predictive density'
    });
    axR.bars(h.xs, h.ys, h.binW, 'rgba(37,99,235,0.55)', '#1d4ed8');
    axR.vline(tObs, '#dc2626', 2.6, []);
    axR.legend([
      { type: 'bar', color: 'rgba(37,99,235,0.65)', label: 'Replicated T(x̃)' },
      { type: 'line', color: '#dc2626', label: 'Observed T(x_obs) = ' + tObs.toFixed(2) }
    ], 'tl');
  }

  selM.addEventListener('change', draw);
  selT.addEventListener('change', draw);
  slN.addEventListener('input', draw);
  chkMis.addEventListener('change', draw);
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W15: Empirical Coverage & SBC Explorer on Galton Board (Slide 33)
    # =========================================================================
    "w15_coverage": {
        "title": "W15: Galton-Board Empirical Coverage & Simulation-Based Calibration",
        "height": 440,
        "data_keys": ["coverage"],
        "html": """
<div class="iw-widget" id="w15-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w15-meth">Estimator</label>
      <select class="iw-select" id="w15-meth">
        <option value="all" selected>Compare NPE / NLE / NRE / ABC</option>
        <option value="NPE">NPE only</option>
        <option value="NLE">NLE only</option>
        <option value="NRE">NRE only</option>
        <option value="ABC">ABC (fixed &varepsilon;=0.35)</option>
        <option value="synth">Synthetic miscalibration explorer</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w15-n">Validation runs</label>
      <input type="range" id="w15-n" min="40" max="200" step="20" value="200">
      <span class="iw-val" id="w15-n-val">200</span>
    </div>
    <div class="iw-ctrl" id="w15-syn-box" style="display:none;">
      <label for="w15-scale">Width scale s</label>
      <input type="range" id="w15-scale" min="0.50" max="1.80" step="0.05" value="0.65" style="width:90px;">
      <span class="iw-val" id="w15-scale-val">0.65</span>
    </div>
    <span class="iw-status" id="w15-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w15-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w15-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Evaluated on 200 strictly independent unseen Galton-board experiments (&theta;<sup>&starf;</sup>, x) &sim; p(&theta;)p(x | &theta;). Neural estimators hug the diagonal 95% binomial band; fixed-&varepsilon; ABC sits above the diagonal (conservative / overdispersed due to finite tolerance!).
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w15-root');
  if (!root || !window.SBI_DATA) return;
  var C = window.SBI_DATA.coverage;
  var selM = document.getElementById('w15-meth');
  var slN = document.getElementById('w15-n');
  var synBox = document.getElementById('w15-syn-box');
  var slSc = document.getElementById('w15-scale');
  var nom = SBI.linspace(0, 1, 41);

  function empCurve(ranks, nUse) {
    return nom.map(function(a) {
      var cnt = 0;
      for (var i = 0; i < nUse; i++) if (ranks[i] <= a) cnt++;
      return cnt / nUse;
    });
  }

  function pitHist(pits, nUse, nBins) {
    var counts = new Array(nBins).fill(0);
    for (var i = 0; i < nUse; i++) {
      var b = Math.min(nBins - 1, Math.max(0, Math.floor(pits[i] * nBins)));
      counts[b]++;
    }
    return counts.map(function(c) { return c * nBins / nUse; });
  }

  function draw() {
    var mode = selM.value, nUse = parseInt(slN.value, 10), sc = parseFloat(slSc.value);
    document.getElementById('w15-n-val').textContent = String(nUse);
    document.getElementById('w15-scale-val').textContent = sc.toFixed(2);
    synBox.style.display = mode === 'synth' ? 'inline-flex' : 'none';

    var loBand = nom.map(function(a) { return Math.max(0, a - 1.96 * Math.sqrt(a * (1 - a) / nUse)); });
    var hiBand = nom.map(function(a) { return Math.min(1, a + 1.96 * Math.sqrt(a * (1 - a) / nUse)); });

    var sL = SBI.prepCanvas(document.getElementById('w15-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: 'Empirical 2D HPD Credible Coverage',
      xRange: [0, 1], yRange: [0, 1],
      xLabel: 'Nominal credibility 1 − α', yLabel: 'Empirical coverage'
    });
    axL.band(nom, loBand, hiBand, 'rgba(148,163,184,0.26)');
    axL.line([0, 1], [0, 1], '#475569', 1.8, [5, 4]);

    var sR = SBI.prepCanvas(document.getElementById('w15-right'), 300);
    var nB = 10, bx = SBI.linspace(0.05, 0.95, nB);
    var axR = SBI.createAxes(sR, {
      title: '1D Simulation-Based Calibration (α PIT ranks)',
      xRange: [0, 1], yRange: [0, 2.3],
      xLabel: 'PIT / Fractional rank u ∈ [0, 1]', yLabel: 'Normalised rank density'
    });
    var ciH = 1.96 * Math.sqrt((1 / nB) * (1 - 1 / nB) / nUse) * nB;
    axR.band([0, 1], [1 - ciH, 1 - ciH], [1 + ciH, 1 + ciH], 'rgba(148,163,184,0.25)');
    axR.hline(1.0, '#475569', 1.8, [5, 4]);

    if (mode === 'synth') {
      var rng = SBI.createRNG(71), synHpd = [], synPit = [];
      for (var k = 0; k < nUse; k++) {
        var z = SBI.gaussN(rng);
        synPit.push(SBI.gaussCdf(z / sc));
        synHpd.push(2 * SBI.gaussCdf(Math.abs(z) / sc) - 1);
      }
      axL.line(nom, empCurve(synHpd, nUse), '#dc2626', 2.8, []);
      axR.bars(bx, pitHist(synPit, nUse, nB), 0.082, 'rgba(220,38,38,0.58)', '#b91c1c');
      document.getElementById('w15-stat').textContent =
        sc < 0.9 ? 'Overconfident (∪-shaped SBC, under-covers!)' : sc > 1.1 ? 'Overdispersed (∩-shaped SBC, over-covers)' : 'Calibrated ✓';
    } else {
      var specs = [
        { k: 'NPE', c: '#2563eb' },
        { k: 'NLE', c: '#059669' },
        { k: 'NRE', c: '#7c3aed' },
        { k: 'ABC', c: '#d97706' }
      ];
      var leg = [];
      specs.forEach(function(sp) {
        if (mode !== 'all' && mode !== sp.k) return;
        var emp = empCurve(C.methods[sp.k].hpd, nUse);
        axL.line(nom, emp, sp.c, 2.6, sp.k === 'ABC' ? [5, 3] : []);
        leg.push({ type: 'line', color: sp.c, dash: sp.k === 'ABC' ? [5, 3] : [], label: sp.k });
      });
      axL.legend(leg, 'tl');

      var pitTarget = (mode === 'all') ? 'NPE' : mode;
      var colMap = { NPE: 'rgba(37,99,235,0.60)', NLE: 'rgba(5,150,105,0.60)', NRE: 'rgba(124,58,237,0.60)', ABC: 'rgba(217,119,6,0.60)' };
      axR.bars(bx, pitHist(C.methods[pitTarget].pit_a, nUse, nB), 0.082, colMap[pitTarget], '#1e293b');
      axR.legend([{ type: 'bar', color: colMap[pitTarget], label: pitTarget + ' 1D SBC ranks' }], 'tl');

      var c68 = empCurve(C.methods[pitTarget].hpd, nUse)[27];
      document.getElementById('w15-stat').textContent = pitTarget + ' 68% HPD empirical coverage = ' + (100 * c68).toFixed(1) + '%';
    }
  }

  selM.addEventListener('change', draw);
  slN.addEventListener('input', draw);
  slSc.addEventListener('input', draw);
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # Backup B8: Live SGD Conditional Density Trainer (Kept animation!)
    # =========================================================================
    "wb_mlp_sgd": {
        "title": "Backup B8: Live Neural Conditional Density SGD Trainer",
        "height": 420,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="wb1-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="wb1-w">Hidden width H</label>
      <input type="range" id="wb1-w" min="8" max="36" step="4" value="20">
      <span class="iw-val" id="wb1-w-val">20</span>
    </div>
    <div class="iw-ctrl">
      <label for="wb1-lr">log<sub>10</sub> &eta;</label>
      <input type="range" id="wb1-lr" min="-3.2" max="-1.6" step="0.1" value="-2.2">
      <span class="iw-val" id="wb1-lr-val">-2.2</span>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="wb1-train">&#9654; Train</button>
    <button type="button" class="iw-btn" id="wb1-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="wb1-stat">step 0</span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="wb1-left" data-height="285"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wb1-right" data-height="285"></canvas></div>
  </div>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('wb1-root');
  if (!root) return;
  var slW = document.getElementById('wb1-w'), slLr = document.getElementById('wb1-lr');
  var btnT = document.getElementById('wb1-train');
  var xL = SBI.linspace(-1, 1, 120), dX = [], dY = [], hist = [], step = 0, run = false, raf = null, net = null;

  function init() {
    run = false; btnT.textContent = '▶ Train'; if (raf) cancelAnimationFrame(raf);
    var rng = SBI.createRNG(88), H = parseInt(slW.value, 10);
    dX = []; dY = []; hist = []; step = 0;
    for (var i = 0; i < 140; i++) {
      var x = -1 + 2 * rng();
      dX.push(x); dY.push(Math.sin(Math.PI * x) + (0.08 + 0.42 * Math.max(0, x + 0.2)) * SBI.gaussN(rng));
    }
    var w1 = new Float64Array(H), b1 = new Float64Array(H), w2m = new Float64Array(H), w2s = new Float64Array(H);
    for (var j = 0; j < H; j++) {
      w1[j] = 1.6 * SBI.gaussN(rng); b1[j] = 0.8 * SBI.gaussN(rng);
      w2m[j] = 0.15 * SBI.gaussN(rng); w2s[j] = 0.08 * SBI.gaussN(rng);
    }
    net = { H: H, w1: w1, b1: b1, w2m: w2m, w2s: w2s, bm: 0, bs: -0.7 };
    document.getElementById('wb1-stat').textContent = 'step 0';
    draw();
  }

  function fwd(x) {
    var H = net.H, h = new Float64Array(H), mu = net.bm, ls = net.bs;
    for (var j = 0; j < H; j++) { var z = net.w1[j] * x + net.b1[j]; h[j] = z > 0 ? z : 0; mu += net.w2m[j] * h[j]; ls += net.w2s[j] * h[j]; }
    ls = Math.max(-2.6, Math.min(1.0, ls));
    return { mu: mu, ls: ls, h: h };
  }

  function stepSGD() {
    var lr = Math.pow(10, parseFloat(slLr.value)), N = dX.length, H = net.H, tot = 0;
    var gw1 = new Float64Array(H), gb1 = new Float64Array(H), gm = new Float64Array(H), gs = new Float64Array(H), gbm = 0, gbs = 0;
    for (var n = 0; n < N; n++) {
      var f = fwd(dX[n]), iv = Math.exp(-2 * f.ls), d = f.mu - dY[n];
      tot += 0.5 * d * d * iv + f.ls + 0.919;
      var dMu = (d * iv) / N, dLs = (1 - d * d * iv) / N;
      gbm += dMu; gbs += dLs;
      for (var j = 0; j < H; j++) {
        gm[j] += dMu * f.h[j]; gs[j] += dLs * f.h[j];
        if (f.h[j] > 0) { var dh = dMu * net.w2m[j] + dLs * net.w2s[j]; gb1[j] += dh; gw1[j] += dh * dX[n]; }
      }
    }
    for (var j2 = 0; j2 < H; j2++) { net.w1[j2] -= lr * gw1[j2]; net.b1[j2] -= lr * gb1[j2]; net.w2m[j2] -= lr * gm[j2]; net.w2s[j2] -= lr * gs[j2]; }
    net.bm -= lr * gbm; net.bs -= lr * gbs;
    step++; hist.push(tot / N);
    return tot / N;
  }

  function draw() {
    document.getElementById('wb1-w-val').textContent = slW.value;
    document.getElementById('wb1-lr-val').textContent = parseFloat(slLr.value).toFixed(1);
    var mu = [], lo = [], hi = [];
    for (var i = 0; i < xL.length; i++) { var f = fwd(xL[i]), s = Math.exp(f.ls); mu.push(f.mu); lo.push(f.mu - s); hi.push(f.mu + s); }
    var sL = SBI.prepCanvas(document.getElementById('wb1-left'), 285);
    var axL = SBI.createAxes(sL, { title: 'Learned μ_φ(x) ± σ_φ(x)', xRange: [-1, 1], yRange: [-2.1, 2.1], xLabel: 'x', yLabel: 'y' });
    axL.band(xL, lo, hi, 'rgba(37,99,235,0.20)');
    axL.scatter(dX, dY, 'rgba(71,85,105,0.50)', 2.2);
    axL.line(xL, mu, '#2563eb', 2.6, []);

    var sR = SBI.prepCanvas(document.getElementById('wb1-right'), 285);
    var axR = SBI.createAxes(sR, { title: 'Gaussian NLL training curve', xRange: [0, Math.max(60, hist.length)], yRange: [-0.4, 1.5], xLabel: 'Step', yLabel: 'Mean NLL' });
    if (hist.length > 1) axR.line(SBI.linspace(1, hist.length, hist.length), hist, '#2563eb', 2.3, []);
  }

  function loop() {
    if (!run) return;
    var l = 0; for (var k = 0; k < 5; k++) l = stepSGD();
    document.getElementById('wb1-stat').textContent = 'step ' + step + '  |  NLL ' + l.toFixed(3);
    draw(); raf = requestAnimationFrame(loop);
  }

  btnT.addEventListener('click', function() {
    run = !run; btnT.textContent = run ? '⏸ Pause' : '▶ Train';
    if (run) loop();
  });
  document.getElementById('wb1-reset').addEventListener('click', init);
  slW.addEventListener('change', init);
  slW.addEventListener('input', function() { document.getElementById('wb1-w-val').textContent = slW.value; });
  slLr.addEventListener('input', function() { document.getElementById('wb1-lr-val').textContent = parseFloat(slLr.value).toFixed(1); });
  SBI.observeVisibility(root, function() { run = false; btnT.textContent = '▶ Train'; });
  SBI.onResize(root, draw);
  init();
})();
""",
    },

    # =========================================================================
    # Backup B12: Live 2D Affine-Coupling RealNVP Flow Trainer (Kept animation!)
    # =========================================================================
    "wb_flow_2d": {
        "title": "Backup B12: Live 4-Layer 2D Affine-Coupling Flow Trainer",
        "height": 385,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="wb2-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="wb2-curv">Banana curvature b</label>
      <input type="range" id="wb2-curv" min="0.00" max="0.85" step="0.05" value="0.50">
      <span class="iw-val" id="wb2-curv-val">0.50</span>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="wb2-train">&#9654; Train flow</button>
    <button type="button" class="iw-btn" id="wb2-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="wb2-stat">step 0</span>
  </div>
  <div class="iw-row" style="gap:8px;">
    <div class="iw-col"><canvas class="iw-canvas" id="wb2-c0" data-height="235"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wb2-c1" data-height="235"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wb2-c2" data-height="235"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wb2-c3" data-height="235"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wb2-c4" data-height="235"></canvas></div>
  </div>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('wb2-root');
  if (!root) return;
  var slC = document.getElementById('wb2-curv'), btnT = document.getElementById('wb2-train');
  var H = 20, NP = 240, NT = 450, z0 = new Float64Array(NP), z1 = new Float64Array(NP), zCol = new Array(NP);
  var t0 = new Float64Array(NT), t1 = new Float64Array(NT), layers = [], step = 0, run = false, raf = null;
  var trRng = SBI.createRNG(707);

  (function() {
    var r = SBI.createRNG(12);
    for (var i = 0; i < NP; i++) {
      z0[i] = SBI.gaussN(r); z1[i] = SBI.gaussN(r);
      zCol[i] = 'hsl(' + Math.floor(((Math.atan2(z1[i], z0[i]) + Math.PI) / (2 * Math.PI)) * 360) + ',72%,44%)';
    }
  })();

  function refreshTgt() {
    var b = parseFloat(slC.value), r = SBI.createRNG(99);
    for (var i = 0; i < NT; i++) { var x = SBI.gaussN(r); t0[i] = x; t1[i] = b * (x * x - 1) + 0.36 * SBI.gaussN(r); }
  }

  function mkLayer(mask, r) {
    var w1 = new Float64Array(H), b1 = new Float64Array(H), ws = new Float64Array(H), wt = new Float64Array(H);
    for (var i = 0; i < H; i++) { w1[i] = 0.55 * SBI.gaussN(r); b1[i] = 0.55 * SBI.gaussN(r); ws[i] = 0.02 * SBI.gaussN(r); wt[i] = 0.02 * SBI.gaussN(r); }
    return {
      mask: mask, w1: w1, b1: b1, ws: ws, wt: wt, bs: 0, bt: 0,
      mw1: new Float64Array(H), vw1: new Float64Array(H), mb1: new Float64Array(H), vb1: new Float64Array(H),
      mws: new Float64Array(H), vws: new Float64Array(H), mwt: new Float64Array(H), vwt: new Float64Array(H),
      mbs: 0, vbs: 0, mbt: 0, vbt: 0
    };
  }

  function evalL(L, a) {
    var h = new Float64Array(H), pre = new Float64Array(H), r = L.bs, t = L.bt;
    for (var i = 0; i < H; i++) { var z = L.w1[i] * a + L.b1[i]; pre[i] = z; var hi = z > 0 ? z : 0; h[i] = hi; r += L.ws[i] * hi; t += L.wt[i] * hi; }
    return { s: 0.85 * Math.tanh(r), r: r, t: t, h: h, pre: pre };
  }

  function trainStep() {
    var curv = parseFloat(slC.value), lr = 0.003, B = 80, tot = 0;
    var grd = layers.map(function() { return { gw1: new Float64Array(H), gb1: new Float64Array(H), gws: new Float64Array(H), gwt: new Float64Array(H), gbs: 0, gbt: 0 }; });
    for (var n = 0; n < B; n++) {
      var x0 = SBI.gaussN(trRng), x1 = curv * (x0 * x0 - 1) + 0.36 * SBI.gaussN(trRng);
      var u = new Array(5); u[4] = [x0, x1]; var cs = new Array(4), sumS = 0;
      for (var k = 3; k >= 0; k--) {
        var L = layers[k], p = L.mask, c = 1 - p, a = u[k + 1][p], y = u[k + 1][c];
        var m = evalL(L, a), ens = Math.exp(-m.s), uc = (y - m.t) * ens, uk = [0, 0];
        uk[p] = a; uk[c] = uc; u[k] = uk; sumS += m.s; cs[k] = { m: m, a: a, uc: uc, ens: ens };
      }
      tot += 0.5 * (u[0][0] * u[0][0] + u[0][1] * u[0][1]) + 1.838 + sumS;
      var g = [u[0][0] / B, u[0][1] / B];
      for (var k2 = 0; k2 < 4; k2++) {
        var L2 = layers[k2], G = grd[k2], C = cs[k2], p2 = L2.mask, c2 = 1 - p2;
        var ds = (1 / B) - g[c2] * C.uc, dr = ds * 0.85 * (1 - Math.pow(Math.tanh(C.m.r), 2)), dt = -g[c2] * C.ens;
        G.gbs += dr; G.gbt += dt; var da = 0;
        for (var i = 0; i < H; i++) {
          G.gws[i] += dr * C.m.h[i]; G.gwt[i] += dt * C.m.h[i];
          if (C.m.pre[i] > 0) { var dp = dr * L2.ws[i] + dt * L2.wt[i]; G.gb1[i] += dp; G.gw1[i] += dp * C.a; da += dp * L2.w1[i]; }
        }
        var gn = [0, 0]; gn[p2] = g[p2] + da; gn[c2] = g[c2] * C.ens; g = gn;
      }
    }
    step++; var b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step);
    layers.forEach(function(L, idx) {
      var G = grd[idx];
      for (var i = 0; i < H; i++) {
        ['w1','b1','ws','wt'].forEach(function(nm) {
          var gArr = G['g' + nm], mArr = L['m' + nm], vArr = L['v' + nm];
          mArr[i] = b1 * mArr[i] + (1 - b1) * gArr[i];
          vArr[i] = b2 * vArr[i] + (1 - b2) * gArr[i] * gArr[i];
          L[nm][i] -= lr * (mArr[i] / c1) / (Math.sqrt(vArr[i] / c2) + 1e-8);
        });
      }
      L.mbs = b1 * L.mbs + (1 - b1) * G.gbs; L.vbs = b2 * L.vbs + (1 - b2) * G.gbs * G.gbs;
      L.bs -= lr * (L.mbs / c1) / (Math.sqrt(L.vbs / c2) + 1e-8);
      L.mbt = b1 * L.mbt + (1 - b1) * G.gbt; L.vbt = b2 * L.vbt + (1 - b2) * G.gbt * G.gbt;
      L.bt -= lr * (L.mbt / c1) / (Math.sqrt(L.vbt / c2) + 1e-8);
    });
    return tot / B;
  }

  function draw() {
    document.getElementById('wb2-curv-val').textContent = parseFloat(slC.value).toFixed(2);
    var stg = [{ x: Array.from(z0), y: Array.from(z1) }], cx = z0.slice(), cy = z1.slice();
    for (var k = 0; k < 4; k++) {
      var L = layers[k], nx = new Float64Array(NP), ny = new Float64Array(NP);
      for (var i = 0; i < NP; i++) {
        if (L.mask === 0) { var m0 = evalL(L, cx[i]); nx[i] = cx[i]; ny[i] = cy[i] * Math.exp(m0.s) + m0.t; }
        else { var m1 = evalL(L, cy[i]); nx[i] = cx[i] * Math.exp(m1.s) + m1.t; ny[i] = cy[i]; }
      }
      cx = nx; cy = ny; stg.push({ x: Array.from(cx), y: Array.from(cy) });
    }
    var titles = ['Base z', 'After f₁', 'After f₂', 'After f₃', 'f₄(z) vs Target'];
    for (var s = 0; s < 5; s++) {
      var surf = SBI.prepCanvas(document.getElementById('wb2-c' + s), 235);
      var ax = SBI.createAxes(surf, { title: titles[s], xRange: [-2.8, 2.8], yRange: [-2.3, 2.7], pad: { l: 26, r: 8, t: 24, b: 24 }, nXTicks: 3, nYTicks: 3 });
      if (s === 4) ax.scatter(Array.from(t0), Array.from(t1), 'rgba(148,163,184,0.36)', 1.7);
      ax.scatter(stg[s].x, stg[s].y, zCol, 2.1);
    }
  }

  function reset() {
    run = false; btnT.textContent = '▶ Train flow'; if (raf) cancelAnimationFrame(raf);
    var r = SBI.createRNG(515);
    layers = [mkLayer(0, r), mkLayer(1, r), mkLayer(0, r), mkLayer(1, r)];
    step = 0; refreshTgt(); document.getElementById('wb2-stat').textContent = 'step 0'; draw();
  }

  function loop() {
    if (!run) return;
    var l = 0; for (var k = 0; k < 4; k++) l = trainStep();
    document.getElementById('wb2-stat').textContent = 'step ' + step + '  |  NLL ' + l.toFixed(2);
    draw(); raf = requestAnimationFrame(loop);
  }

  btnT.addEventListener('click', function() { run = !run; btnT.textContent = run ? '⏸ Pause' : '▶ Train flow'; if (run) loop(); });
  document.getElementById('wb2-reset').addEventListener('click', reset);
  slC.addEventListener('input', function() { refreshTgt(); draw(); });
  SBI.observeVisibility(root, function() { run = false; btnT.textContent = '▶ Train flow'; });
  SBI.onResize(root, draw);
  reset();
})();
""",
    },
}
