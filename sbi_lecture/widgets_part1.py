"""Interactive SBI Lecture Widgets — Part 1 (W01 through W09).

Implements all foundational & Galton-board ABC / density-estimation slider widgets
from Section 5 of the Master Build Prompt, operating on reproducible seeds and
cached Galton-board / Gamma-GMM benchmarks in `window.SBI_DATA`.
"""

from __future__ import annotations

WIDGETS_PART1: dict[str, dict[str, object]] = {
    # =========================================================================
    # W01: Forward vs Inverse Problem on the Galton Board (Slide 2)
    # =========================================================================
    "w01_forward_inverse": {
        "title": "W01: Forward Prediction vs Bayesian Inverse Problem",
        "height": 430,
        "data_keys": ["observations"],
        "html": """
<div class="iw-widget" id="w01-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w01-mode">Direction</label>
      <select class="iw-select" id="w01-mode">
        <option value="forward" selected>Forward problem: &theta; = (&alpha;, &kappa;) &rarr; x</option>
        <option value="inverse">Inverse problem: observed x_obs &rarr; which (&alpha;, &kappa;)?</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w01-a">Bias &alpha;</label>
      <input type="range" id="w01-a" min="-1.50" max="1.50" step="0.05" value="0.40">
      <span class="iw-val" id="w01-a-val">0.40</span>
    </div>
    <div class="iw-ctrl">
      <label for="w01-k">Centering &kappa;</label>
      <input type="range" id="w01-k" min="0.00" max="3.00" step="0.05" value="0.80">
      <span class="iw-val" id="w01-k-val">0.80</span>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="w01-sim">&#x21bb; Simulate 100 balls</button>
    <button type="button" class="iw-btn" id="w01-reset">&#x21ba; Reset</button>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w01-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w01-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption" id="w01-cap"></p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w01-root');
  if (!root) return;
  var selM = document.getElementById('w01-mode');
  var slA = document.getElementById('w01-a');
  var slK = document.getElementById('w01-k');
  var cap = document.getElementById('w01-cap');
  var seed = 2026;
  var xObs = (window.SBI_DATA && window.SBI_DATA.observations)
    ? window.SBI_DATA.observations[0].x_obs
    : [0,0,1,3,8,15,24,22,15,8,3,1,0];

  function draw() {
    var mode = selM.value;
    var a = parseFloat(slA.value), k = parseFloat(slK.value);
    document.getElementById('w01-a-val').textContent = a.toFixed(2);
    document.getElementById('w01-k-val').textContent = k.toFixed(2);

    var rng = SBI.createRNG(seed);
    var sim = SBI.simPlinko(a, k, 100, 12, rng, 18);
    var bins = SBI.linspace(0, 12, 13);

    var sL = SBI.prepCanvas(document.getElementById('w01-left'), 300);
    if (mode === 'forward') {
      cap.textContent = 'Forward mode: fixing θ = (α, κ) and running the stochastic simulator produces a random 100-ball histogram x ~ p(x | θ).';
      var axL = SBI.createAxes(sL, {
        title: 'Parameter space & 18 sample ball paths',
        xRange: [-13, 13], yRange: [-12.5, 0.8],
        xLabel: 'Horizontal displacement s_t', yLabel: 'Peg row −t'
      });
      for (var p = 0; p < sim.paths.length; p++) {
        axL.line(sim.paths[p].xs, sim.paths[p].ys, p === 0 ? '#dc2626' : 'rgba(37,99,235,0.28)', p === 0 ? 2.4 : 1.2, []);
      }
      var px = [], py = [];
      for (var r = 0; r < 12; r++) for (var c = 0; c <= r; c++) { px.push(2 * c - r); py.push(-r); }
      axL.scatter(px, py, '#475569', 2.6);
    } else {
      cap.textContent = 'Inverse mode: nature gives ONLY the dark Mystery Histogram x_obs (100 balls). Try adjusting (α, κ) to match x_obs — many nearby pairs plausibly explain the noisy counts!';
      var axLI = SBI.createAxes(sL, {
        title: 'Prior box (α, κ) ∈ [−1.5, 1.5] × [0, 3]: which θ caused x_obs?',
        xRange: [-1.5, 1.5], yRange: [0, 3.0],
        xLabel: 'Horizontal bias α', yLabel: 'Restoring strength κ'
      });
      axLI.vline(0, '#cbd5e1', 1.1, [4, 4]);
      axLI.star(a, k, '#2563eb', 8.0);
      var wDist = SBI.histDistance(sim.counts, xObs, 'wasserstein');
      axLI.text(0.05, 0.10, 'Candidate θ = (' + a.toFixed(2) + ', ' + k.toFixed(2) + ')', '#2563eb', 'left');
      axLI.text(0.05, 0.20, 'Wasserstein distance to x_obs: ' + wDist.toFixed(3) + ' bins', '#1f2937', 'left');
    }

    var yMax = Math.max(36, Math.max.apply(null, sim.counts) * 1.18, Math.max.apply(null, xObs) * 1.18);
    var sR = SBI.prepCanvas(document.getElementById('w01-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: mode === 'forward' ? 'Simulated 100-ball outcome x ~ p(x | θ)' : 'Mystery x_obs (dark) vs Candidate simulation (blue)',
      xRange: [-0.7, 12.7], yRange: [0, yMax],
      xLabel: 'Landing bin k ∈ {0, …, 12}', yLabel: 'Ball count (N = 100)'
    });
    if (mode === 'forward') {
      axR.bars(bins, sim.counts, 0.74, 'rgba(37,99,235,0.62)', '#1d4ed8');
      var theo = SBI.plinkoProbs(a, k, 12).map(function(v) { return v * 100; });
      axR.line(bins, theo, '#dc2626', 2.4, [5, 3]);
      axR.legend([
        { type: 'bar', color: 'rgba(37,99,235,0.7)', label: '100-ball realization' },
        { type: 'line', color: '#dc2626', dash: [5, 3], label: 'Expected E[n_k | θ]' }
      ], 'tl');
    } else {
      axR.bars(bins, xObs, 0.38, 'rgba(31,41,55,0.82)', '#111827', -0.20);
      axR.bars(bins, sim.counts, 0.38, 'rgba(37,99,235,0.65)', '#1d4ed8', 0.20);
      axR.legend([
        { type: 'bar', color: 'rgba(31,41,55,0.85)', label: 'Observed x_obs (unknown θ*)' },
        { type: 'bar', color: 'rgba(37,99,235,0.7)', label: 'Simulated at (α, κ)' }
      ], 'tl');
    }
  }

  selM.addEventListener('change', draw);
  slA.addEventListener('input', draw);
  slK.addEventListener('input', draw);
  document.getElementById('w01-sim').addEventListener('click', function() { seed += 13; draw(); });
  document.getElementById('w01-reset').addEventListener('click', function() {
    selM.value = 'forward'; slA.value = '0.40'; slK.value = '0.80'; seed = 2026; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W02: Animated Galton Board / Plinko Simulator (Slides 5 & 7)
    # =========================================================================
    "w02_plinko_sim": {
        "title": "W02: Animated Mystery Galton Board Simulator",
        "height": 440,
        "data_keys": ["observations"],
        "html": """
<div class="iw-widget" id="w02-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w02-a">Bias &alpha;</label>
      <input type="range" id="w02-a" min="-1.50" max="1.50" step="0.05" value="0.60">
      <span class="iw-val" id="w02-a-val">0.60</span>
    </div>
    <div class="iw-ctrl">
      <label for="w02-k">Centering &kappa;</label>
      <input type="range" id="w02-k" min="0.00" max="3.00" step="0.05" value="1.20">
      <span class="iw-val" id="w02-k-val">1.20</span>
    </div>
    <div class="iw-ctrl">
      <label for="w02-r">Rows R</label>
      <input type="range" id="w02-r" min="8" max="14" step="2" value="12">
      <span class="iw-val" id="w02-r-val">12</span>
    </div>
    <div class="iw-ctrl">
      <label for="w02-seed">Seed</label>
      <input type="range" id="w02-seed" min="1" max="50" step="1" value="7">
      <span class="iw-val" id="w02-seed-val">7</span>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="w02-drop1">&#9654; Drop 1 ball</button>
    <button type="button" class="iw-btn" id="w02-drop100">&#9193; +100 balls</button>
    <button type="button" class="iw-btn" id="w02-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w02-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w02-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w02-right" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption">
    Peg colour shows local rightward bounce probability P(right | s<sub>t</sub>, &alpha;, &kappa;) = &sigma;(&alpha; &minus; &kappa; s<sub>t</sub>/&radic;R): warm orange &gt; 0.5, cool blue &lt; 0.5. Press <strong>&#9654; Drop 1 ball</strong> to animate an individual trajectory or adjust (&alpha;, &kappa;) with a fixed seed.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w02-root');
  if (!root) return;
  var slA = document.getElementById('w02-a');
  var slK = document.getElementById('w02-k');
  var slR = document.getElementById('w02-r');
  var slS = document.getElementById('w02-seed');
  var counts = [], paths = [], animPath = null, animRow = 0, timer = null;
  var liveRng = SBI.createRNG(7);

  function rebuildFixedBatch() {
    stopAnim();
    var a = parseFloat(slA.value), k = parseFloat(slK.value);
    var R = parseInt(slR.value, 10), sd = parseInt(slS.value, 10);
    liveRng = SBI.createRNG(sd * 97 + 13);
    var sim = SBI.simPlinko(a, k, 100, R, liveRng, 20);
    counts = sim.counts;
    paths = sim.paths;
    animPath = paths[0];
    animRow = R;
    draw();
  }

  function stopAnim() {
    if (timer) { clearInterval(timer); timer = null; }
  }

  function dropOneAnimated() {
    stopAnim();
    var a = parseFloat(slA.value), k = parseFloat(slK.value), R = parseInt(slR.value, 10);
    var one = SBI.simPlinko(a, k, 1, R, liveRng, 1);
    animPath = one.paths[0];
    animRow = 0;
    timer = setInterval(function() {
      animRow += 1;
      if (animRow >= R) {
        stopAnim();
        counts[animPath.bin] += 1;
        paths.unshift(animPath);
        if (paths.length > 24) paths.pop();
      }
      draw();
    }, 55);
  }

  function dropHundred() {
    stopAnim();
    var a = parseFloat(slA.value), k = parseFloat(slK.value), R = parseInt(slR.value, 10);
    var batch = SBI.simPlinko(a, k, 100, R, liveRng, 12);
    for (var b = 0; b <= R; b++) counts[b] += batch.counts[b];
    paths = batch.paths.concat(paths).slice(0, 24);
    animPath = paths[0];
    animRow = R;
    draw();
  }

  function draw() {
    var a = parseFloat(slA.value), k = parseFloat(slK.value);
    var R = parseInt(slR.value, 10), sd = parseInt(slS.value, 10);
    document.getElementById('w02-a-val').textContent = a.toFixed(2);
    document.getElementById('w02-k-val').textContent = k.toFixed(2);
    document.getElementById('w02-r-val').textContent = String(R);
    document.getElementById('w02-seed-val').textContent = String(sd);

    var st = SBI.histStats(counts);
    document.getElementById('w02-stat').textContent =
      'N = ' + st.total + '  |  k̄ = ' + st.mean.toFixed(2) + '  |  SD = ' + st.std.toFixed(2);

    var sqR = Math.sqrt(R);
    var sL = SBI.prepCanvas(document.getElementById('w02-left'), 305);
    var axL = SBI.createAxes(sL, {
      title: 'Galton board pegs & stochastic trajectories',
      xRange: [-(R + 1), R + 1], yRange: [-(R + 0.6), 0.8],
      xLabel: 'Horizontal state s_t = 2k_t − t', yLabel: 'Row −t'
    });
    for (var p = 0; p < paths.length; p++) {
      axL.line(paths[p].xs, paths[p].ys, 'rgba(37,99,235,0.22)', 1.2, []);
    }
    if (animPath) {
      var len = Math.min(animRow + 1, animPath.xs.length);
      axL.line(animPath.xs.slice(0, len), animPath.ys.slice(0, len), '#dc2626', 2.8, []);
      axL.scatter([animPath.xs[len - 1]], [animPath.ys[len - 1]], '#dc2626', 5.2);
    }
    var px = [], py = [], pc = [];
    for (var r = 0; r < R; r++) {
      for (var c = 0; c <= r; c++) {
        var s = 2 * c - r;
        var pr = SBI.sigmoid(a - k * s / sqR);
        px.push(s); py.push(-r);
        var red = Math.round(37 + 195 * pr), blu = Math.round(215 - 175 * pr);
        pc.push('rgb(' + red + ',105,' + blu + ')');
      }
    }
    axL.scatter(px, py, pc, 3.3);

    var bins = SBI.linspace(0, R, R + 1);
    var empFreq = counts.map(function(c) { return c / Math.max(1, st.total); });
    var yMax = Math.max(0.32, Math.max.apply(null, empFreq) * 1.22);
    var sR = SBI.prepCanvas(document.getElementById('w02-right'), 305);
    var axR = SBI.createAxes(sR, {
      title: 'Empirical landing histogram (normalised)',
      xRange: [-0.7, R + 0.7], yRange: [0, yMax],
      xLabel: 'Landing bin k ∈ {0, …, ' + R + '}', yLabel: 'Empirical fraction n_k / N'
    });
    axR.bars(bins, empFreq, 0.76, 'rgba(37,99,235,0.60)', '#1d4ed8');
  }

  [slA, slK, slR, slS].forEach(function(el) { el.addEventListener('input', rebuildFixedBatch); });
  document.getElementById('w02-drop1').addEventListener('click', dropOneAnimated);
  document.getElementById('w02-drop100').addEventListener('click', dropHundred);
  document.getElementById('w02-reset').addEventListener('click', function() {
    slA.value = '0.60'; slK.value = '1.20'; slR.value = '12'; slS.value = '7';
    rebuildFixedBatch();
  });
  SBI.observeVisibility(root, stopAnim);
  SBI.onResize(root, draw);
  rebuildFixedBatch();
})();
""",
    },

    # =========================================================================
    # W03: Analytic Binomial vs Generalized Galton Board (Slide 6)
    # =========================================================================
    "w03_analytic_galton": {
        "title": "W03: Analytic Binomial (kappa = 0) vs State-Dependent Galton Board",
        "height": 420,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="w03-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w03-mode">Board model</label>
      <select class="iw-select" id="w03-mode">
        <option value="binomial" selected>Constant bounce p (&kappa; = 0, exact Binomial)</option>
        <option value="general">State-dependent (&alpha;, &kappa; &gt; 0, non-Binomial)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w03-p" id="w03-p-lbl">Right bounce p</label>
      <input type="range" id="w03-p" min="0.15" max="0.85" step="0.01" value="0.65">
      <span class="iw-val" id="w03-p-val">0.65</span>
    </div>
    <div class="iw-ctrl">
      <label for="w03-k">Centering &kappa;</label>
      <input type="range" id="w03-k" min="0.00" max="3.00" step="0.05" value="1.40">
      <span class="iw-val" id="w03-k-val">0.00</span>
    </div>
    <div class="iw-ctrl">
      <label for="w03-n">Balls N</label>
      <input type="range" id="w03-n" min="25" max="2000" step="25" value="100">
      <span class="iw-val" id="w03-n-val">100</span>
    </div>
    <button type="button" class="iw-btn" id="w03-reset">&#x21ba; Reset</button>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w03-left" data-height="295"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w03-right" data-height="295"></canvas></div>
  </div>
  <p class="iw-caption">
    When &kappa; = 0, bounces are i.i.d. Bernoulli(p) and landing bins follow Binomial(R, p). Switching &kappa; &gt; 0 couples each step to position s<sub>t</sub>, narrowing the spread below Binomial(R, p) and removing the elementary closed-form formula.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w03-root');
  if (!root) return;
  var selM = document.getElementById('w03-mode');
  var slP = document.getElementById('w03-p');
  var slK = document.getElementById('w03-k');
  var slN = document.getElementById('w03-n');
  var R = 12, bins = SBI.linspace(0, R, R + 1);

  function draw() {
    var isBin = selM.value === 'binomial';
    var p = parseFloat(slP.value);
    var alpha = Math.log(p / (1.0 - p));
    var kappa = isBin ? 0.0 : parseFloat(slK.value);
    var N = parseInt(slN.value, 10);
    slK.disabled = isBin;

    document.getElementById('w03-p-val').textContent = p.toFixed(2) + ' (α=' + alpha.toFixed(2) + ')';
    document.getElementById('w03-k-val').textContent = kappa.toFixed(2);
    document.getElementById('w03-n-val').textContent = String(N);

    var rng = SBI.createRNG(42);
    var sim = SBI.simPlinko(alpha, kappa, N, R, rng, 0);
    var emp = sim.counts.map(function(c) { return c / N; });
    var binomP = SBI.binomProbs(p, R);
    var exactP = SBI.plinkoProbs(alpha, kappa, R);

    /* Left: right-bounce probability vs horizontal position s_t */
    var sVals = SBI.linspace(-12, 12, 97);
    var pCurve = sVals.map(function(s) { return SBI.sigmoid(alpha - kappa * s / Math.sqrt(R)); });
    var sL = SBI.prepCanvas(document.getElementById('w03-left'), 295);
    var axL = SBI.createAxes(sL, {
      title: 'Step transition rule P(right | s_t)',
      xRange: [-12, 12], yRange: [0, 1.02],
      xLabel: 'Horizontal displacement s_t', yLabel: 'P(right | s_t)'
    });
    axL.hline(p, '#94a3b8', 1.6, [5, 4]);
    axL.line(sVals, pCurve, isBin ? '#2563eb' : '#059669', 2.8, []);
    axL.legend([
      { type: 'line', color: isBin ? '#2563eb' : '#059669', label: isBin ? 'Constant p = σ(α)' : 'State-dependent σ(α − κ s_t/√R)' },
      { type: 'line', color: '#94a3b8', dash: [5, 4], label: 'Baseline p at centre s_t = 0' }
    ], 'bl');

    var yMax = Math.max(0.36, Math.max.apply(null, emp) * 1.18, Math.max.apply(null, exactP) * 1.18);
    var sR = SBI.prepCanvas(document.getElementById('w03-right'), 295);
    var axR = SBI.createAxes(sR, {
      title: 'Empirical histogram (N=' + N + ') vs theoretical law',
      xRange: [-0.7, R + 0.7], yRange: [0, yMax],
      xLabel: 'Landing bin k', yLabel: 'Probability'
    });
    axR.bars(bins, emp, 0.74, 'rgba(37,99,235,0.52)', '#1d4ed8');
    axR.line(bins, binomP, '#94a3b8', 2.2, [5, 4]);
    axR.line(bins, exactP, '#dc2626', 2.6, []);
    axR.scatter(bins, exactP, '#dc2626', 3.3);
    axR.legend([
      { type: 'bar', color: 'rgba(37,99,235,0.65)', label: 'Empirical n_k / N' },
      { type: 'line', color: '#dc2626', label: isBin ? 'Exact Binomial(12, p)' : 'State-dependent P(K=k | α, κ)' },
      { type: 'line', color: '#94a3b8', dash: [5, 4], label: 'Reference Binomial (κ=0)' }
    ], 'tl');
  }

  selM.addEventListener('change', draw);
  [slP, slK, slN].forEach(function(el) { el.addEventListener('input', draw); });
  document.getElementById('w03-reset').addEventListener('click', function() {
    selM.value = 'binomial'; slP.value = '0.65'; slK.value = '1.40'; slN.value = '100'; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W02c: Path Explosion & Mystery Challenge Matcher (Slides 8 & 9)
    # =========================================================================
    "w02c_paths_challenge": {
        "title": "Latent Path Marginalisation & Candidate Challenge",
        "height": 420,
        "data_keys": ["observations"],
        "html": """
<div class="iw-widget" id="w02c-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w02c-bin">Inspect target bin k</label>
      <input type="range" id="w02c-bin" min="0" max="12" step="1" value="7">
      <span class="iw-val" id="w02c-bin-val">bin 7 (792 paths)</span>
    </div>
    <div class="iw-ctrl">
      <label for="w02c-cand">Candidate &theta;</label>
      <select class="iw-select" id="w02c-cand">
        <option value="A">Candidate A: (&alpha; = 0.25, &kappa; = 0.25)</option>
        <option value="B" selected>Candidate B: (&alpha; = 0.55, &kappa; = 1.15)</option>
        <option value="C">Candidate C: (&alpha; = 0.95, &kappa; = 2.45)</option>
      </select>
    </div>
    <button type="button" class="iw-btn" id="w02c-res">&#x21bb; New 100-ball realization</button>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w02c-left" data-height="295"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w02c-right" data-height="295"></canvas></div>
  </div>
  <p class="iw-caption">
    Left: every trajectory reaching bin k requires k right bounces and 12 &minus; k left bounces, but different path orderings visit different positions s<sub>t</sub> and carry <em>different</em> probabilities when &kappa; &gt; 0! Right: compare three candidate settings against the Mystery Machine x<sub>obs</sub>.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w02c-root');
  if (!root) return;
  var slBin = document.getElementById('w02c-bin');
  var selC = document.getElementById('w02c-cand');
  var seed = 88, R = 12;
  var xObs = (window.SBI_DATA && window.SBI_DATA.observations)
    ? window.SBI_DATA.observations[0].x_obs
    : [0,0,1,3,8,15,24,22,15,8,3,1,0];

  function nCr(n, k) {
    if (k < 0 || k > n) return 0;
    var res = 1;
    for (var i = 1; i <= Math.min(k, n - k); i++) res = res * (n - i + 1) / i;
    return Math.round(res);
  }

  function candTheta(code) {
    if (code === 'A') return [0.25, 0.25];
    if (code === 'B') return [0.55, 1.15];
    return [0.95, 2.45];
  }

  function sampleBridgePaths(targetK, count, rng) {
    var out = [];
    for (var m = 0; m < count; m++) {
      var rightsLeft = targetK, stepsLeft = R, k = 0, xs = [0], ys = [0];
      for (var t = 0; t < R; t++) {
        if (rng() < rightsLeft / stepsLeft) { k++; rightsLeft--; }
        stepsLeft--;
        xs.push(2 * k - (t + 1)); ys.push(-(t + 1));
      }
      out.push({ xs: xs, ys: ys });
    }
    return out;
  }

  function draw() {
    var kBin = parseInt(slBin.value, 10);
    var nPaths = nCr(R, kBin);
    document.getElementById('w02c-bin-val').textContent = 'bin ' + kBin + ' (' + nPaths + ' paths)';
    var th = candTheta(selC.value);

    var pRng = SBI.createRNG(31 + kBin);
    var bridges = sampleBridgePaths(kBin, Math.min(nPaths, 45), pRng);

    var sL = SBI.prepCanvas(document.getElementById('w02c-left'), 295);
    var axL = SBI.createAxes(sL, {
      title: nPaths + ' latent paths z terminating at bin k = ' + kBin,
      xRange: [-13, 13], yRange: [-12.5, 0.8],
      xLabel: 'Horizontal displacement s_t', yLabel: 'Row −t'
    });
    for (var i = 1; i < bridges.length; i++) axL.line(bridges[i].xs, bridges[i].ys, 'rgba(124,58,237,0.20)', 1.2, []);
    if (bridges.length > 0) axL.line(bridges[0].xs, bridges[0].ys, '#dc2626', 2.8, []);
    var px = [], py = [];
    for (var r = 0; r < R; r++) for (var c = 0; c <= r; c++) { px.push(2 * c - r); py.push(-r); }
    axL.scatter(px, py, '#64748b', 2.4);
    axL.legend([
      { type: 'line', color: '#dc2626', label: '1 realized path (cheap to simulate)' },
      { type: 'line', color: 'rgba(124,58,237,0.65)', label: 'All ' + nPaths + ' compatible paths in Σ_z p(x,z|θ)' }
    ], 'tl');

    var cRng = SBI.createRNG(seed);
    var sim = SBI.simPlinko(th[0], th[1], 100, R, cRng, 0);
    var bins = SBI.linspace(0, R, R + 1);
    var wDist = SBI.histDistance(sim.counts, xObs, 'wasserstein');

    var sR = SBI.prepCanvas(document.getElementById('w02c-right'), 295);
    var axR = SBI.createAxes(sR, {
      title: 'Mystery x_obs vs Candidate ' + selC.value + ' (W₁ = ' + wDist.toFixed(2) + ')',
      xRange: [-0.7, 12.7], yRange: [0, 38],
      xLabel: 'Landing bin k', yLabel: 'Count (100 balls)'
    });
    axR.bars(bins, xObs, 0.38, 'rgba(31,41,55,0.84)', '#111827', -0.20);
    axR.bars(bins, sim.counts, 0.38, 'rgba(217,119,6,0.68)', '#b45309', 0.20);
    axR.legend([
      { type: 'bar', color: 'rgba(31,41,55,0.85)', label: 'Mystery observed x_obs' },
      { type: 'bar', color: 'rgba(217,119,6,0.75)', label: 'Candidate ' + selC.value + ' (100 balls)' }
    ], 'tl');
  }

  slBin.addEventListener('input', draw);
  selC.addEventListener('change', draw);
  document.getElementById('w02c-res').addEventListener('click', function() { seed += 19; draw(); });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W04: Original 1D Chi2(5) + Normal(sqrt(theta), 1) ABC Explorer (S11-13)
    # =========================================================================
    "w04_abc_1d": {
        "title": "W04: Original 1D Chi-Squared / Normal Joint-Slice ABC Explorer",
        "height": 430,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="w04-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w04-eps">Tolerance &varepsilon;</label>
      <input type="range" id="w04-eps" min="0.04" max="2.00" step="0.02" value="0.35">
      <span class="iw-val" id="w04-eps-val">0.35</span>
    </div>
    <div class="iw-ctrl">
      <label for="w04-xobs">Observed x<sub>obs</sub></label>
      <input type="range" id="w04-xobs" min="0.00" max="5.00" step="0.10" value="2.20">
      <span class="iw-val" id="w04-xobs-val">2.20</span>
    </div>
    <div class="iw-ctrl">
      <label for="w04-n">Pool size N</label>
      <input type="range" id="w04-n" min="400" max="6000" step="200" value="2400">
      <span class="iw-val" id="w04-n-val">2400</span>
    </div>
    <button type="button" class="iw-btn" id="w04-seed">&#x21bb; New seed</button>
    <button type="button" class="iw-btn" id="w04-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w04-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w04-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w04-right" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption">
    Original course model: &theta; &sim; &chi;<sup>2</sup>(5), &nbsp; x | &theta; &sim; N(&radic;&theta;, 1). Shrinking &varepsilon; reduces bias toward p(&theta; | x<sub>obs</sub>), but shrinks the accepted count M and inflates Monte Carlo variance.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w04-root');
  if (!root) return;
  var slE = document.getElementById('w04-eps');
  var slX = document.getElementById('w04-xobs');
  var slN = document.getElementById('w04-n');
  var seed = 42, cache = { key: '', th: [], x: [] };
  var thGrid = SBI.linspace(0.05, 18.0, 260);

  /* Chi2(5) = sum_{j=1}^5 Z_j^2 */
  function getPool(N) {
    var k = N + '|' + seed;
    if (cache.key !== k) {
      var rng = SBI.createRNG(seed), th = new Array(N), x = new Array(N);
      for (var i = 0; i < N; i++) {
        var s = 0;
        for (var j = 0; j < 5; j++) { var z = SBI.gaussN(rng); s += z * z; }
        th[i] = s;
        x[i] = Math.sqrt(s) + SBI.gaussN(rng);
      }
      cache = { key: k, th: th, x: x };
    }
    return cache;
  }

  /* Precompute 2D joint density p(theta, x) grid for contour backdrop */
  var tg2 = SBI.linspace(0.1, 16.5, 45), xg2 = SBI.linspace(-1.5, 6.2, 40);
  var jointZ = tg2.map(function(t) {
    var pt = SBI.chi2Pdf5(t), rt = Math.sqrt(t);
    return xg2.map(function(xv) { return pt * SBI.normPdf(xv, rt, 1.0); });
  });

  function draw() {
    var eps = parseFloat(slE.value);
    var xObs = parseFloat(slX.value);
    var N = parseInt(slN.value, 10);
    document.getElementById('w04-eps-val').textContent = eps.toFixed(2);
    document.getElementById('w04-xobs-val').textContent = xObs.toFixed(2);
    document.getElementById('w04-n-val').textContent = String(N);

    var pool = getPool(N);
    var accT = [], accX = [], rejT = [], rejX = [];
    var stride = N > 2500 ? 2 : 1;
    for (var i = 0; i < N; i++) {
      if (Math.abs(pool.x[i] - xObs) < eps) {
        accT.push(pool.th[i]); accX.push(pool.x[i]);
      } else if (i % stride === 0) {
        rejT.push(pool.th[i]); rejX.push(pool.x[i]);
      }
    }
    document.getElementById('w04-stat').textContent =
      'Accepted ' + accT.length + ' / ' + N + ' (' + (100 * accT.length / N).toFixed(1) + '%)';

    var sL = SBI.prepCanvas(document.getElementById('w04-left'), 305);
    var axL = SBI.createAxes(sL, {
      title: 'Joint distribution p(θ, x) & ε-acceptance band',
      xRange: [0, 16.5], yRange: [-1.5, 6.2],
      xLabel: 'Parameter θ ~ χ²(5)', yLabel: 'Observation x ~ N(√θ, 1)'
    });
    axL.heatmapAK(tg2, xg2, jointZ, '37,99,235', 0.28);
    axL.band([0, 16.5], [xObs - eps, xObs - eps], [xObs + eps, xObs + eps], 'rgba(217,119,6,0.18)');
    axL.scatter(rejT, rejX, 'rgba(148,163,184,0.42)', 1.9);
    axL.scatter(accT, accX, '#d97706', 2.8);
    axL.hline(xObs, '#111827', 1.7, [5, 4]);
    axL.legend([
      { type: 'dot', color: '#94a3b8', label: 'Rejected sims' },
      { type: 'dot', color: '#d97706', label: 'Accepted |x − x_obs| < ε' }
    ], 'tr');

    var exactRaw = thGrid.map(function(t) { return SBI.chi2Pdf5(t) * SBI.normPdf(xObs, Math.sqrt(t), 1.0); });
    var exactP = SBI.trapzNorm(thGrid, exactRaw);
    var priorP = thGrid.map(SBI.chi2Pdf5);
    var h = SBI.histogramDensity(accT, 0, 18.0, 28);
    var yMax = Math.max(0.32, Math.max.apply(null, exactP) * 1.25, Math.max.apply(null, h.ys) * 1.15);

    var sR = SBI.prepCanvas(document.getElementById('w04-right'), 305);
    var axR = SBI.createAxes(sR, {
      title: 'Accepted θ slice vs exact posterior p(θ | x_obs)',
      xRange: [0, 16.5], yRange: [0, yMax],
      xLabel: 'Parameter θ', yLabel: 'Density'
    });
    if (accT.length > 0) axR.bars(h.xs, h.ys, h.binW, 'rgba(217,119,6,0.52)', '#b45309');
    axR.line(thGrid, priorP, '#94a3b8', 1.8, [4, 4]);
    axR.line(thGrid, exactP, '#111827', 2.6, []);
    axR.legend([
      { type: 'bar', color: 'rgba(217,119,6,0.65)', label: 'ABC accepted θ' },
      { type: 'line', color: '#111827', label: 'Exact posterior' },
      { type: 'line', color: '#94a3b8', dash: [4, 4], label: 'Prior χ²(5)' }
    ], 'tr');
  }

  [slE, slX, slN].forEach(function(el) { el.addEventListener('input', draw); });
  document.getElementById('w04-seed').addEventListener('click', function() { seed += 17; cache.key = ''; draw(); });
  document.getElementById('w04-reset').addEventListener('click', function() {
    slE.value = '0.35'; slX.value = '2.20'; slN.value = '2400'; seed = 42; cache.key = ''; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W05: Correlated-Gaussian ABC Explorer with Dual Tolerance (Slide 14)
    # =========================================================================
    "w05_corr_gauss": {
        "title": "W05: Correlated Bivariate Gaussian ABC Tolerance Explorer",
        "height": 430,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="w05-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w05-rho">Correlation &rho;</label>
      <input type="range" id="w05-rho" min="-0.95" max="0.95" step="0.05" value="0.80">
      <span class="iw-val" id="w05-rho-val">0.80</span>
    </div>
    <div class="iw-ctrl">
      <label for="w05-e1">Tight &varepsilon;<sub>1</sub></label>
      <input type="range" id="w05-e1" min="0.05" max="1.50" step="0.05" value="0.25">
      <span class="iw-val" id="w05-e1-val">0.25</span>
    </div>
    <div class="iw-ctrl">
      <label for="w05-e2">Wide &varepsilon;<sub>2</sub></label>
      <input type="range" id="w05-e2" min="0.20" max="2.50" step="0.05" value="1.25">
      <span class="iw-val" id="w05-e2-val">1.25</span>
    </div>
    <div class="iw-ctrl">
      <label for="w05-x">x<sub>obs</sub></label>
      <input type="range" id="w05-x" min="-2.00" max="2.00" step="0.10" value="1.20">
      <span class="iw-val" id="w05-x-val">1.20</span>
    </div>
    <button type="button" class="iw-btn" id="w05-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w05-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w05-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w05-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Exact conditional: E[&theta; | x] = &rho; x, &nbsp; Var(&theta; | x) = 1 &minus; &rho;<sup>2</sup>. Comparing tight &varepsilon;<sub>1</sub> (blue) and wide &varepsilon;<sub>2</sub> (orange) shows both variance inflation (+&rho;<sup>2</sup>&varepsilon;<sup>2</sup>/3) and mean shrinkage toward the prior mean 0.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w05-root');
  if (!root) return;
  var slR = document.getElementById('w05-rho');
  var slE1 = document.getElementById('w05-e1');
  var slE2 = document.getElementById('w05-e2');
  var slX = document.getElementById('w05-x');
  var tg = SBI.linspace(-3.4, 3.4, 240);

  function abcCurve(rho, eps, xObs) {
    var rt = Math.sqrt(Math.max(1e-5, 1 - rho * rho));
    var raw = tg.map(function(t) {
      var hi = (xObs + eps - rho * t) / rt;
      var lo = (xObs - eps - rho * t) / rt;
      return SBI.normPdf(t, 0, 1) * Math.max(0, SBI.gaussCdf(hi) - SBI.gaussCdf(lo));
    });
    return SBI.trapzNorm(tg, raw);
  }

  function draw() {
    var rho = parseFloat(slR.value), e1 = parseFloat(slE1.value), e2 = parseFloat(slE2.value), xObs = parseFloat(slX.value);
    document.getElementById('w05-rho-val').textContent = rho.toFixed(2);
    document.getElementById('w05-e1-val').textContent = e1.toFixed(2);
    document.getElementById('w05-e2-val').textContent = e2.toFixed(2);
    document.getElementById('w05-x-val').textContent = xObs.toFixed(2);

    var condMean = rho * xObs, condVar = 1.0 - rho * rho;
    document.getElementById('w05-stat').textContent =
      'Exact E[θ|x] = ' + condMean.toFixed(2) + ',  Var(θ|x) = ' + condVar.toFixed(2);

    var rng = SBI.createRNG(505);
    var rt = Math.sqrt(Math.max(1e-5, condVar));
    var thAll = [], xAll = [], th1 = [], x1 = [];
    for (var i = 0; i < 2000; i++) {
      var xv = SBI.gaussN(rng);
      var tv = rho * xv + rt * SBI.gaussN(rng);
      thAll.push(tv); xAll.push(xv);
      if (Math.abs(xv - xObs) <= e1) { th1.push(tv); x1.push(xv); }
    }

    var sL = SBI.prepCanvas(document.getElementById('w05-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: 'Correlated Gaussian p(θ, x) (ρ = ' + rho.toFixed(2) + ')',
      xRange: [-3.4, 3.4], yRange: [-3.4, 3.4],
      xLabel: 'θ', yLabel: 'x'
    });
    axL.band([-3.4, 3.4], [xObs - e2, xObs - e2], [xObs + e2, xObs + e2], 'rgba(217,119,6,0.12)');
    axL.band([-3.4, 3.4], [xObs - e1, xObs - e1], [xObs + e1, xObs + e1], 'rgba(37,99,235,0.20)');
    axL.scatter(thAll, xAll, 'rgba(148,163,184,0.38)', 1.8);
    axL.scatter(th1, x1, '#2563eb', 2.4);
    axL.hline(xObs, '#111827', 1.6, [5, 4]);

    var pEx = tg.map(function(t) { return SBI.normPdf(t, condMean, rt); });
    var p1 = abcCurve(rho, e1, xObs);
    var p2 = abcCurve(rho, e2, xObs);
    var yMax = Math.max.apply(null, pEx) * 1.18 + 0.08;

    var sR = SBI.prepCanvas(document.getElementById('w05-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: 'Exact conditional vs tight ε₁ & wide ε₂ ABC',
      xRange: [-3.4, 3.4], yRange: [0, yMax],
      xLabel: 'θ', yLabel: 'Posterior density'
    });
    var h1 = SBI.histogramDensity(th1, -3.4, 3.4, 30);
    if (th1.length > 5) axR.bars(h1.xs, h1.ys, h1.binW, 'rgba(37,99,235,0.30)', '#2563eb');
    axR.line(tg, pEx, '#111827', 2.6, []);
    axR.line(tg, p1, '#2563eb', 2.4, [5, 3]);
    axR.line(tg, p2, '#d97706', 2.5, []);
    axR.legend([
      { type: 'line', color: '#111827', label: 'Exact N(ρx, 1−ρ²)' },
      { type: 'line', color: '#2563eb', dash: [5, 3], label: 'ABC tight ε₁ = ' + e1.toFixed(2) },
      { type: 'line', color: '#d97706', label: 'ABC wide ε₂ = ' + e2.toFixed(2) }
    ], 'tl');
  }

  [slR, slE1, slE2, slX].forEach(function(el) { el.addEventListener('input', draw); });
  document.getElementById('w05-reset').addEventListener('click', function() {
    slR.value = '0.80'; slE1.value = '0.25'; slE2.value = '1.25'; slX.value = '1.20'; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W06: Mystery Galton Board Rejection ABC Explorer (Slide 15)
    # =========================================================================
    "w06_plinko_abc": {
        "title": "W06: Rejection ABC on the Mystery Galton Board",
        "height": 440,
        "data_keys": ["grid", "observations", "abc_bank"],
        "html": """
<div class="iw-widget" id="w06-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w06-dist">Distance d(x, x<sub>obs</sub>)</label>
      <select class="iw-select" id="w06-dist">
        <option value="wasserstein" selected>1D Wasserstein (EMD, bins)</option>
        <option value="tv">Total Variation</option>
        <option value="l2">Binwise L2</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w06-eps">Tolerance &varepsilon;</label>
      <input type="range" id="w06-eps" min="0.06" max="1.40" step="0.02" value="0.36">
      <span class="iw-val" id="w06-eps-val">0.36</span>
    </div>
    <div class="iw-ctrl">
      <label for="w06-n">Sim budget</label>
      <input type="range" id="w06-n" min="300" max="3000" step="100" value="3000">
      <span class="iw-val" id="w06-n-val">3000</span>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w06-ref">
      <label for="w06-ref">Show exact reference</label>
    </div>
    <button type="button" class="iw-btn" id="w06-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w06-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w06-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w06-right" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption">
    Cached prior-predictive Galton-board simulation bank: dragging &varepsilon; filters (&alpha;<sub>i</sub>, &kappa;<sub>i</sub>) instantaneously without re-running the simulator.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w06-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA;
  var obs = D.observations[0];
  var bank = D.abc_bank, gr = D.grid;

  var selD = document.getElementById('w06-dist');
  var slE = document.getElementById('w06-eps');
  var slN = document.getElementById('w06-n');
  var chkRef = document.getElementById('w06-ref');

  function syncEpsRange() {
    var kind = selD.value;
    if (kind === 'wasserstein') { slE.min = '0.10'; slE.max = '1.50'; slE.step = '0.02'; slE.value = '0.36'; }
    else if (kind === 'tv') { slE.min = '0.05'; slE.max = '0.45'; slE.step = '0.01'; slE.value = '0.16'; }
    else { slE.min = '0.03'; slE.max = '0.30'; slE.step = '0.01'; slE.value = '0.09'; }
  }

  function draw() {
    var kind = selD.value;
    var eps = parseFloat(slE.value);
    var N = Math.min(bank.alphas.length, parseInt(slN.value, 10));
    document.getElementById('w06-eps-val').textContent = eps.toFixed(2);
    document.getElementById('w06-n-val').textContent = String(N);

    var accA = [], accK = [], rejA = [], rejK = [];
    var bestIdx = 0, bestD = 1e9;
    for (var i = 0; i < N; i++) {
      var d = SBI.histDistance(bank.hists[i], obs.x_obs, kind);
      if (d < bestD) { bestD = d; bestIdx = i; }
      if (d < eps) { accA.push(bank.alphas[i]); accK.push(bank.kappas[i]); }
      else { rejA.push(bank.alphas[i]); rejK.push(bank.kappas[i]); }
    }
    document.getElementById('w06-stat').textContent =
      'Accepted ' + accA.length + ' / ' + N + ' (' + (100 * accA.length / N).toFixed(1) + '%)';

    var bins = SBI.linspace(0, 12, 13);
    var sL = SBI.prepCanvas(document.getElementById('w06-left'), 305);
    var axL = SBI.createAxes(sL, {
      title: 'Observed x_obs vs closest accepted sim (d=' + bestD.toFixed(3) + ')',
      xRange: [-0.7, 12.7], yRange: [0, 36],
      xLabel: 'Landing bin k', yLabel: 'Count (100 balls)'
    });
    axL.bars(bins, obs.x_obs, 0.38, 'rgba(31,41,55,0.85)', '#111827', -0.20);
    axL.bars(bins, bank.hists[bestIdx], 0.38, 'rgba(217,119,6,0.70)', '#b45309', 0.20);
    axL.legend([
      { type: 'bar', color: 'rgba(31,41,55,0.85)', label: 'Mystery x_obs' },
      { type: 'bar', color: 'rgba(217,119,6,0.75)', label: 'Closest sim (' + bank.alphas[bestIdx].toFixed(2) + ', ' + bank.kappas[bestIdx].toFixed(2) + ')' }
    ], 'tl');

    var sR = SBI.prepCanvas(document.getElementById('w06-right'), 305);
    var axR = SBI.createAxes(sR, {
      title: 'ABC posterior in (α, κ) parameter space',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Horizontal bias α', yLabel: 'Restoring strength κ'
    });
    axR.scatter(rejA, rejK, 'rgba(156,163,175,0.34)', 1.8);
    if (accA.length >= 8) {
      var kde = SBI.kde2DAK(accA, accK, gr.alpha, gr.kappa, gr.da, gr.dk);
      axR.heatmapAK(gr.alpha, gr.kappa, kde, '217,119,6', 0.42);
      axR.contourAK(gr.alpha, gr.kappa, kde, SBI.hpdLevel(kde, gr.da, gr.dk, 0.68), '#d97706', 2.2, []);
      axR.contourAK(gr.alpha, gr.kappa, kde, SBI.hpdLevel(kde, gr.da, gr.dk, 0.95), '#d97706', 1.5, [5, 4]);
    }
    axR.scatter(accA, accK, '#d97706', 2.7);
    if (chkRef.checked) {
      var ex = obs.post.Exact;
      axR.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.68), '#111827', 2.2, []);
      axR.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.95), '#111827', 1.5, [4, 4]);
    }
    var leg = [
      { type: 'dot', color: '#d97706', label: 'ABC accepted (68%/95% HPD)' },
      { type: 'dot', color: '#9ca3af', label: 'Rejected prior draws' }
    ];
    if (chkRef.checked) leg.push({ type: 'line', color: '#111827', label: 'Exact DP posterior' });
    axR.legend(leg, 'tl');
  }

  selD.addEventListener('change', function() { syncEpsRange(); draw(); });
  [slE, slN].forEach(function(el) { el.addEventListener('input', draw); });
  chkRef.addEventListener('change', draw);
  document.getElementById('w06-reset').addEventListener('click', function() {
    selD.value = 'wasserstein'; syncEpsRange(); slN.value = '3000'; chkRef.checked = false; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W07: Galton Board Summary-Statistic & Distance Explorer (Slide 16)
    # =========================================================================
    "w07_plinko_summaries": {
        "title": "W07: Summary-Statistic Sufficiency vs Information Loss on Galton Board",
        "height": 440,
        "data_keys": ["grid", "observations", "abc_bank"],
        "html": """
<div class="iw-widget" id="w07-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w07-sum">Summary S(x)</label>
      <select class="iw-select" id="w07-sum">
        <option value="mean">1. Mean bin only: S(x) = k&#x0304; (lossy!)</option>
        <option value="mean_var" selected>2. Mean &amp; variance: S(x) = (k&#x0304;, s&sup2;)</option>
        <option value="full">3. Full 13-bin histogram (Wasserstein)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w07-m">Keep closest M</label>
      <input type="range" id="w07-m" min="25" max="600" step="25" value="150">
      <span class="iw-val" id="w07-m-val">150 (5.0%)</span>
    </div>
    <div class="iw-ctrl">
      <input type="checkbox" id="w07-ref" checked>
      <label for="w07-ref">Overlay exact posterior</label>
    </div>
    <button type="button" class="iw-btn" id="w07-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w07-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w07-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w07-right" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption">
    Switch to <strong>Mean bin only</strong> and shrink M: even as &varepsilon;<sub>eff</sub> &rarr; 0, &kappa; remains unidentifiable along a diagonal ridge because k&#x0304; discarded spread information! Including variance s<sup>2</sup> immediately breaks the degeneracy.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w07-root');
  if (!root || !window.SBI_DATA) return;
  var D = window.SBI_DATA, obs = D.observations[0], bank = D.abc_bank, gr = D.grid;
  var obsStats = SBI.histStats(obs.x_obs);

  var selS = document.getElementById('w07-sum');
  var slM = document.getElementById('w07-m');
  var chkRef = document.getElementById('w07-ref');

  function draw() {
    var sk = selS.value;
    var M = parseInt(slM.value, 10);
    var N = bank.alphas.length;
    document.getElementById('w07-m-val').textContent = M + ' (' + (100 * M / N).toFixed(1) + '%)';

    var dists = new Float64Array(N);
    for (var i = 0; i < N; i++) {
      if (sk === 'mean') {
        dists[i] = Math.abs(bank.means[i] - obsStats.mean);
      } else if (sk === 'mean_var') {
        var dm = (bank.means[i] - obsStats.mean) / 1.5;
        var dv = (bank.vars[i] - obsStats.variance) / 1.2;
        dists[i] = Math.sqrt(dm * dm + dv * dv);
      } else {
        dists[i] = SBI.histDistance(bank.hists[i], obs.x_obs, 'wasserstein');
      }
    }
    var idx = new Array(N);
    for (var q = 0; q < N; q++) idx[q] = q;
    idx.sort(function(a, b) { return dists[a] - dists[b]; });
    var keep = idx.slice(0, M);
    var accA = keep.map(function(ii) { return bank.alphas[ii]; });
    var accK = keep.map(function(ii) { return bank.kappas[ii]; });
    var epsEff = dists[keep[M - 1]];
    document.getElementById('w07-stat').textContent = 'ε_eff = ' + epsEff.toFixed(3);

    var kde = SBI.kde2DAK(accA, accK, gr.alpha, gr.kappa, gr.da, gr.dk);
    var mAbc = SBI.marginalsAK(kde, gr.da, gr.dk);
    var ex = obs.post.Exact;
    var mEx = SBI.marginalsAK(ex, gr.da, gr.dk);

    var sL = SBI.prepCanvas(document.getElementById('w07-left'), 305);
    var axL = SBI.createAxes(sL, {
      title: '2D ABC posterior p_ε(α, κ | S(x_obs))',
      xRange: [gr.alpha_min, gr.alpha_max], yRange: [gr.kappa_min, gr.kappa_max],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axL.heatmapAK(gr.alpha, gr.kappa, kde, '217,119,6', 0.45);
    axL.scatter(accA, accK, 'rgba(217,119,6,0.75)', 2.4);
    axL.contourAK(gr.alpha, gr.kappa, kde, SBI.hpdLevel(kde, gr.da, gr.dk, 0.68), '#d97706', 2.4, []);
    if (chkRef.checked) {
      axL.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.68), '#111827', 2.2, []);
      axL.contourAK(gr.alpha, gr.kappa, ex, SBI.hpdLevel(ex, gr.da, gr.dk, 0.95), '#111827', 1.5, [4, 4]);
    }
    axL.legend([
      { type: 'line', color: '#d97706', label: 'ABC 68% HPD (' + sk + ')' },
      { type: 'line', color: '#111827', label: 'Exact full-data posterior' }
    ], 'tl');

    /* Right: 1D marginal for kappa (where sufficiency vs lossy mean is dramatic!) */
    var yMax = Math.max(Math.max.apply(null, mEx.kappa), Math.max.apply(null, mAbc.kappa)) * 1.22 + 0.1;
    var sR = SBI.prepCanvas(document.getElementById('w07-right'), 305);
    var axR = SBI.createAxes(sR, {
      title: 'Marginal posterior p(κ | S(x_obs))',
      xRange: [gr.kappa_min, gr.kappa_max], yRange: [0, yMax],
      xLabel: 'Restoring strength κ', yLabel: 'Marginal density'
    });
    axR.fillUnder(gr.kappa, mAbc.kappa, 'rgba(217,119,6,0.22)', 0);
    axR.line(gr.kappa, mAbc.kappa, '#d97706', 2.8, []);
    if (chkRef.checked) axR.line(gr.kappa, mEx.kappa, '#111827', 2.4, [5, 4]);
    axR.hline(1.0 / (gr.kappa_max - gr.kappa_min), '#94a3b8', 1.4, [3, 3]);
    axR.legend([
      { type: 'line', color: '#d97706', label: 'ABC marginal p(κ | S)' },
      { type: 'line', color: '#111827', dash: [5, 4], label: 'Exact marginal p(κ | x_obs)' },
      { type: 'line', color: '#94a3b8', dash: [3, 3], label: 'Uniform prior' }
    ], 'tr');
  }

  selS.addEventListener('change', draw);
  slM.addEventListener('input', draw);
  chkRef.addEventListener('change', draw);
  document.getElementById('w07-reset').addEventListener('click', function() {
    selS.value = 'mean_var'; slM.value = '150'; chkRef.checked = true; draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W08: Interactive 1D Gaussian Mixture Editor (Slide 21)
    # =========================================================================
    "w08_gmm_editor": {
        "title": "W08: Interactive 1D Gaussian Mixture Model Editor",
        "height": 430,
        "data_keys": [],
        "html": """
<div class="iw-widget" id="w08-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w08-w1">w<sub>1</sub>, &mu;<sub>1</sub>, &sigma;<sub>1</sub></label>
      <input type="range" id="w08-w1" min="0.05" max="1.00" step="0.05" value="0.45" style="width:72px;">
      <input type="range" id="w08-m1" min="-3.0" max="3.0" step="0.1" value="-1.4" style="width:72px;">
      <input type="range" id="w08-s1" min="0.20" max="1.60" step="0.05" value="0.55" style="width:72px;">
    </div>
    <div class="iw-ctrl">
      <label for="w08-w2">w<sub>2</sub>, &mu;<sub>2</sub>, &sigma;<sub>2</sub></label>
      <input type="range" id="w08-w2" min="0.05" max="1.00" step="0.05" value="0.35" style="width:72px;">
      <input type="range" id="w08-m2" min="-3.0" max="3.0" step="0.1" value="0.6" style="width:72px;">
      <input type="range" id="w08-s2" min="0.20" max="1.60" step="0.05" value="0.45" style="width:72px;">
    </div>
    <div class="iw-ctrl">
      <label for="w08-w3">w<sub>3</sub>, &mu;<sub>3</sub>, &sigma;<sub>3</sub></label>
      <input type="range" id="w08-w3" min="0.05" max="1.00" step="0.05" value="0.20" style="width:72px;">
      <input type="range" id="w08-m3" min="-3.0" max="3.0" step="0.1" value="2.1" style="width:72px;">
      <input type="range" id="w08-s3" min="0.20" max="1.60" step="0.05" value="0.70" style="width:72px;">
    </div>
    <button type="button" class="iw-btn" id="w08-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w08-stat"></span>
  </div>
  <canvas class="iw-canvas" id="w08-canvas" data-height="295"></canvas>
  <p class="iw-caption">
    q<sub>&phi;</sub>(x) = &sum;<sub>c=1</sub><sup>3</sup> &pi;<sub>c</sub> N(x; &mu;<sub>c</sub>, &sigma;<sub>c</sub><sup>2</sup>) with normalised weights &pi;<sub>c</sub> = w<sub>c</sub> / &sum;<sub>j</sub> w<sub>j</sub> and positive scales &sigma;<sub>c</sub> &gt; 0: simple Gaussian building blocks combine into skewed or multimodal densities that always integrate to 1.
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w08-root');
  if (!root) return;
  var ids = ['w08-w1','w08-m1','w08-s1','w08-w2','w08-m2','w08-s2','w08-w3','w08-m3','w08-s3'];
  var els = {};
  ids.forEach(function(id) { els[id] = document.getElementById(id); });
  var xg = SBI.linspace(-4.8, 5.2, 360);

  function draw() {
    var rw = [parseFloat(els['w08-w1'].value), parseFloat(els['w08-w2'].value), parseFloat(els['w08-w3'].value)];
    var sw = rw[0] + rw[1] + rw[2];
    var pi = [rw[0] / sw, rw[1] / sw, rw[2] / sw];
    var mu = [parseFloat(els['w08-m1'].value), parseFloat(els['w08-m2'].value), parseFloat(els['w08-m3'].value)];
    var sg = [parseFloat(els['w08-s1'].value), parseFloat(els['w08-s2'].value), parseFloat(els['w08-s3'].value)];

    document.getElementById('w08-stat').textContent =
      'π = (' + pi[0].toFixed(2) + ', ' + pi[1].toFixed(2) + ', ' + pi[2].toFixed(2) + ')  |  ∫ q_φ(x) dx = 1.00';

    var c1 = xg.map(function(x) { return pi[0] * SBI.normPdf(x, mu[0], sg[0]); });
    var c2 = xg.map(function(x) { return pi[1] * SBI.normPdf(x, mu[1], sg[1]); });
    var c3 = xg.map(function(x) { return pi[2] * SBI.normPdf(x, mu[2], sg[2]); });
    var mix = xg.map(function(_, i) { return c1[i] + c2[i] + c3[i]; });
    var yMax = Math.max(0.45, Math.max.apply(null, mix) * 1.18);

    var surf = SBI.prepCanvas(document.getElementById('w08-canvas'), 295);
    var ax = SBI.createAxes(surf, {
      title: 'Gaussian Mixture Model q_φ(x) & weighted components π_c N(x; μ_c, σ_c²)',
      xRange: [-4.8, 5.2], yRange: [0, yMax],
      xLabel: 'x', yLabel: 'Density q_φ(x)'
    });
    ax.fillUnder(xg, mix, 'rgba(37,99,235,0.14)', 0);
    ax.line(xg, c1, '#d97706', 2.0, [5, 4]);
    ax.line(xg, c2, '#059669', 2.0, [5, 4]);
    ax.line(xg, c3, '#7c3aed', 2.0, [5, 4]);
    ax.line(xg, mix, '#2563eb', 3.0, []);
    ax.legend([
      { type: 'line', color: '#2563eb', label: 'Mixture sum q_φ(x)' },
      { type: 'line', color: '#d97706', dash: [5, 4], label: 'Comp 1 (π₁=' + pi[0].toFixed(2) + ', μ₁=' + mu[0].toFixed(1) + ')' },
      { type: 'line', color: '#059669', dash: [5, 4], label: 'Comp 2 (π₂=' + pi[1].toFixed(2) + ', μ₂=' + mu[1].toFixed(1) + ')' },
      { type: 'line', color: '#7c3aed', dash: [5, 4], label: 'Comp 3 (π₃=' + pi[2].toFixed(2) + ', μ₃=' + mu[2].toFixed(1) + ')' }
    ], 'tr');
  }

  ids.forEach(function(id) { els[id].addEventListener('input', draw); });
  document.getElementById('w08-reset').addEventListener('click', function() {
    els['w08-w1'].value = '0.45'; els['w08-m1'].value = '-1.4'; els['w08-s1'].value = '0.55';
    els['w08-w2'].value = '0.35'; els['w08-m2'].value = '0.6';  els['w08-s2'].value = '0.45';
    els['w08-w3'].value = '0.20'; els['w08-m3'].value = '2.1';  els['w08-s3'].value = '0.70';
    draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # W09: Fitting a GMM to Gamma(3, 1/3) via Empirical NLL (Slide 22)
    # =========================================================================
    "w09_gmm_gamma": {
        "title": "W09: Learning Gamma(3, 1/3) from Samples via Maximum Likelihood",
        "height": 435,
        "data_keys": ["gmm_gamma"],
        "html": """
<div class="iw-widget" id="w09-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="w09-step">Checkpoint</label>
      <input type="range" id="w09-step" min="0" max="8" step="1" value="8">
      <span class="iw-val" id="w09-step-val">iter 160</span>
    </div>
    <div class="iw-ctrl">
      <label for="w09-n">Sample size N</label>
      <select class="iw-select" id="w09-n">
        <option value="150">N = 150</option>
        <option value="1000" selected>N = 1000</option>
        <option value="4000">N = 4000</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="w09-k">Components K</label>
      <select class="iw-select" id="w09-k">
        <option value="1">K = 1 (single Gaussian)</option>
        <option value="3">K = 3</option>
        <option value="5" selected>K = 5 (course default)</option>
        <option value="8">K = 8</option>
      </select>
    </div>
    <button type="button" class="iw-btn" id="w09-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="w09-stat"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="w09-left" data-height="300"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="w09-right" data-height="300"></canvas></div>
  </div>
  <p class="iw-caption">
    Minimising empirical NLL &minus;(1/N)&sum; log q<sub>&phi;</sub>(x<sub>i</sub>) on Gamma(shape=3, scale=1/3) draws drives D<sub>KL</sub>(p &Vert; q<sub>&phi;</sub>) &rarr; 0 without ever evaluating the Gamma PDF during training!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('w09-root');
  if (!root || !window.SBI_DATA) return;
  var G = window.SBI_DATA.gmm_gamma;
  var steps = G.steps;
  var slSt = document.getElementById('w09-step');
  var selN = document.getElementById('w09-n');
  var selK = document.getElementById('w09-k');
  slSt.max = String(steps.length - 1);
  slSt.value = String(steps.length - 1);
  var xg = SBI.linspace(0.0, 4.2, 260);
  var trueY = xg.map(SBI.gamma3Pdf);

  function draw() {
    var sIdx = parseInt(slSt.value, 10);
    var st = steps[sIdx];
    var key = 'N' + selN.value + '_K' + selK.value;
    var run = G.runs[key];
    var ck = run.ckpts[String(st)];
    document.getElementById('w09-step-val').textContent = 'iter ' + st;
    document.getElementById('w09-stat').textContent =
      'NLL = ' + ck.nll.toFixed(3) + '  |  D_KL(p ∥ q_φ) = ' + ck.kl.toFixed(3);

    var qY = xg.map(function(x) {
      var s = 0;
      for (var c = 0; c < ck.w.length; c++) s += ck.w[c] * SBI.normPdf(x, ck.m[c], ck.s[c]);
      return s;
    });
    var binW = run.hist_x.length > 1 ? (run.hist_x[1] - run.hist_x[0]) : 0.13;

    var sL = SBI.prepCanvas(document.getElementById('w09-left'), 300);
    var axL = SBI.createAxes(sL, {
      title: 'Target Gamma(3, 1/3) vs fitted GMM (K=' + selK.value + ')',
      xRange: [-0.2, 4.2], yRange: [0, 1.15],
      xLabel: 'x', yLabel: 'Density'
    });
    axL.bars(run.hist_x, run.hist_y, binW * 0.92, 'rgba(148,163,184,0.34)', '#94a3b8');
    for (var c = 0; c < ck.w.length; c++) {
      var cy = xg.map(function(x) { return ck.w[c] * SBI.normPdf(x, ck.m[c], ck.s[c]); });
      axL.line(xg, cy, 'rgba(37,99,235,0.32)', 1.2, [4, 3]);
    }
    axL.line(xg, trueY, '#111827', 2.4, [5, 4]);
    axL.line(xg, qY, '#2563eb', 2.8, []);
    axL.legend([
      { type: 'bar', color: 'rgba(148,163,184,0.55)', label: 'Training samples (N=' + selN.value + ')' },
      { type: 'line', color: '#111827', dash: [5, 4], label: 'True Gamma(3, 1/3)' },
      { type: 'line', color: '#2563eb', label: 'GMM q_φ(x) at iter ' + st }
    ], 'tr');

    var iters = SBI.linspace(0, run.nll_hist.length - 1, run.nll_hist.length);
    var nMin = Math.min(run.entropy, Math.min.apply(null, run.nll_hist)) - 0.05;
    var nMax = Math.max.apply(null, run.nll_hist) + 0.06;
    var sR = SBI.prepCanvas(document.getElementById('w09-right'), 300);
    var axR = SBI.createAxes(sR, {
      title: 'Empirical NLL vs training iteration',
      xRange: [0, run.nll_hist.length - 1], yRange: [nMin, nMax],
      xLabel: 'Optimization iteration', yLabel: 'Empirical NLL'
    });
    axR.hline(run.entropy, '#059669', 1.7, [5, 4]);
    axR.line(iters, run.nll_hist, '#2563eb', 2.5, []);
    axR.vline(st, '#dc2626', 1.8, [3, 3]);
    axR.scatter([st], [ck.nll], '#dc2626', 4.5);
    axR.legend([
      { type: 'line', color: '#2563eb', label: '−(1/N) Σ log q_φ(x_i)' },
      { type: 'line', color: '#059669', dash: [5, 4], label: 'Target entropy H(p)' }
    ], 'tr');
  }

  slSt.addEventListener('input', draw);
  selN.addEventListener('change', draw);
  selK.addEventListener('change', draw);
  document.getElementById('w09-reset').addEventListener('click', function() {
    selN.value = '1000'; selK.value = '5'; slSt.value = String(steps.length - 1); draw();
  });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },
}
