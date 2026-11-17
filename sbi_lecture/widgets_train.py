"""Interactive Galton-Board Neural Training Widgets (NPE, NLE, NRE).

Provides live epoch-by-epoch training visualizations (`▶ Train` / `⏸ Pause` / `↺ Reset` +
movable Epoch & Hyperparameter controls) showing how NPE, NLE, and NRE learn to solve the
Mystery Galton Board from prior-predictive simulations.
"""

from __future__ import annotations

WIDGETS_TRAIN: dict[str, dict[str, object]] = {
    # =========================================================================
    # WT_NPE: Live Galton-Board Neural Posterior Estimation (NPE) Trainer
    # =========================================================================
    "wt_npe_plinko": {
        "title": "Train NPE on the Galton Board: Conditional Posterior q_φ(α, κ | x_obs)",
        "height": 450,
        "data_keys": ["train_traj"],
        "html": """
<div class="iw-widget" id="wtnpe-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="wtnpe-run">Training setup</label>
      <select class="iw-select" id="wtnpe-run">
        <option value="N36000_K8" selected>M = 36,000 sims, K = 8 MDN (converges)</option>
        <option value="N4000_K8">M = 4,000 sims, K = 8 MDN (moderate budget)</option>
        <option value="N800_K8">M = 800 sims, K = 8 MDN (overfits!)</option>
        <option value="N36000_K1">M = 36,000 sims, K = 1 single Gaussian</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="wtnpe-ep">Epoch</label>
      <input type="range" id="wtnpe-ep" min="0" max="80" step="1" value="0">
      <span class="iw-val" id="wtnpe-ep-val">0</span>
      <progress class="iw-progress" id="wtnpe-prog" max="80" value="0" aria-label="NPE training progress"></progress>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="wtnpe-play">&#9654; Train</button>
    <button type="button" class="iw-btn" id="wtnpe-reset">&#x21ba; Reset</button>
    <div class="iw-ctrl">
      <input type="checkbox" id="wtnpe-ex" checked>
      <label for="wtnpe-ex">Exact DP</label>
    </div>
    <span class="iw-status" id="wtnpe-stat" role="status" aria-live="polite"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="wtnpe-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wtnpe-right" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption">
    <strong>NPE Training Objective:</strong> Minimize &minus;E<sub>p(&theta;,x)</sub>[log q<sub>&phi;</sub>(&theta; | x)] over simulated pairs. Press <strong>&#9654; Train</strong> to watch q<sub>&phi;</sub>(&alpha;, &kappa; | x<sub>obs</sub>) contract from the flat prior onto the exact Galton-board posterior&mdash;and switch to M = 800 to observe validation NLL overfitting!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('wtnpe-root');
  if (!root || !window.SBI_DATA || !window.SBI_DATA.train_traj) return;
  var T = window.SBI_DATA.train_traj;
  var selRun = document.getElementById('wtnpe-run');
  var slEp = document.getElementById('wtnpe-ep');
  var prog = document.getElementById('wtnpe-prog');
  var btnPlay = document.getElementById('wtnpe-play');
  var chkEx = document.getElementById('wtnpe-ex');
  var snaps = T.snap_epochs;
  var anim = false, raf = null, epFloat = 0.0;

  function interpGrid(snapsObj, ep) {
    var e = Math.max(0, Math.min(T.max_epoch, ep));
    var k0 = snaps[0], k1 = snaps[snaps.length - 1];
    for (var i = 0; i < snaps.length - 1; i++) {
      if (e >= snaps[i] && e <= snaps[i + 1]) { k0 = snaps[i]; k1 = snaps[i + 1]; break; }
    }
    var g0 = snapsObj[String(k0)], g1 = snapsObj[String(k1)];
    if (k0 === k1) return g0;
    var w = (e - k0) / (k1 - k0);
    var na = g0.length, nk = g0[0].length, out = new Array(na), sum = 0.0;
    for (var r = 0; r < na; r++) {
      out[r] = new Float64Array(nk);
      for (var c = 0; c < nk; c++) {
        var v = (1 - w) * g0[r][c] + w * g1[r][c];
        out[r][c] = v; sum += v;
      }
    }
    var norm = Math.max(1e-12, sum * T.da * T.dk);
    for (var r2 = 0; r2 < na; r2++) for (var c2 = 0; c2 < nk; c2++) out[r2][c2] /= norm;
    return out;
  }

  function samplePost(postAK, count, seed) {
    var na = T.alpha.length, nk = T.kappa.length, tot = 0;
    var cdf = new Float64Array(na * nk);
    for (var i = 0; i < na; i++) for (var j = 0; j < nk; j++) { tot += postAK[i][j]; cdf[i * nk + j] = tot; }
    var rng = SBI.createRNG(seed), sa = [], sk = [];
    for (var s = 0; s < count; s++) {
      var u = rng() * tot, lo = 0, hi = cdf.length - 1;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      var ia = Math.floor(lo / nk), ik = lo % nk;
      sa.push(T.alpha[ia] + (rng() - 0.5) * T.da);
      sk.push(T.kappa[ik] + (rng() - 0.5) * T.dk);
    }
    return { a: sa, k: sk };
  }

  function draw() {
    var run = T.npe[selRun.value];
    var epInt = Math.round(epFloat);
    slEp.value = String(epInt);
    slEp.setAttribute('aria-valuetext', 'Epoch ' + epInt + ' of ' + T.max_epoch);
    prog.value = epInt;
    document.getElementById('wtnpe-ep-val').textContent = String(epInt);

    var post = interpGrid(run.snaps, epFloat);
    var tv = run.tv[epInt], trL = run.train[epInt], vaL = run.val[epInt];
    document.getElementById('wtnpe-stat').textContent =
      'Val NLL: ' + vaL.toFixed(2) + '  |  TV to Exact: ' + tv.toFixed(3);

    var sL = SBI.prepCanvas(document.getElementById('wtnpe-left'), 305);
    var axL = SBI.createAxes(sL, {
      title: 'Learned NPE q_φ(α, κ | x_obs) at Epoch ' + epInt,
      xRange: [-1.5, 1.5], yRange: [0.0, 3.0],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axL.heatmapAK(T.alpha, T.kappa, post, '37,99,235', 0.48);
    var pts = samplePost(post, 180, 301 + epInt);
    axL.scatter(pts.a, pts.k, 'rgba(37,99,235,0.48)', 2.0);
    axL.contourAK(T.alpha, T.kappa, post, SBI.hpdLevel(post, T.da, T.dk, 0.68), '#2563eb', 2.5, []);
    axL.contourAK(T.alpha, T.kappa, post, SBI.hpdLevel(post, T.da, T.dk, 0.95), '#2563eb', 1.5, [5, 4]);
    if (chkEx.checked) {
      axL.contourAK(T.alpha, T.kappa, T.exact, SBI.hpdLevel(T.exact, T.da, T.dk, 0.68), '#111827', 2.0, [4, 4]);
    }
    axL.star(T.alpha_true, T.kappa_true, '#dc2626', 7.0);
    axL.legend([
      { type: 'line', color: '#2563eb', label: 'NPE 68%/95% HPD' },
      { type: 'line', color: '#111827', dash: [4, 4], label: 'Exact DP 68% HPD' },
      { type: 'star', color: '#dc2626', label: 'True θ*' }
    ], 'tl');

    var allVals = run.train.concat(run.val);
    var yMin = Math.min.apply(null, allVals) - 0.18;
    var yMax = Math.min(3.2, Math.max.apply(null, allVals) + 0.15);
    var xsAll = SBI.linspace(0, T.max_epoch, T.max_epoch + 1);
    var xsCur = xsAll.slice(0, epInt + 1);

    var sR = SBI.prepCanvas(document.getElementById('wtnpe-right'), 305);
    var axR = SBI.createAxes(sR, {
      title: 'Forward-KL Negative Log-Likelihood (−E[log q_φ(θ | x)])',
      xRange: [0, T.max_epoch], yRange: [yMin, yMax],
      xLabel: 'Training epoch', yLabel: 'Negative log-likelihood'
    });
    axR.line(xsAll, run.train, 'rgba(37,99,235,0.20)', 1.5, []);
    axR.line(xsAll, run.val, 'rgba(220,38,38,0.20)', 1.5, [4, 3]);
    axR.line(xsCur, run.train.slice(0, epInt + 1), '#2563eb', 2.7, []);
    axR.line(xsCur, run.val.slice(0, epInt + 1), '#dc2626', 2.4, [5, 4]);
    axR.vline(epInt, '#475569', 1.3, [3, 3]);
    axR.scatter([epInt, epInt], [trL, vaL], '#111827', 3.4);
    axR.legend([
      { type: 'line', color: '#2563eb', label: 'Train NLL (' + trL.toFixed(2) + ')' },
      { type: 'line', color: '#dc2626', dash: [5, 4], label: 'Held-out Val NLL (' + vaL.toFixed(2) + ')' }
    ], 'tr');
  }

  function tick() {
    if (!anim) return;
    epFloat += 0.35;
    if (epFloat >= T.max_epoch) {
      epFloat = T.max_epoch; anim = false; btnPlay.textContent = '▶ Train';
      draw(); return;
    }
    draw();
    raf = requestAnimationFrame(tick);
  }

  btnPlay.addEventListener('click', function() {
    if (epFloat >= T.max_epoch) epFloat = 0.0;
    anim = !anim;
    btnPlay.textContent = anim ? '⏸ Pause' : '▶ Train';
    if (anim) tick();
  });
  document.getElementById('wtnpe-reset').addEventListener('click', function() {
    anim = false; btnPlay.textContent = '▶ Train'; if (raf) cancelAnimationFrame(raf);
    epFloat = 0.0; draw();
  });
  slEp.addEventListener('input', function() {
    anim = false; btnPlay.textContent = '▶ Train';
    epFloat = parseFloat(slEp.value); draw();
  });
  selRun.addEventListener('change', function() {
    anim = false; btnPlay.textContent = '▶ Train';
    epFloat = 0.0; draw();
  });
  chkEx.addEventListener('change', draw);
  SBI.observeVisibility(root, function() { anim = false; btnPlay.textContent = '▶ Train'; });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # WT_NLE: Live Galton-Board Neural Likelihood Estimation (NLE) Trainer
    # =========================================================================
    "wt_nle_plinko": {
        "title": "Train NLE on the Galton Board: Surrogate Likelihood q_ψ(x | α, κ) & Prior Swap",
        "height": 455,
        "data_keys": ["train_traj"],
        "html": """
<div class="iw-widget" id="wtnle-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="wtnle-run">Simulations</label>
      <select class="iw-select" id="wtnle-run">
        <option value="N36000" selected>M = 36,000 sims (converges)</option>
        <option value="N4000">M = 4,000 sims (moderate)</option>
        <option value="N600">M = 600 sims (noisy surrogate)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="wtnle-prior">Prior p&prime;(&theta;)</label>
      <select class="iw-select" id="wtnle-prior">
        <option value="flat" selected>Uniform [&minus;1.5,1.5]&times;[0,3]</option>
        <option value="gauss_a">Calibrated &alpha; &sim; N(0.10, 0.25&sup2;)</option>
        <option value="low_k">Weak centering &kappa; &sim; Exp(1.2)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="wtnle-ep">Epoch</label>
      <input type="range" id="wtnle-ep" min="0" max="80" step="1" value="0" style="width:105px;">
      <span class="iw-val" id="wtnle-ep-val">0</span>
      <progress class="iw-progress" id="wtnle-prog" max="80" value="0" style=" --progress-bar-fill:#059669;" aria-label="NLE training progress"></progress>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="wtnle-play" style="background:#059669;border-color:#059669;">&#9654; Train</button>
    <button type="button" class="iw-btn" id="wtnle-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="wtnle-stat" role="status" aria-live="polite"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="wtnle-bins" data-height="295"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wtnle-post" data-height="295"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wtnle-loss" data-height="295"></canvas></div>
  </div>
  <p class="iw-caption">
    <strong>NLE Training Objective:</strong> Minimize &minus;E<sub>p(&theta;,x)</sub>[log q<sub>&psi;</sub>(x | &theta;)]. Watch the 13-bin softmax distribution &pi;&#x0302;<sub>&psi;</sub>(k | &theta;<sup>&starf;</sup>) snap from uniform 1/13 bars onto the true Galton histogram, and change <strong>Prior p&prime;(&theta;)</strong> at any epoch without retraining!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('wtnle-root');
  if (!root || !window.SBI_DATA || !window.SBI_DATA.train_traj) return;
  var T = window.SBI_DATA.train_traj;
  var selRun = document.getElementById('wtnle-run');
  var selPr = document.getElementById('wtnle-prior');
  var slEp = document.getElementById('wtnle-ep');
  var prog = document.getElementById('wtnle-prog');
  var btnPlay = document.getElementById('wtnle-play');
  var snaps = T.snap_epochs, bins = SBI.linspace(0, 12, 13);
  var dpTrueBins = SBI.plinkoProbs(T.alpha_true, T.kappa_true, 12);
  var anim = false, raf = null, epFloat = 0.0;

  function bracket(ep) {
    var e = Math.max(0, Math.min(T.max_epoch, ep));
    var k0 = snaps[0], k1 = snaps[snaps.length - 1];
    for (var i = 0; i < snaps.length - 1; i++) {
      if (e >= snaps[i] && e <= snaps[i + 1]) { k0 = snaps[i]; k1 = snaps[i + 1]; break; }
    }
    var w = k0 === k1 ? 0.0 : (e - k0) / (k1 - k0);
    return { k0: String(k0), k1: String(k1), w: w };
  }

  function draw() {
    var run = T.nle[selRun.value], pKind = selPr.value;
    var epInt = Math.round(epFloat);
    slEp.value = String(epInt);
    slEp.setAttribute('aria-valuetext', 'Epoch ' + epInt + ' of ' + T.max_epoch);
    prog.value = epInt;
    document.getElementById('wtnle-ep-val').textContent = String(epInt);

    var br = bracket(epFloat);
    var b0 = run.snaps_bins[br.k0], b1 = run.snaps_bins[br.k1];
    var curBins = b0.map(function(v, idx) { return (1 - br.w) * v + br.w * b1[idx]; });

    var ll0 = run.snaps_ll[br.k0], ll1 = run.snaps_ll[br.k1];
    var na = T.alpha.length, nk = T.kappa.length, post = new Array(na), sum = 0.0;
    for (var i = 0; i < na; i++) {
      post[i] = new Float64Array(nk);
      var av = T.alpha[i];
      for (var j = 0; j < nk; j++) {
        var kv = T.kappa[j];
        var ll = (1 - br.w) * ll0[i][j] + br.w * ll1[i][j];
        var lp = 0.0;
        if (pKind === 'gauss_a') lp = -0.5 * Math.pow((av - 0.10) / 0.25, 2);
        else if (pKind === 'low_k') lp = -1.2 * kv;
        var v = Math.exp(Math.max(-50, ll + lp));
        post[i][j] = v; sum += v;
      }
    }
    var norm = Math.max(1e-12, sum * T.da * T.dk);
    for (var i2 = 0; i2 < na; i2++) for (var j2 = 0; j2 < nk; j2++) post[i2][j2] /= norm;

    var trL = run.train[epInt], vaL = run.val[epInt], tv = run.tv[epInt];
    document.getElementById('wtnle-stat').textContent =
      'Val NLL: ' + vaL.toFixed(2) + '  |  Flat-prior TV: ' + tv.toFixed(3);

    /* Panel 1: 13-bin probabilities at theta* */
    var sB = SBI.prepCanvas(document.getElementById('wtnle-bins'), 295);
    var axB = SBI.createAxes(sB, {
      title: 'π̂_ψ(k | θ*) at Epoch ' + epInt,
      xRange: [-0.7, 12.7], yRange: [0, 0.30],
      xLabel: 'Bin k', yLabel: 'Probability'
    });
    axB.bars(bins, curBins, 0.72, 'rgba(5,150,105,0.58)', '#047857');
    axB.line(bins, dpTrueBins, '#111827', 2.2, [5, 4]);
    axB.legend([
      { type: 'bar', color: 'rgba(5,150,105,0.65)', label: 'NLE π̂_ψ(k | θ*)' },
      { type: 'line', color: '#111827', dash: [5, 4], label: 'True simulator' }
    ], 'tl');

    /* Panel 2: Induced 2D posterior under active prior */
    var sP = SBI.prepCanvas(document.getElementById('wtnle-post'), 295);
    var axP = SBI.createAxes(sP, {
      title: 'q_ψ(x_obs | θ) p′(θ) (Epoch ' + epInt + ')',
      xRange: [-1.5, 1.5], yRange: [0.0, 3.0],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axP.heatmapAK(T.alpha, T.kappa, post, '5,150,105', 0.48);
    axP.contourAK(T.alpha, T.kappa, post, SBI.hpdLevel(post, T.da, T.dk, 0.68), '#059669', 2.5, []);
    axP.contourAK(T.alpha, T.kappa, post, SBI.hpdLevel(post, T.da, T.dk, 0.95), '#059669', 1.5, [5, 4]);
    if (pKind === 'flat') {
      axP.contourAK(T.alpha, T.kappa, T.exact, SBI.hpdLevel(T.exact, T.da, T.dk, 0.68), '#111827', 2.0, [4, 4]);
    }
    axP.star(T.alpha_true, T.kappa_true, '#dc2626', 6.8);

    /* Panel 3: Multinomial NLL curves */
    var allV = run.train.concat(run.val);
    var yMin = Math.min.apply(null, allV) - 0.8;
    var yMax = Math.min(55, Math.max.apply(null, allV) + 1.0);
    var xsAll = SBI.linspace(0, T.max_epoch, T.max_epoch + 1);
    var xsCur = xsAll.slice(0, epInt + 1);
    var sL = SBI.prepCanvas(document.getElementById('wtnle-loss'), 295);
    var axL = SBI.createAxes(sL, {
      title: 'Multinomial NLL (−E[log q_ψ(x | θ)])',
      xRange: [0, T.max_epoch], yRange: [yMin, yMax],
      xLabel: 'Epoch', yLabel: 'NLL'
    });
    axL.line(xsAll, run.train, 'rgba(5,150,105,0.20)', 1.5, []);
    axL.line(xsAll, run.val, 'rgba(220,38,38,0.20)', 1.5, [4, 3]);
    axL.line(xsCur, run.train.slice(0, epInt + 1), '#059669', 2.6, []);
    axL.line(xsCur, run.val.slice(0, epInt + 1), '#dc2626', 2.3, [5, 4]);
    axL.vline(epInt, '#475569', 1.3, [3, 3]);
    axL.legend([
      { type: 'line', color: '#059669', label: 'Train (' + trL.toFixed(1) + ')' },
      { type: 'line', color: '#dc2626', dash: [5, 4], label: 'Val (' + vaL.toFixed(1) + ')' }
    ], 'tr');
  }

  function tick() {
    if (!anim) return;
    epFloat += 0.35;
    if (epFloat >= T.max_epoch) {
      epFloat = T.max_epoch; anim = false; btnPlay.textContent = '▶ Train';
      draw(); return;
    }
    draw();
    raf = requestAnimationFrame(tick);
  }

  btnPlay.addEventListener('click', function() {
    if (epFloat >= T.max_epoch) epFloat = 0.0;
    anim = !anim;
    btnPlay.textContent = anim ? '⏸ Pause' : '▶ Train';
    if (anim) tick();
  });
  document.getElementById('wtnle-reset').addEventListener('click', function() {
    anim = false; btnPlay.textContent = '▶ Train'; if (raf) cancelAnimationFrame(raf);
    epFloat = 0.0; draw();
  });
  slEp.addEventListener('input', function() {
    anim = false; btnPlay.textContent = '▶ Train';
    epFloat = parseFloat(slEp.value); draw();
  });
  selRun.addEventListener('change', function() {
    anim = false; btnPlay.textContent = '▶ Train';
    epFloat = 0.0; draw();
  });
  selPr.addEventListener('change', draw);
  SBI.observeVisibility(root, function() { anim = false; btnPlay.textContent = '▶ Train'; });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },

    # =========================================================================
    # WT_NRE: Live Galton-Board Neural Ratio Estimation (NRE) Classifier Trainer
    # =========================================================================
    "wt_nre_plinko": {
        "title": "Train NRE on the Galton Board: Joint-vs-Marginal Classifier D_ω(θ, x_obs)",
        "height": 455,
        "data_keys": ["train_traj"],
        "html": """
<div class="iw-widget" id="wtnre-root">
  <div class="iw-toolbar">
    <div class="iw-ctrl">
      <label for="wtnre-run">Classifier regime</label>
      <select class="iw-select" id="wtnre-run">
        <option value="N36000_K24" selected>M = 36,000, K_neg = 24 multi-negative (fast &amp; crisp)</option>
        <option value="N36000_K1">M = 36,000, K_neg = 1 single-negative (noisy / wider)</option>
        <option value="N1200_K16">M = 1,200, K_neg = 16 (small-data overfit!)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="wtnre-surf">Display</label>
      <select class="iw-select" id="wtnre-surf">
        <option value="post" selected>Ratio posterior p(&theta;) r<sub>&omega;</sub>(&theta;, x<sub>obs</sub>)</option>
        <option value="prob">Classifier score D<sub>&omega;</sub>(&theta;, x<sub>obs</sub>) &isin; (0,1)</option>
      </select>
    </div>
    <div class="iw-ctrl">
      <label for="wtnre-ep">Epoch</label>
      <input type="range" id="wtnre-ep" min="0" max="80" step="1" value="0" style="width:105px;">
      <span class="iw-val" id="wtnre-ep-val">0</span>
      <progress class="iw-progress" id="wtnre-prog" max="80" value="0" style="--progress-bar-fill:#7c3aed;" aria-label="NRE training progress"></progress>
    </div>
    <button type="button" class="iw-btn iw-btn-primary" id="wtnre-play" style="background:#7c3aed;border-color:#7c3aed;">&#9654; Train</button>
    <button type="button" class="iw-btn" id="wtnre-reset">&#x21ba; Reset</button>
    <span class="iw-status" id="wtnre-stat" role="status" aria-live="polite"></span>
  </div>
  <div class="iw-row">
    <div class="iw-col"><canvas class="iw-canvas" id="wtnre-left" data-height="305"></canvas></div>
    <div class="iw-col"><canvas class="iw-canvas" id="wtnre-right" data-height="305"></canvas></div>
  </div>
  <p class="iw-caption">
    <strong>NRE Training Objective:</strong> Binary cross-entropy classifying joint pairs (&theta;, x) &sim; p(&theta;,x) vs marginal pairs (&theta;, x&prime;) &sim; p(&theta;)p(x). Compare <strong>K<sub>neg</sub> = 24</strong> vs <strong>K<sub>neg</sub> = 1</strong> to see why contrasting multiple shuffled negatives per step dramatically sharpens ratio convergence!
  </p>
</div>
""",
        "js": r"""
(function() {
  var root = document.getElementById('wtnre-root');
  if (!root || !window.SBI_DATA || !window.SBI_DATA.train_traj) return;
  var T = window.SBI_DATA.train_traj;
  var selRun = document.getElementById('wtnre-run');
  var selSurf = document.getElementById('wtnre-surf');
  var slEp = document.getElementById('wtnre-ep');
  var prog = document.getElementById('wtnre-prog');
  var btnPlay = document.getElementById('wtnre-play');
  var snaps = T.snap_epochs;
  var anim = false, raf = null, epFloat = 0.0;

  function bracket(ep) {
    var e = Math.max(0, Math.min(T.max_epoch, ep));
    var k0 = snaps[0], k1 = snaps[snaps.length - 1];
    for (var i = 0; i < snaps.length - 1; i++) {
      if (e >= snaps[i] && e <= snaps[i + 1]) { k0 = snaps[i]; k1 = snaps[i + 1]; break; }
    }
    var w = k0 === k1 ? 0.0 : (e - k0) / (k1 - k0);
    return { k0: String(k0), k1: String(k1), w: w };
  }

  function draw() {
    var run = T.nre[selRun.value], sMode = selSurf.value;
    var epInt = Math.round(epFloat);
    slEp.value = String(epInt);
    slEp.setAttribute('aria-valuetext', 'Epoch ' + epInt + ' of ' + T.max_epoch);
    prog.value = epInt;
    document.getElementById('wtnre-ep-val').textContent = String(epInt);

    var br = bracket(epFloat);
    var lg0 = run.snaps_logit[br.k0], lg1 = run.snaps_logit[br.k1];
    var na = T.alpha.length, nk = T.kappa.length;
    var post = new Array(na), probGrid = new Array(na), maxLg = -1e9;
    var lgCur = new Array(na);
    for (var i = 0; i < na; i++) {
      lgCur[i] = new Float64Array(nk);
      probGrid[i] = new Float64Array(nk);
      for (var j = 0; j < nk; j++) {
        var lv = (1 - br.w) * lg0[i][j] + br.w * lg1[i][j];
        lgCur[i][j] = lv;
        probGrid[i][j] = SBI.sigmoid(lv);
        if (lv > maxLg) maxLg = lv;
      }
    }
    var sum = 0.0;
    for (var i2 = 0; i2 < na; i2++) {
      post[i2] = new Float64Array(nk);
      for (var j2 = 0; j2 < nk; j2++) {
        var u = Math.exp(lgCur[i2][j2] - maxLg);
        post[i2][j2] = u; sum += u;
      }
    }
    var norm = Math.max(1e-12, sum * T.da * T.dk);
    for (var i3 = 0; i3 < na; i3++) for (var j3 = 0; j3 < nk; j3++) post[i3][j3] /= norm;

    var trL = run.train[epInt], vaL = run.val[epInt], tv = run.tv[epInt], zInt = run.norm[epInt];
    document.getElementById('wtnre-stat').textContent =
      'Val BCE: ' + vaL.toFixed(3) + '  |  ∫p(θ)r_ω dθ = ' + zInt.toFixed(2) + '  |  TV: ' + tv.toFixed(3);

    var sL = SBI.prepCanvas(document.getElementById('wtnre-left'), 305);
    var axL = SBI.createAxes(sL, {
      title: (sMode === 'prob' ? 'Classifier D_ω(α, κ, x_obs)' : 'NRE Posterior p(θ)r_ω(θ, x_obs)') + ' — Epoch ' + epInt,
      xRange: [-1.5, 1.5], yRange: [0.0, 3.0],
      xLabel: 'Bias α', yLabel: 'Restoring κ'
    });
    axL.heatmapAK(T.alpha, T.kappa, sMode === 'prob' ? probGrid : post, '124,58,237', 0.50);
    axL.contourAK(T.alpha, T.kappa, post, SBI.hpdLevel(post, T.da, T.dk, 0.68), '#7c3aed', 2.5, []);
    axL.contourAK(T.alpha, T.kappa, post, SBI.hpdLevel(post, T.da, T.dk, 0.95), '#7c3aed', 1.5, [5, 4]);
    axL.contourAK(T.alpha, T.kappa, T.exact, SBI.hpdLevel(T.exact, T.da, T.dk, 0.68), '#111827', 2.0, [4, 4]);
    axL.star(T.alpha_true, T.kappa_true, '#dc2626', 7.0);
    axL.legend([
      { type: 'line', color: '#7c3aed', label: 'NRE 68%/95% HPD' },
      { type: 'line', color: '#111827', dash: [4, 4], label: 'Exact DP 68% HPD' },
      { type: 'star', color: '#dc2626', label: 'True θ*' }
    ], 'tl');

    var allV = run.train.concat(run.val);
    var yMin = Math.max(0.05, Math.min.apply(null, allV) - 0.04);
    var yMax = Math.max(0.73, Math.max.apply(null, allV) + 0.03);
    var xsAll = SBI.linspace(0, T.max_epoch, T.max_epoch + 1);
    var xsCur = xsAll.slice(0, epInt + 1);

    var sR = SBI.prepCanvas(document.getElementById('wtnre-right'), 305);
    var axR = SBI.createAxes(sR, {
      title: 'Binary Cross-Entropy Loss (Starts at ln 2 ≈ 0.693)',
      xRange: [0, T.max_epoch], yRange: [yMin, yMax],
      xLabel: 'Training epoch', yLabel: 'Balanced BCE loss'
    });
    axR.hline(0.6931, '#94a3b8', 1.4, [3, 3]);
    axR.line(xsAll, run.train, 'rgba(124,58,237,0.20)', 1.5, []);
    axR.line(xsAll, run.val, 'rgba(220,38,38,0.20)', 1.5, [4, 3]);
    axR.line(xsCur, run.train.slice(0, epInt + 1), '#7c3aed', 2.7, []);
    axR.line(xsCur, run.val.slice(0, epInt + 1), '#dc2626', 2.4, [5, 4]);
    axR.vline(epInt, '#475569', 1.3, [3, 3]);
    axR.scatter([epInt, epInt], [trL, vaL], '#111827', 3.4);
    axR.legend([
      { type: 'line', color: '#7c3aed', label: 'Train BCE (' + trL.toFixed(3) + ')' },
      { type: 'line', color: '#dc2626', dash: [5, 4], label: 'Held-out Val BCE (' + vaL.toFixed(3) + ')' },
      { type: 'line', color: '#94a3b8', dash: [3, 3], label: 'Untrained D=0.5 (ln 2)' }
    ], 'tr');
  }

  function tick() {
    if (!anim) return;
    epFloat += 0.35;
    if (epFloat >= T.max_epoch) {
      epFloat = T.max_epoch; anim = false; btnPlay.textContent = '▶ Train';
      draw(); return;
    }
    draw();
    raf = requestAnimationFrame(tick);
  }

  btnPlay.addEventListener('click', function() {
    if (epFloat >= T.max_epoch) epFloat = 0.0;
    anim = !anim;
    btnPlay.textContent = anim ? '⏸ Pause' : '▶ Train';
    if (anim) tick();
  });
  document.getElementById('wtnre-reset').addEventListener('click', function() {
    anim = false; btnPlay.textContent = '▶ Train'; if (raf) cancelAnimationFrame(raf);
    epFloat = 0.0; draw();
  });
  slEp.addEventListener('input', function() {
    anim = false; btnPlay.textContent = '▶ Train';
    epFloat = parseFloat(slEp.value); draw();
  });
  selRun.addEventListener('change', function() {
    anim = false; btnPlay.textContent = '▶ Train';
    epFloat = 0.0; draw();
  });
  selSurf.addEventListener('change', draw);
  SBI.observeVisibility(root, function() { anim = false; btnPlay.textContent = '▶ Train'; });
  SBI.onResize(root, draw);
  draw();
})();
""",
    },
}

