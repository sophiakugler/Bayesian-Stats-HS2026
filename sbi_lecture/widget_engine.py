"""Client-side interactive widget engine (Retina HTML5 Canvas + ipywidgets-style controls).

Powers all live movable slider widgets identically in:
  1. `slides/week10_simulation_based_inference.slides.html` (direct DOM, continuous scroll)
  2. `LECTURES/week10_simulation_based_inference.ipynb` (self-contained `<iframe srcdoc>` cells,
     working at 60 FPS offline in VS Code, Antigravity, JupyterLab, and HTML exports without
     static PNG fallbacks shadowing the interactive sliders).
"""

from __future__ import annotations

import html as _html
import json
import ipywidgets as widgets
from IPython.display import display

WIDGET_CSS = r"""
.iw-widget {
  color-scheme: light;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #ffffff;
  border: 1px solid #dce4ec;
  border-radius: 10px;
  padding: 12px 14px;
  margin: 10px 0;
  box-sizing: border-box;
  color: #1f2937;
  text-align: left;
}
.iw-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 15px;
  align-items: center;
  background: #f6f8fa;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 8px 12px;
  margin-bottom: 10px;
}
.iw-ctrl {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  line-height: 1.3;
}
.iw-ctrl label {
  font-weight: 500;
  color: #1f2937;
  white-space: nowrap;
  cursor: pointer;
}
.iw-ctrl input[type="range"] {
  width: 126px;
  margin: 0;
  cursor: pointer;
  accent-color: #1d6fa5;
  vertical-align: middle;
}
.iw-ctrl input[type="checkbox"] {
  width: 16px;
  height: 16px;
  margin: 0;
  cursor: pointer;
  accent-color: #1d6fa5;
}
.iw-ctrl input[type="range"]:focus-visible,
.iw-ctrl input[type="checkbox"]:focus-visible,
.iw-select:focus-visible,
.iw-btn:focus-visible {
  outline: 2px solid #1d6fa5;
  outline-offset: 2px;
}
@supports not (accent-color: #1d6fa5) {
  .iw-ctrl input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    background: transparent;
    height: 20px;
  }
  .iw-ctrl input[type="range"]::-webkit-slider-runnable-track {
    height: 6px;
    border-radius: 3px;
    background: #cbd5e1;
  }
  .iw-ctrl input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #1d6fa5;
    border: 2px solid #ffffff;
    margin-top: -5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
  }
  .iw-ctrl input[type="range"]::-moz-range-track {
    height: 6px;
    border-radius: 3px;
    background: #cbd5e1;
  }
  .iw-ctrl input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #1d6fa5;
    border: 2px solid #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
  }
}
.iw-val {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12px;
  font-weight: 600;
  color: #16527a;
  background: #e8f2f8;
  border: 1px solid #cbe0ee;
  border-radius: 5px;
  padding: 2px 6px;
  min-width: 3.4em;
  text-align: right;
  display: inline-block;
}
.iw-select {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 500;
  color: #1f2937;
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 3px 8px;
  cursor: pointer;
}
.iw-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  color: #1f2937;
  background: #ffffff;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
}
.iw-btn:hover {
  background: #edf2f7;
  border-color: #94a3b8;
}
.iw-btn-primary {
  background: #1d6fa5;
  color: #ffffff;
  border-color: #1d6fa5;
}
.iw-btn-primary:hover {
  background: #165885;
  border-color: #165885;
}
.iw-status {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #475569;
  margin-left: auto;
}
.iw-progress {
  --progress-bar-radius: 999px;
  --progress-bar-fill: #1d6fa5;
  appearance: none;
  border: 0 solid;
  overflow: clip;
  inline-size: 74px;
  block-size: 8px;
  border-radius: var(--progress-bar-radius);
  background: #cbd5e1;
  vertical-align: middle;
}
.iw-progress::-webkit-progress-bar {
  background: transparent;
}
.iw-progress::-webkit-progress-value {
  background: var(--progress-bar-fill, #1d6fa5);
  border-radius: var(--progress-bar-radius);
  transition: inline-size 0.12s linear;
}
.iw-progress::-moz-progress-bar {
  background: var(--progress-bar-fill, #1d6fa5);
  border-radius: var(--progress-bar-radius);
}
.iw-row {
  display: flex;
  gap: 12px;
  align-items: stretch;
  width: 100%;
}
.iw-col {
  flex: 1 1 0;
  min-width: 0;
}
.iw-canvas {
  display: block;
  width: 100%;
  background: #ffffff;
  border-radius: 6px;
}
.iw-caption {
  font-size: 12px;
  color: #64748b;
  text-align: center;
  margin: 6px 0 0;
  line-height: 1.42;
}
@media (max-width: 760px) {
  .iw-row { flex-direction: column; }
}
"""

WIDGET_ENGINE_JS = r"""
/* Shared PRNG, Galton Simulator, 2D HPD Contours & Retina Scientific Canvas Engine */
window.SBI = (function() {
  var PAL = {
    obs: '#1f2937',
    exact: '#111827',
    truth: '#dc2626',
    rejected: 'rgba(156,163,175,0.45)',
    abc: '#d97706',
    NPE: '#2563eb',
    NLE: '#059669',
    NRE: '#7c3aed'
  };

  function createRNG(seed) {
    seed = (seed | 0) || 1;
    return function() {
      seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussN(rng) {
    var u = rng() * 0.9999998 + 1e-7;
    var v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  function sigmoid(z) { return 1.0 / (1.0 + Math.exp(-z)); }

  function normPdf(x, mu, sig) {
    var z = (x - mu) / sig;
    return Math.exp(-0.5 * z * z) / (sig * 2.5066282746310002);
  }

  function gaussCdf(x) {
    var t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
    var d = 0.3989422804014327 * Math.exp(-0.5 * x * x);
    var p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1.0 - p : p;
  }

  /* Chi-square(df=5) pdf: x^(3/2) exp(-x/2) / (2^(5/2) Gamma(5/2)), Gamma(2.5) = 1.329340388 */
  function chi2Pdf5(th) {
    if (th <= 0) return 0.0;
    return Math.pow(th, 1.5) * Math.exp(-0.5 * th) / (5.65685424949 * 1.32934038818);
  }

  /* Gamma(shape=3, scale=1/3) pdf: x^2 * 27/2 * exp(-3x) = 13.5 x^2 exp(-3x) */
  function gamma3Pdf(x) {
    if (x <= 0) return 0.0;
    return 13.5 * x * x * Math.exp(-3.0 * x);
  }

  /* Exact Dynamic-Programming Galton/Plinko bin probabilities pi_R(k | alpha, kappa) */
  function plinkoProbs(alpha, kappa, R) {
    R = R || 12;
    var sqR = Math.sqrt(R);
    var pi = [1.0];
    for (var t = 0; t < R; t++) {
      var nxt = new Array(t + 2).fill(0.0);
      for (var k = 0; k <= t; k++) {
        var st = 2 * k - t;
        var pR = sigmoid(alpha - kappa * st / sqR);
        nxt[k] += pi[k] * (1.0 - pR);
        nxt[k + 1] += pi[k] * pR;
      }
      pi = nxt;
    }
    return pi;
  }

  function binomProbs(pRight, R) {
    R = R || 12;
    var pi = [1.0];
    for (var t = 0; t < R; t++) {
      var nxt = new Array(t + 2).fill(0.0);
      for (var k = 0; k <= t; k++) {
        nxt[k] += pi[k] * (1.0 - pRight);
        nxt[k + 1] += pi[k] * pRight;
      }
      pi = nxt;
    }
    return pi;
  }

  function simPlinko(alpha, kappa, nBalls, R, rng, keepPaths) {
    R = R || 12;
    var sqR = Math.sqrt(R);
    var counts = new Array(R + 1).fill(0);
    var paths = [];
    var maxKeep = keepPaths || 0;
    for (var n = 0; n < nBalls; n++) {
      var k = 0;
      var sx = maxKeep > n ? [0] : null;
      var sy = maxKeep > n ? [0] : null;
      for (var t = 0; t < R; t++) {
        var st = 2 * k - t;
        if (rng() < sigmoid(alpha - kappa * st / sqR)) k++;
        if (sx) { sx.push(2 * k - (t + 1)); sy.push(-(t + 1)); }
      }
      counts[k]++;
      if (sx) paths.push({ xs: sx, ys: sy, bin: k });
    }
    return { counts: counts, paths: paths };
  }

  function histStats(counts) {
    var tot = 0, m = 0, v = 0;
    for (var k = 0; k < counts.length; k++) { tot += counts[k]; m += k * counts[k]; }
    m /= Math.max(1, tot);
    for (var k2 = 0; k2 < counts.length; k2++) {
      var d = k2 - m;
      v += counts[k2] * d * d;
    }
    v /= Math.max(1, tot);
    return { mean: m, variance: v, std: Math.sqrt(v), total: tot };
  }

  /* Histogram distances between two count vectors (normalised to empirical frequencies) */
  function histDistance(h1, h2, kind) {
    var n1 = 0, n2 = 0;
    for (var i = 0; i < h1.length; i++) { n1 += h1[i]; n2 += h2[i]; }
    n1 = Math.max(1, n1); n2 = Math.max(1, n2);
    if (kind === 'wasserstein') {
      var c1 = 0, c2 = 0, w = 0;
      for (var k = 0; k < h1.length - 1; k++) {
        c1 += h1[k] / n1;
        c2 += h2[k] / n2;
        w += Math.abs(c1 - c2);
      }
      return w;
    }
    if (kind === 'tv') {
      var s = 0;
      for (var k2 = 0; k2 < h1.length; k2++) s += Math.abs(h1[k2] / n1 - h2[k2] / n2);
      return 0.5 * s;
    }
    /* l2 on frequencies */
    var sq = 0;
    for (var k3 = 0; k3 < h1.length; k3++) {
      var d = h1[k3] / n1 - h2[k3] / n2;
      sq += d * d;
    }
    return Math.sqrt(sq);
  }

  function linspace(lo, hi, n) {
    var arr = new Array(n);
    if (n <= 1) { arr[0] = lo; return arr; }
    var step = (hi - lo) / (n - 1);
    for (var i = 0; i < n; i++) arr[i] = lo + i * step;
    return arr;
  }

  function trapzNorm(xs, ys) {
    var area = 0.0;
    for (var i = 0; i < xs.length - 1; i++) area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]);
    var inv = area > 1e-14 ? 1.0 / area : 1.0;
    return ys.map(function(v) { return v * inv; });
  }

  function histogramDensity(samples, lo, hi, nBins, weights) {
    var binW = (hi - lo) / nBins;
    var counts = new Array(nBins).fill(0.0);
    var xs = new Array(nBins);
    for (var b = 0; b < nBins; b++) xs[b] = lo + (b + 0.5) * binW;
    var tot = 0.0;
    for (var i = 0; i < samples.length; i++) {
      var v = samples[i];
      if (v < lo || v > hi) continue;
      var idx = Math.floor((v - lo) / binW);
      if (idx < 0) idx = 0;
      if (idx >= nBins) idx = nBins - 1;
      var w = weights ? weights[i] : 1.0;
      counts[idx] += w;
      tot += w;
    }
    if (tot > 1e-14) {
      for (var k = 0; k < nBins; k++) counts[k] /= (tot * binW);
    }
    return { xs: xs, ys: counts, binW: binW };
  }

  /* 2D HPD density threshold & KDE on (alpha, kappa) grid [na][nk] */
  function hpdLevel(postAK, da, dk, mass) {
    var flat = [];
    for (var i = 0; i < postAK.length; i++) {
      var row = postAK[i];
      for (var j = 0; j < row.length; j++) if (row[j] > 0) flat.push(row[j]);
    }
    if (!flat.length) return 1.0;
    flat.sort(function(a, b) { return b - a; });
    var cellArea = da * dk, acc = 0.0;
    for (var k = 0; k < flat.length; k++) {
      acc += flat[k] * cellArea;
      if (acc >= mass) return flat[k];
    }
    return flat[flat.length - 1];
  }

  function marginalsAK(postAK, da, dk) {
    var na = postAK.length, nk = postAK[0].length;
    var ma = new Array(na).fill(0.0), mk = new Array(nk).fill(0.0);
    for (var i = 0; i < na; i++) {
      for (var j = 0; j < nk; j++) {
        ma[i] += postAK[i][j] * dk;
        mk[j] += postAK[i][j] * da;
      }
    }
    return { alpha: ma, kappa: mk };
  }

  function kde2DAK(alphas, kappas, aGrid, kGrid, da, dk) {
    var n = alphas.length, na = aGrid.length, nk = kGrid.length;
    var out = new Array(na);
    for (var i = 0; i < na; i++) out[i] = new Float64Array(nk);
    if (n < 3) return out;
    var factor = Math.pow(n, -1 / 6);
    var hA = Math.max(0.08, 0.42 * factor), hK = Math.max(0.12, 0.55 * factor);
    for (var s = 0; s < n; s++) {
      var av = alphas[s], kv = kappas[s];
      for (var ia = 0; ia < na; ia++) {
        var za = (aGrid[ia] - av) / hA;
        if (Math.abs(za) > 3.4) continue;
        var wa = Math.exp(-0.5 * za * za);
        for (var ik = 0; ik < nk; ik++) {
          var zk = (kGrid[ik] - kv) / hK;
          if (Math.abs(zk) > 3.4) continue;
          out[ia][ik] += wa * Math.exp(-0.5 * zk * zk);
        }
      }
    }
    var sum = 0.0;
    for (var i2 = 0; i2 < na; i2++) for (var j2 = 0; j2 < nk; j2++) sum += out[i2][j2];
    var norm = sum * da * dk;
    if (norm > 1e-14) {
      for (var i3 = 0; i3 < na; i3++) for (var j3 = 0; j3 < nk; j3++) out[i3][j3] /= norm;
    }
    return out;
  }

  /* Retina Canvas Plotter */
  function prepCanvas(canvas, heightPx) {
    if (!canvas) return null;
    var parentW = (canvas.parentElement && canvas.parentElement.clientWidth) ? canvas.parentElement.clientWidth : 440;
    var w = Math.max(180, Math.floor(parentW));
    var h = heightPx || parseInt(canvas.getAttribute('data-height') || '300', 10);
    var dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = h + 'px';
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  function niceTicks(lo, hi, count) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var raw = span / Math.max(2, count || 5);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    var step = (norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10) * mag;
    var first = Math.ceil((lo - 1e-9 * step) / step) * step;
    var ticks = [];
    for (var v = first; v <= hi + 1e-7 * step && ticks.length < 20; v += step) {
      ticks.push(Math.abs(v) < 1e-12 ? 0 : Number(v.toPrecision(10)));
    }
    return ticks;
  }

  function fmtTick(v) {
    if (Math.abs(v) === 0) return '0';
    if (Math.abs(v) >= 1000 || Math.abs(v) < 0.01) return v.toExponential(0);
    return Number(v.toFixed(3)).toString();
  }

  function createAxes(surface, opts) {
    var ctx = surface.ctx, W = surface.w, H = surface.h;
    var pad = Object.assign({ l: 46, r: 14, t: opts.title ? 26 : 12, b: 36 }, opts.pad || {});
    var x0 = pad.l, y0 = pad.t, pw = Math.max(40, W - pad.l - pad.r), ph = Math.max(40, H - pad.t - pad.b);
    var xRange = opts.xRange || [0, 1], yRange = opts.yRange || [0, 1];
    var xMin = xRange[0], xMax = xRange[1], yMin = yRange[0], yMax = yRange[1];

    function sx(x) { return x0 + ((x - xMin) / (xMax - xMin)) * pw; }
    function sy(y) { return y0 + ph - ((y - yMin) / (yMax - yMin)) * ph; }

    var xticks = opts.xTicks || niceTicks(xMin, xMax, opts.nXTicks || 5);
    var ytics = opts.yTicks || niceTicks(yMin, yMax, opts.nYTicks || 4);

    ctx.save();
    ctx.strokeStyle = '#eef2f6';
    ctx.lineWidth = 1;
    for (var i = 0; i < xticks.length; i++) {
      var gx = sx(xticks[i]);
      ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y0 + ph); ctx.stroke();
    }
    for (var j = 0; j < ytics.length; j++) {
      var gy = sy(ytics[j]);
      ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x0 + pw, gy); ctx.stroke();
    }

    if (opts.title) {
      ctx.fillStyle = '#1f2937';
      ctx.font = '600 12.5px Inter, -apple-system, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(opts.title, x0, 6);
    }

    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.15;
    ctx.beginPath();
    ctx.moveTo(x0, y0); ctx.lineTo(x0, y0 + ph); ctx.lineTo(x0 + pw, y0 + ph);
    ctx.stroke();

    ctx.fillStyle = '#475569';
    ctx.font = '11px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (var ti = 0; ti < xticks.length; ti++) {
      var tx = sx(xticks[ti]);
      ctx.strokeStyle = '#94a3b8';
      ctx.beginPath(); ctx.moveTo(tx, y0 + ph); ctx.lineTo(tx, y0 + ph + 4); ctx.stroke();
      ctx.fillText(fmtTick(xticks[ti]), tx, y0 + ph + 6);
    }
    if (!opts.hideYAxis) {
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (var tj = 0; tj < ytics.length; tj++) {
        var ty = sy(ytics[tj]);
        ctx.strokeStyle = '#94a3b8';
        ctx.beginPath(); ctx.moveTo(x0 - 4, ty); ctx.lineTo(x0, ty); ctx.stroke();
        ctx.fillText(fmtTick(ytics[tj]), x0 - 6, ty);
      }
    }

    ctx.fillStyle = '#334155';
    ctx.font = '500 11.5px Inter, -apple-system, sans-serif';
    if (opts.xLabel) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(opts.xLabel, x0 + pw * 0.5, H - 2);
    }
    if (opts.yLabel && !opts.hideYAxis) {
      ctx.save();
      ctx.translate(11, y0 + ph * 0.5);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(opts.yLabel, 0, 0);
      ctx.restore();
    }
    ctx.restore();

    function withClip(fn) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x0, y0, pw, ph);
      ctx.clip();
      fn(ctx);
      ctx.restore();
    }

    return {
      ctx: ctx, x0: x0, y0: y0, pw: pw, ph: ph,
      xMin: xMin, xMax: xMax, yMin: yMin, yMax: yMax,
      sx: sx, sy: sy, withClip: withClip,
      line: function(xs, ys, color, width, dash) {
        withClip(function(c) {
          c.strokeStyle = color || '#2563eb';
          c.lineWidth = width || 2.2;
          c.setLineDash(dash || []);
          c.beginPath();
          var started = false;
          for (var k = 0; k < xs.length; k++) {
            if (xs[k] === null || ys[k] === null || !isFinite(ys[k])) { started = false; continue; }
            var px = sx(xs[k]), py = sy(ys[k]);
            if (!started) { c.moveTo(px, py); started = true; } else { c.lineTo(px, py); }
          }
          c.stroke();
          c.setLineDash([]);
        });
      },
      fillUnder: function(xs, ys, fillRgba, baseY) {
        if (!xs.length) return;
        var by = sy(baseY !== undefined ? baseY : 0);
        withClip(function(c) {
          c.fillStyle = fillRgba || 'rgba(37,99,235,0.14)';
          c.beginPath();
          c.moveTo(sx(xs[0]), by);
          for (var k = 0; k < xs.length; k++) c.lineTo(sx(xs[k]), sy(ys[k]));
          c.lineTo(sx(xs[xs.length - 1]), by);
          c.closePath();
          c.fill();
        });
      },
      band: function(xs, loYs, hiYs, fillRgba) {
        if (!xs.length) return;
        withClip(function(c) {
          c.fillStyle = fillRgba || 'rgba(127,140,141,0.24)';
          c.beginPath();
          for (var k = 0; k < xs.length; k++) {
            var px = sx(xs[k]), py = sy(hiYs[k]);
            if (k === 0) c.moveTo(px, py); else c.lineTo(px, py);
          }
          for (var m = xs.length - 1; m >= 0; m--) c.lineTo(sx(xs[m]), sy(loYs[m]));
          c.closePath();
          c.fill();
        });
      },
      bars: function(xs, ys, barWidthData, fillRgba, strokeColor, xOffset) {
        var off = xOffset || 0;
        withClip(function(c) {
          var by = sy(0);
          c.fillStyle = fillRgba || 'rgba(37,99,235,0.52)';
          c.strokeStyle = strokeColor || 'rgba(30,64,175,0.8)';
          c.lineWidth = 0.8;
          for (var k = 0; k < xs.length; k++) {
            var xc = xs[k] + off;
            var left = sx(xc - 0.5 * barWidthData);
            var right = sx(xc + 0.5 * barWidthData);
            var top = sy(ys[k]);
            var bw = Math.max(1, right - left);
            var bh = by - top;
            if (Array.isArray(fillRgba)) c.fillStyle = fillRgba[k];
            c.fillRect(left, top, bw, bh);
            c.strokeRect(left, top, bw, bh);
          }
        });
      },
      scatter: function(xs, ys, color, radius) {
        var r = radius || 2.5;
        withClip(function(c) {
          c.fillStyle = color || 'rgba(156,163,175,0.45)';
          for (var k = 0; k < xs.length; k++) {
            if (Array.isArray(color)) c.fillStyle = color[k];
            c.beginPath();
            c.arc(sx(xs[k]), sy(ys[k]), r, 0, 6.2831853);
            c.fill();
          }
        });
      },
      cross: function(x, y, color, sizePx, lw) {
        var s = sizePx || 6.5, px = sx(x), py = sy(y);
        ctx.save();
        ctx.strokeStyle = color || '#dc2626';
        ctx.lineWidth = lw || 2.8;
        ctx.beginPath();
        ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s);
        ctx.moveTo(px - s, py + s); ctx.lineTo(px + s, py - s);
        ctx.stroke();
        ctx.restore();
      },
      star: function(x, y, color, rOuter) {
        var cx = sx(x), cy = sy(y), R = rOuter || 7, r = R * 0.44;
        ctx.save();
        ctx.fillStyle = color || '#dc2626';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        for (var i = 0; i < 10; i++) {
          var a = -Math.PI / 2 + i * Math.PI / 5;
          var rad = (i % 2 === 0) ? R : r;
          var px = cx + rad * Math.cos(a), py = cy + rad * Math.sin(a);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      },
      vline: function(x, color, lw, dash) {
        this.line([x, x], [yMin, yMax], color, lw || 1.5, dash || [5, 4]);
      },
      hline: function(y, color, lw, dash) {
        this.line([xMin, xMax], [y, y], color, lw || 1.5, dash || [5, 4]);
      },
      heatmapAK: function(xGrid, yGrid, Z_xy, rgbBase, maxAlpha) {
        /* Z_xy[i][j] indexed by (xGrid[i], yGrid[j]) */
        var nx = xGrid.length, ny = yGrid.length, zMax = 0;
        for (var i = 0; i < nx; i++) for (var j = 0; j < ny; j++) if (Z_xy[i][j] > zMax) zMax = Z_xy[i][j];
        if (zMax <= 0) return;
        var rgb = rgbBase || '37,99,235', aCap = maxAlpha || 0.55;
        var dx = nx > 1 ? (xGrid[1] - xGrid[0]) : 0.1;
        var dy = ny > 1 ? (yGrid[1] - yGrid[0]) : 0.1;
        withClip(function(c) {
          for (var i2 = 0; i2 < nx; i2++) {
            var xL = sx(xGrid[i2] - 0.5 * dx), xR = sx(xGrid[i2] + 0.5 * dx);
            for (var j2 = 0; j2 < ny; j2++) {
              var val = Z_xy[i2][j2] / zMax;
              if (val < 0.03) continue;
              var yTop = sy(yGrid[j2] + 0.5 * dy), yBot = sy(yGrid[j2] - 0.5 * dy);
              c.fillStyle = 'rgba(' + rgb + ',' + (aCap * Math.pow(val, 0.72)).toFixed(3) + ')';
              c.fillRect(xL, yTop, Math.ceil(xR - xL) + 0.5, Math.ceil(yBot - yTop) + 0.5);
            }
          }
        });
      },
      contourAK: function(xGrid, yGrid, Z_xy, level, color, lw, dash) {
        /* Marching squares on Z_xy[i][j] at threshold `level` */
        var nx = xGrid.length, ny = yGrid.length;
        function interp(p1, p2, v1, v2) {
          var d = v2 - v1;
          var t = Math.abs(d) > 1e-12 ? (level - v1) / d : 0.5;
          t = Math.max(0, Math.min(1, t));
          return p1 + t * (p2 - p1);
        }
        withClip(function(c) {
          c.strokeStyle = color || '#111827';
          c.lineWidth = lw || 2.0;
          c.setLineDash(dash || []);
          c.beginPath();
          for (var i = 0; i < nx - 1; i++) {
            var xA = xGrid[i], xB = xGrid[i + 1];
            for (var j = 0; j < ny - 1; j++) {
              var yA = yGrid[j], yB = yGrid[j + 1];
              var v00 = Z_xy[i][j], v10 = Z_xy[i + 1][j], v11 = Z_xy[i + 1][j + 1], v01 = Z_xy[i][j + 1];
              var idx = (v00 >= level ? 1 : 0) | (v10 >= level ? 2 : 0) | (v11 >= level ? 4 : 0) | (v01 >= level ? 8 : 0);
              if (idx === 0 || idx === 15) continue;
              var eB = [interp(xA, xB, v00, v10), yA];
              var eR = [xB, interp(yA, yB, v10, v11)];
              var eT = [interp(xA, xB, v01, v11), yB];
              var eL = [xA, interp(yA, yB, v00, v01)];
              var segs = [];
              if (idx === 1 || idx === 14) segs.push([eL, eB]);
              else if (idx === 2 || idx === 13) segs.push([eB, eR]);
              else if (idx === 3 || idx === 12) segs.push([eL, eR]);
              else if (idx === 4 || idx === 11) segs.push([eR, eT]);
              else if (idx === 5) { segs.push([eL, eT]); segs.push([eB, eR]); }
              else if (idx === 6 || idx === 9) segs.push([eB, eT]);
              else if (idx === 7 || idx === 8) segs.push([eL, eT]);
              else if (idx === 10) { segs.push([eL, eB]); segs.push([eR, eT]); }
              for (var s = 0; s < segs.length; s++) {
                c.moveTo(sx(segs[s][0][0]), sy(segs[s][0][1]));
                c.lineTo(sx(segs[s][1][0]), sy(segs[s][1][1]));
              }
            }
          }
          c.stroke();
          c.setLineDash([]);
        });
      },
      legend: function(items, pos) {
        if (!items || !items.length) return;
        ctx.save();
        ctx.font = '11px Inter, -apple-system, sans-serif';
        var maxW = 0;
        for (var i = 0; i < items.length; i++) {
          var tw = ctx.measureText(items[i].label).width;
          if (tw > maxW) maxW = tw;
        }
        var boxW = maxW + 30, boxH = items.length * 15.5 + 8;
        var lx = (pos === 'tr' || pos === 'br') ? (x0 + pw - boxW - 6) : (x0 + 6);
        var ly = (pos === 'bl' || pos === 'br') ? (y0 + ph - boxH - 6) : (y0 + 6);
        ctx.fillStyle = 'rgba(255,255,255,0.88)';
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.fillRect(lx, ly, boxW, boxH);
        ctx.strokeRect(lx, ly, boxW, boxH);
        for (var k = 0; k < items.length; k++) {
          var it = items[k], ry = ly + 11 + k * 15.5;
          if (it.type === 'dot') {
            ctx.fillStyle = it.color;
            ctx.beginPath(); ctx.arc(lx + 12, ry, 3.5, 0, 6.2832); ctx.fill();
          } else if (it.type === 'star') {
            ctx.fillStyle = it.color;
            ctx.beginPath(); ctx.arc(lx + 12, ry, 4.2, 0, 6.2832); ctx.fill();
          } else if (it.type === 'bar') {
            ctx.fillStyle = it.color;
            ctx.fillRect(lx + 6, ry - 4, 13, 8);
          } else {
            ctx.strokeStyle = it.color;
            ctx.lineWidth = 2.3;
            ctx.setLineDash(it.dash || []);
            ctx.beginPath(); ctx.moveTo(lx + 5, ry); ctx.lineTo(lx + 19, ry); ctx.stroke();
            ctx.setLineDash([]);
          }
          ctx.fillStyle = '#1f2937';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(it.label, lx + 24, ry);
        }
        ctx.restore();
      },
      text: function(xFrac, yFrac, str, color, align) {
        ctx.save();
        ctx.fillStyle = color || '#1f2937';
        ctx.font = '600 11.5px Inter, -apple-system, sans-serif';
        ctx.textAlign = align || 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(str, x0 + xFrac * pw, y0 + yFrac * ph);
        ctx.restore();
      }
    };
  }

  function observeVisibility(el, onHide, onShow) {
    if (!el || typeof IntersectionObserver === 'undefined') return;
    var obs = new IntersectionObserver(function(entries) {
      entries.forEach(function(ent) {
        if (!ent.isIntersecting && onHide) onHide();
        if (ent.isIntersecting && onShow) onShow();
      });
    }, { threshold: 0.02 });
    obs.observe(el);
  }

  function onResize(el, cb) {
    if (!el || typeof ResizeObserver === 'undefined') return;
    var lastW = 0;
    var ro = new ResizeObserver(function(entries) {
      var w = Math.round(entries[0].contentRect.width);
      if (w > 0 && Math.abs(w - lastW) > 2) {
        lastW = w;
        cb();
      }
    });
    ro.observe(el);
  }

  return {
    PAL: PAL,
    createRNG: createRNG,
    gaussN: gaussN,
    sigmoid: sigmoid,
    normPdf: normPdf,
    gaussCdf: gaussCdf,
    chi2Pdf5: chi2Pdf5,
    gamma3Pdf: gamma3Pdf,
    plinkoProbs: plinkoProbs,
    binomProbs: binomProbs,
    simPlinko: simPlinko,
    histStats: histStats,
    histDistance: histDistance,
    linspace: linspace,
    trapzNorm: trapzNorm,
    histogramDensity: histogramDensity,
    hpdLevel: hpdLevel,
    marginalsAK: marginalsAK,
    kde2DAK: kde2DAK,
    prepCanvas: prepCanvas,
    createAxes: createAxes,
    observeVisibility: observeVisibility,
    onResize: onResize
  };
})();
"""


def build_iframe_srcdoc(
    widget_html: str,
    widget_js: str,
    height_px: int = 430,
    data_slice: dict[str, object] | None = None,
) -> str:
    """Wrap a widget's HTML + JS + compact data slice into a self-contained `<iframe srcdoc>`."""
    data_script = ""
    if data_slice is not None:
        json_str = json.dumps(data_slice, separators=(",", ":"))
        data_script = f"<script>window.SBI_DATA = {json_str};</script>"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    overflow-x: hidden;
    overflow-y: visible;
    color-scheme: light;
  }}
  .iw-widget {{ margin: 0; }}
  {WIDGET_CSS}
</style>
</head>
<body>
{widget_html}
{data_script}
<script>
{WIDGET_ENGINE_JS}
</script>
<script>
{widget_js}
</script>
</body>
</html>"""
    escaped = _html.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%; height:{height_px}px; min-height:{height_px}px; '
        f'border:none; display:block; background:transparent;" '
        f'title="Interactive SBI Widget"></iframe>'
    )


class InteractiveDemo:
    """Jupyter display object rendering live ipywidgets-style interactive sliders."""

    def __init__(
        self,
        demo_id: str,
        title: str,
        html: str,
        js: str,
        height_px: int = 430,
        data_slice: dict[str, object] | None = None,
    ):
        self.demo_id = demo_id
        self.title = title
        self.html = html
        self.js = js
        self.height_px = height_px
        self.data_slice = data_slice

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {
            "text/html": build_iframe_srcdoc(self.html, self.js, self.height_px, self.data_slice),
            "text/plain": f"<InteractiveDemo {self.demo_id}: {self.title}>",
        }

    def show(self) -> None:
        display(self)

    def to_ipywidget(self) -> widgets.HTML:
        return widgets.HTML(
            value=build_iframe_srcdoc(self.html, self.js, self.height_px, self.data_slice)
        )
