"""Export `slides/week10_simulation_based_inference.slides.html` via `nbconvert` from the notebook.

Uses the exact `jupyter nbconvert --to slides` configuration from `course_tools/make_slides.sh`
(JupyterLab Light + Reveal.js `"white"` theme + `course_tools/nbconvert_templates` MathJax 3
`tex-svg.js` + `remove-cell` / `remove-output` / `remove-input` tag preprocessors) so typography,
equations, headings, and cell formatting match Weeks 1–9, while enabling smooth continuous
vertical scrolling across all slides without 700px viewport clipping.

Usage:
    python -m sbi_lecture.export_slides
"""

from __future__ import annotations

import re
from pathlib import Path
import nbformat
from nbconvert import SlidesExporter
from traitlets.config import Config


CONTINUOUS_SCROLL_CSS = """\
<style id="nbconvert-continuous-scroll" type="text/css">
/* Continuous vertical scroll on top of nbconvert Reveal "white" theme */
html, body {
  height: auto !important;
  min-height: 100% !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  scroll-behavior: smooth;
  background-color: #ffffff;
}
body.jp-Notebook {
  --jp-content-font-size1: 19.5px;
  --jp-content-font-size2: 23px;
  --jp-content-font-size3: 27px;
  --jp-content-font-size4: 33px;
  --jp-content-font-size5: 42px;
  --jp-code-font-size: 14.5px;
  margin: 0;
  padding: 0;
}
main {
  height: auto !important;
  overflow: visible !important;
}
.reveal {
  position: relative !important;
  width: 100% !important;
  height: auto !important;
  overflow: visible !important;
  font-size: 20px !important;
}
.reveal .slides {
  position: relative !important;
  width: min(1040px, 94vw) !important;
  height: auto !important;
  left: auto !important;
  right: auto !important;
  top: auto !important;
  bottom: auto !important;
  margin: 0 auto !important;
  padding: 18px 0 80px !important;
  transform: none !important;
  perspective: none !important;
  overflow: visible !important;
  text-align: left;
}
.reveal .slides > section,
.reveal .slides > section > section {
  position: relative !important;
  display: block !important;
  opacity: 1 !important;
  visibility: visible !important;
  transform: none !important;
  top: auto !important;
  left: auto !important;
  width: 100% !important;
  height: auto !important;
  min-height: 0 !important;
  margin: 0 !important;
  overflow: visible !important;
  pointer-events: auto !important;
}
.reveal .slides > section {
  padding: 34px 0 38px !important;
  border-bottom: 1px solid #e5e7eb;
}
.reveal .slides > section:last-child {
  border-bottom: none;
}
.reveal .slides > section > section {
  padding: 0 !important;
}
.reveal .slides .fragment {
  opacity: 1 !important;
  visibility: visible !important;
  transform: none !important;
}
.jp-RenderedHTMLCommon {
  padding-right: 0 !important;
}
.jp-OutputArea-output {
  width: 100%;
  overflow: visible !important;
}
.jp-RenderedHTMLCommon details {
  margin: 12px 0 4px;
  padding: 10px 14px;
  border-left: 3px solid #727272;
  background: #f9fafb;
}
.jp-RenderedHTMLCommon details > summary {
  cursor: pointer;
  user-select: none;
}
/* Discrete Reveal-style bottom-right slide number ("c") */
.nb-slide-number {
  position: fixed;
  right: 16px;
  bottom: 14px;
  z-index: 40;
  font-family: "Source Sans Pro", Helvetica, sans-serif;
  font-size: 14px;
  color: #727272;
  background: rgba(255, 255, 255, 0.9);
  padding: 3px 8px;
  border-radius: 3px;
  pointer-events: none;
}
</style>
"""

CONTINUOUS_SCROLL_SCRIPT = """\
<script>
(function() {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.reveal .slides > section'));
  if (!slides.length) return;

  var badge = document.createElement('div');
  badge.className = 'nb-slide-number';
  badge.setAttribute('aria-hidden', 'true');
  badge.textContent = '1 / ' + slides.length;
  document.body.appendChild(badge);

  function currentSlideIndex() {
    var mid = window.innerHeight * 0.35;
    var best = 0;
    for (var i = 0; i < slides.length; i++) {
      var r = slides[i].getBoundingClientRect();
      if (r.top <= mid) best = i;
    }
    return best;
  }

  function updateNumber() {
    badge.textContent = (currentSlideIndex() + 1) + ' / ' + slides.length;
  }

  window.addEventListener('scroll', updateNumber, { passive: true });
  window.addEventListener('resize', updateNumber);
  updateNumber();

  document.addEventListener('keydown', function(e) {
    var tag = e.target && e.target.tagName ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea') return;
    if (e.key === 'PageDown' || e.key === 'PageUp') {
      e.preventDefault();
      var idx = currentSlideIndex();
      var next = e.key === 'PageDown' ? Math.min(slides.length - 1, idx + 1) : Math.max(0, idx - 1);
      slides[next].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
})();
</script>
"""


def export_slides_html() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / "LECTURES" / "week10_simulation_based_inference.ipynb"
    out_path = repo_root / "slides" / "week10_simulation_based_inference.slides.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)

    # Exact configuration from `course_tools/make_slides.sh`
    cfg = Config()
    cfg.TemplateExporter.extra_template_basedirs = [
        str(repo_root / "course_tools" / "nbconvert_templates")
    ]
    cfg.TemplateExporter.exclude_input_prompt = True
    cfg.TemplateExporter.exclude_output_prompt = True
    cfg.SlidesExporter.mathjax_url = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
    cfg.SlidesExporter.reveal_number = "c"
    cfg.SlidesExporter.reveal_theme = "white"
    cfg.SlidesExporter.reveal_scroll = True
    cfg.TagRemovePreprocessor.enabled = True
    cfg.TagRemovePreprocessor.remove_cell_tags = {"remove-cell"}
    cfg.TagRemovePreprocessor.remove_all_outputs_tags = {"remove-output"}
    cfg.TagRemovePreprocessor.remove_input_tags = {"remove-input"}

    exporter = SlidesExporter(config=cfg)
    html_body, _ = exporter.from_notebook_node(
        nb,
        resources={"metadata": {"name": "week10_simulation_based_inference"}},
    )

    # 1. Inject continuous-scroll CSS right before </head>
    html_body = html_body.replace("</head>", CONTINUOUS_SCROLL_CSS + "\n</head>", 1)

    # 2. Replace Reveal's viewport-locking require([reveal.js], ...) init script with continuous-scroll counter & stepper
    html_body = re.sub(
        r"<script>\s*require\(\s*\{[\s\S]*?Reveal\.addEventListener\('slidechanged',\s*setScrollingSlide\);\s*\}\s*\);\s*</script>",
        CONTINUOUS_SCROLL_SCRIPT,
        html_body,
        count=1,
    )

    out_path.write_text(html_body, encoding="utf-8")
    n_top_slides = len(re.findall(r"<section><section>", html_body))
    print(f"Wrote {out_path} via nbconvert ({len(html_body) / 1024:.1f} KB, {n_top_slides} slides)")
    return out_path


def main() -> None:
    export_slides_html()


if __name__ == "__main__":
    main()
