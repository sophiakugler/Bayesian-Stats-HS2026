"""Registry and Python API for all interactive Galton-Board & SBI lecture widgets.

Usage inside `LECTURES/week10_simulation_based_inference.ipynb`:
    from sbi_lecture import interactive_widgets as iw
    iw.demo("w12_unified_compare")
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from IPython.display import HTML

from .widget_engine import InteractiveDemo, WIDGET_CSS, WIDGET_ENGINE_JS
from .widgets_part1 import WIDGETS_PART1
from .widgets_part2 import WIDGETS_PART2
from .widgets_train import WIDGETS_TRAIN

DEMO_SPECS: dict[str, dict[str, object]] = {**WIDGETS_PART1, **WIDGETS_PART2, **WIDGETS_TRAIN}

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "sbi"
ASSETS_FIG_DIR = ASSETS_DIR / "figures"
PAYLOAD_PATH = ASSETS_DIR / "interactive_payload.json"

_PAYLOAD_CACHE: dict[str, object] | None = None


def load_payload() -> dict[str, object]:
    global _PAYLOAD_CACHE
    if _PAYLOAD_CACHE is None:
        if not PAYLOAD_PATH.exists():
            from .payload_builder import build_interactive_payload
            payload = build_interactive_payload()
            PAYLOAD_PATH.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            _PAYLOAD_CACHE = payload
        else:
            _PAYLOAD_CACHE = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    return _PAYLOAD_CACHE


def demo(demo_id: str) -> InteractiveDemo:
    """Return a live ipywidgets-style interactive slider widget for notebooks and HTML slides."""
    if demo_id not in DEMO_SPECS:
        raise KeyError(f"Unknown demo_id {demo_id!r}. Available: {sorted(DEMO_SPECS)}")
    spec = DEMO_SPECS[demo_id]
    keys = list(spec.get("data_keys", []))
    data_slice = None
    if keys:
        full_data = load_payload()
        data_slice = {}
        for k in keys:
            if k == "train_traj" and demo_id in ("wt_npe_plinko", "wt_nle_plinko", "wt_nre_plinko"):
                tr = dict(full_data["train_traj"])
                keep = demo_id.split("_")[1]  # "npe", "nle", or "nre"
                for m_key in ("npe", "nle", "nre"):
                    if m_key != keep:
                        tr[m_key] = {}
                data_slice[k] = tr
            else:
                data_slice[k] = full_data[k]
    return InteractiveDemo(
        demo_id=demo_id,
        title=str(spec["title"]),
        html=str(spec["html"]).strip(),
        js=str(spec["js"]).strip(),
        height_px=int(spec.get("height", 430)),
        data_slice=data_slice,
    )


def figure_data_uri(filename: str) -> str:
    path = ASSETS_FIG_DIR / filename
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_figure_card(filename: str, caption: str, max_width_px: int = 720) -> HTML:
    uri = figure_data_uri(filename)
    return HTML(
        f'<div style="text-align:center; margin:10px 0;">'
        f'<img src="{uri}" alt="{caption}" '
        f'style="max-width:min(100%, {max_width_px}px); height:auto; border-radius:8px; '
        f'border:1px solid #e2e8f0; background:#ffffff; padding:6px;">'
        f'<p style="font-size:12.5px; color:#64748b; margin:6px 0 0;">{caption}</p>'
        f'</div>'
    )


__all__ = [
    "DEMO_SPECS",
    "WIDGET_CSS",
    "WIDGET_ENGINE_JS",
    "InteractiveDemo",
    "load_payload",
    "demo",
    "figure_data_uri",
    "render_figure_card",
]
