"""Convenience re-export of `sbi_lecture` inside `bayesian_stats_course_tools`."""

from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sbi_lecture import (  # noqa: E402,F401
    abc,
    density_estimation,
    interactive_widgets,
    neural_sbi,
    plinko,
    precompute,
    validation,
    visualizations,
)
