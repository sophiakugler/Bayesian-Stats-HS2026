"""Simulation-Based Inference (SBI) lecture package for ETH Zürich Bayesian Statistics HS2026.

Provides interactive slider widgets (`interactive_widgets`), exact analytical and numerical
benchmarks (`abc`, `density_estimation`, `neural_sbi`, `validation`, `plinko`), precomputed
offline bundles (`precompute`), continuous-scroll slide export (`export_slides`), and
synchronized lecture notebook generation (`build_notebook`).

Primary pedagogical source (~90%): Christoph Weniger's ICTP-SAIFR 2026 SBI crash-course
(`teaching-2606-ICTP-SAIFR`, MIT License), harmonized with Tilman Tröster's ETH Zürich
`bayesian_statistical_methods` course (MIT License).
"""

from . import plinko
from . import abc
from . import density_estimation
from . import neural_sbi
from . import validation
from . import precompute
from . import visualizations
from . import interactive_widgets

__all__ = [
    "plinko",
    "abc",
    "density_estimation",
    "neural_sbi",
    "validation",
    "precompute",
    "visualizations",
    "interactive_widgets",
]
