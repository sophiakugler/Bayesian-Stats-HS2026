"""Single source of truth for the 40-slide Week 10 SBI lecture & notebook."""

from __future__ import annotations

from .slides_part1 import SLIDES_PART1
from .slides_part2 import SLIDES_PART2

SLIDES: list[dict[str, object]] = SLIDES_PART1 + SLIDES_PART2

__all__ = ["SLIDES"]

