"""Rebuild interactive payload, HTML slides, executed lecture & exercise notebooks, and run pytest."""

from __future__ import annotations

import sys
import pytest

from .payload_builder import main as build_payload_main
from .export_slides import export_slides_html
from .build_notebook import main as build_nb
from .build_exercises import main as build_ex


def main() -> None:
    build_payload_main()
    build_nb()
    export_slides_html()
    build_ex()
    rc = pytest.main(["tests", "-v"])
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
