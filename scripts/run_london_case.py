#!/usr/bin/env python3
"""Placeholder entry point for the London/Barnet case study."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evcs_planning.config import load_yaml


def main() -> None:
    config = load_yaml(ROOT / "configs" / "london_barnet.yaml")
    print("London case interface is ready, but real data loaders are not implemented in the MVP.")
    print(f"Loaded case config: {config.get('case_name')}")
    print("Next step: connect the files listed in configs/london_barnet.yaml to the data loader layer.")


if __name__ == "__main__":
    main()
