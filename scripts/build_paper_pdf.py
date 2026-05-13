#!/usr/bin/env python3
"""Compile the LaTeX paper and keep a timestamped PDF copy."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TECTONIC = Path.home() / ".codex/plugins/cache/openai-bundled/latex-tectonic/0.1.1/bin/tectonic"
PAPER = ROOT / "paper" / "main.tex"
BUILD = ROOT / "paper" / "build"
VERSIONS = BUILD / "versions"


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    VERSIONS.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "scripts/export_latex_figures.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/review_tex.py", "paper/main.tex"], cwd=ROOT, check=True)
    subprocess.run([str(TECTONIC), "--outdir", str(BUILD), str(PAPER)], cwd=ROOT, check=True)
    latest = BUILD / "main.pdf"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = VERSIONS / f"main_{timestamp}.pdf"
    shutil.copy2(latest, versioned)
    print(f"Latest PDF: {latest}")
    print(f"Versioned PDF: {versioned}")


if __name__ == "__main__":
    main()
