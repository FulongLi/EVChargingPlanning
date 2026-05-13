#!/usr/bin/env python3
"""Lightweight LaTeX manuscript sanity checker."""

from __future__ import annotations

import re
import sys
from pathlib import Path


BLOCKED_TERMS = [
    "TODO",
    "placeholder",
    "MVP",
    "will be added",
    "not yet",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/review_tex.py path/to/main.tex")
        return 2
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Missing file: {root}")
        return 2

    files = [root]
    text = root.read_text(encoding="utf-8")
    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        candidate = root.parent / f"{match.group(1)}.tex"
        if candidate.exists():
            files.append(candidate)

    issues: list[str] = []
    combined = ""
    for file_path in files:
        file_text = file_path.read_text(encoding="utf-8")
        combined += "\n" + file_text
        for line_no, line in enumerate(file_text.splitlines(), start=1):
            lowered = line.lower()
            for term in BLOCKED_TERMS:
                if term.lower() in lowered:
                    issues.append(f"{file_path}:{line_no}: blocked draft term `{term}`")

    required_sections = [
        "Introduction and Literature Review",
        "Problem Formulation",
        "Proposed Grid-Aware Demand Clustering for Location-Capacity Optimisation Methodology",
        "Case Study and Experimental Setup",
        "Results and Discussion",
        "Conclusion",
    ]
    for section in required_sections:
        if section not in combined:
            issues.append(f"missing section: {section}")

    if "\\bibliography{" not in text:
        issues.append("main file does not declare a bibliography")
    if "\\begin{abstract}" not in text:
        issues.append("main file does not contain an abstract")

    if issues:
        print("LaTeX review issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("LaTeX review passed: no blocked draft terms or missing core sections found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
