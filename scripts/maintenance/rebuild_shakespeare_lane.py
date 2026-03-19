#!/usr/bin/env python3
"""
Rebuild the canonical Shakespeare report family.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the canonical Shakespeare lane.")
    parser.add_argument("--output-root", default="reports/shakespeare")
    parser.add_argument("--shareable-root", default="reports/shareable")
    args = parser.parse_args()

    run(
        [
            "scripts/research/generate_shakespeare_canonical_artifacts.py",
            "--output-root",
            args.output_root,
            "--shareable-root",
            args.shareable_root,
        ]
    )


if __name__ == "__main__":
    main()
