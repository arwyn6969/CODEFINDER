#!/usr/bin/env python3
"""
Rebuild the canonical German/Kempten report family.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    run(["scripts/match_character_sorts.py"])
    run(["scripts/formal_stats.py"])
    run(["scripts/damage_evolution.py"])
    run(["scripts/research/generate_greenman_review_sheet.py"])
    run(["scripts/research/generate_manual_review_ledger.py"])
    run(["scripts/prepare_proof_images.py"])
    run(["scripts/generate_final_report.py"])
    run(["scripts/generate_pdf_report.py"])
    run(["scripts/generate_discord_summary.py"])


if __name__ == "__main__":
    main()
