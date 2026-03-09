#!/usr/bin/env python3
"""
Run a repo-local negative control against the German/Kempten corpus.

The default control source is the Shakespeare Aspley facsimile already present
in the repository. This is a convenience control, not a final publication-grade
selection, but it provides a baseline false-similarity check without mutating the
canonical German database.
"""

import argparse
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.services.sort_metric import (
    SORT_METRIC_VERSION,
    SORT_METRIC_WEIGHTS,
    metric_descriptor,
    score_metrics,
)

PUBLICATION_THRESHOLDS = {
    "greenman_matches": 0,
    "median_pair_average_lt": 0.60,
    "max_pair_average_le": 0.65,
}


def run_command(args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def source_character_count(db_path: Path, source_name: str) -> int:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM character_instances ci
        JOIN pages p ON ci.page_id = p.id
        JOIN sources s ON p.source_id = s.id
        WHERE s.name = ?
        """,
        (source_name,),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else 0


def analyse_control(db_path: Path, output_dir: Path, control_source: str, extract_limit: int = 0):
    sort_path = output_dir / "character_sort_match" / "sort_comparison.json"
    stats_path = output_dir / "statistical_analysis" / "formal_stats.json"
    greenman_path = output_dir / "greenman_scan" / "matches.json"

    sort_results = json.loads(sort_path.read_text())
    stats_results = json.loads(stats_path.read_text())
    greenman_results = json.loads(greenman_path.read_text()) if greenman_path.exists() else []

    control_pairs = []
    for item in sort_results:
        character = item["character"]
        for pair, metrics in item.get("pairwise", {}).items():
            if control_source in pair:
                control_pairs.append(
                    {
                        "pair": pair,
                        "character": character,
                        "score": score_metrics(metrics),
                    }
                )

    pair_summary = {}
    for entry in control_pairs:
        pair_summary.setdefault(entry["pair"], []).append(entry["score"])

    char_count = source_character_count(db_path, control_source)
    pair_averages = {pair: sum(scores) / len(scores) for pair, scores in sorted(pair_summary.items())}
    median_pair_average = statistics.median(pair_averages.values()) if pair_averages else 0.0
    max_pair_average = max(pair_averages.values()) if pair_averages else 0.0

    stats_rows = []
    all_distributional_verdicts_different = True
    for pair, data in sorted(stats_results.get("ks_test", {}).items()):
        if control_source not in pair:
            continue
        chi = stats_results.get("chi_squared", {}).get(pair, {})
        boot = stats_results.get("bootstrap", {}).get(pair, {})
        row = {
            "pair": pair,
            "verdict_width": data["verdict_width"],
            "verdict_height": data["verdict_height"],
            "chi_squared_verdict": chi.get("verdict", "n/a"),
            "bootstrap_mean": float(boot.get("mean_similarity", 0.0)),
        }
        stats_rows.append(row)
        if (
            row["verdict_width"] != "DIFFERENT"
            or row["verdict_height"] != "DIFFERENT"
            or row["chi_squared_verdict"] != "DIFFERENT"
        ):
            all_distributional_verdicts_different = False

    thresholds = {
        "greenman_matches": len(greenman_results) == PUBLICATION_THRESHOLDS["greenman_matches"],
        "distributional_tests": all_distributional_verdicts_different,
        "median_pair_average": median_pair_average < PUBLICATION_THRESHOLDS["median_pair_average_lt"],
        "max_pair_average": max_pair_average <= PUBLICATION_THRESHOLDS["max_pair_average_le"],
    }
    accepted = all(thresholds.values())

    return {
        "control_source": control_source,
        "db_path": str(db_path.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "extract_limit": extract_limit,
        "metric_version": SORT_METRIC_VERSION,
        "metric_weights": dict(SORT_METRIC_WEIGHTS),
        "metric_descriptor": metric_descriptor(),
        "character_count": char_count,
        "greenman_matches": len(greenman_results),
        "pair_averages": pair_averages,
        "pair_character_counts": {pair: len(scores) for pair, scores in sorted(pair_summary.items())},
        "stats_rows": stats_rows,
        "median_pair_average": median_pair_average,
        "max_pair_average": max_pair_average,
        "all_distributional_verdicts_different": all_distributional_verdicts_different,
        "publication_grade_accepted": accepted,
        "threshold_checks": thresholds,
    }


def build_memo(analysis: dict, output_dir: Path, control_source: str, internal_only: bool):
    char_count = analysis["character_count"]
    page_scope = "all available pages"
    if analysis["extract_limit"]:
        page_scope = f"first {analysis['extract_limit']} pages"
    lines = [
        "# Negative Control Comparison Memo",
        "",
        f"- Control source: `{control_source}`",
        f"- Database: `{analysis['db_path']}`",
        f"- Output directory: `{analysis['output_dir']}`",
        f"- Extracted page scope: `{page_scope}`",
        f"- Sort metric: `{analysis['metric_version']}` ({analysis['metric_descriptor']})",
        f"- Extracted characters in control source: `{char_count}`",
        f"- Greenman matches in control scan: `{analysis['greenman_matches']}`",
        f"- Median pairwise sort average: `{analysis['median_pair_average']:.3f}`",
        f"- Max pairwise sort average: `{analysis['max_pair_average']:.3f}`",
        f"- Publication-grade status: `{'ACCEPTED' if analysis['publication_grade_accepted'] else 'REJECTED'}`",
        "",
        "## Sort similarity against the German corpus",
        "",
        "| Pair | Avg score | Characters compared |",
        "| --- | ---: | ---: |",
    ]
    for pair, avg in analysis["pair_averages"].items():
        character_count = analysis["pair_character_counts"].get(pair, 0)
        lines.append(f"| `{pair}` | {avg:.3f} | {character_count} |")

    lines.extend(
        [
            "",
            "## Statistical checks involving the control source",
            "",
            "| Pair | KS width | KS height | Chi-squared | Bootstrap mean |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in analysis["stats_rows"]:
        lines.append(
            f"| `{row['pair']}` | {row['verdict_width']} | {row['verdict_height']} | "
            f"{row['chi_squared_verdict']} | {row['bootstrap_mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance checks",
            "",
            f"- `0` Greenman matches: `{'PASS' if analysis['threshold_checks']['greenman_matches'] else 'FAIL'}`",
            f"- All KS and chi-squared verdicts `DIFFERENT`: `{'PASS' if analysis['threshold_checks']['distributional_tests'] else 'FAIL'}`",
            f"- Median pairwise sort average `< 0.60`: `{'PASS' if analysis['threshold_checks']['median_pair_average'] else 'FAIL'}`",
            f"- No pairwise sort average exceeds `0.65`: `{'PASS' if analysis['threshold_checks']['max_pair_average'] else 'FAIL'}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if internal_only:
        lines.extend(
            [
                "This repo-local control is intended to estimate how much similarity the current workflow can produce on an unrelated seventeenth-century print already present in the workspace.",
                "In the current run it yields zero Greenman matches, but the character-sort averages remain comparatively high; that means it is useful as an internal stress test, not as a clean publication-grade negative control.",
                "Treat it as a baseline check, and plan to replace it with a deliberately chosen unrelated seventeenth-century comparator before scholarly submission.",
            ]
        )
    elif analysis["publication_grade_accepted"]:
        lines.extend(
            [
                "This run satisfies the current publication-grade thresholds for a negative control.",
                "Keep it in the German/Kempten packet as the unrelated comparator for the current milestone, and continue to preserve any failed control runs as audit history.",
            ]
        )
    else:
        lines.extend(
            [
                "This run does not satisfy the current publication-grade thresholds for a negative control.",
                "Preserve it as audit history, but do not cite it as the accepted scholarly comparator.",
            ]
        )

    (output_dir / "negative_control_memo.md").write_text("\n".join(lines))
    (output_dir / "negative_control_summary.json").write_text(json.dumps(analysis, indent=2))


def resolve_output_dir(base_output_dir: Path, control_source: str, namespace_output: bool) -> Path:
    if namespace_output:
        return base_output_dir / control_source
    return base_output_dir


def main():
    parser = argparse.ArgumentParser(description="Run a repo-local negative control workflow")
    parser.add_argument("--control-source", default="folger_iiif_aspley",
                        help="Source key from data/sources/config.yaml")
    parser.add_argument("--base-db", default="data/forensic.db",
                        help="Canonical German/Kempten database to copy")
    parser.add_argument("--db-path", default="data/negative_control.db",
                        help="Output database path for the control run")
    parser.add_argument("--output-dir", default="reports/negative_control",
                        help="Output directory root for the control artifacts")
    parser.add_argument("--namespace-output", action="store_true",
                        help="Write artifacts under <output-dir>/<control-source>/")
    parser.add_argument("--extract-limit", type=int, default=0,
                        help="Limit OCR extraction to the first N pages (0 for all)")
    parser.add_argument("--ornament-limit", type=int, default=0,
                        help="Limit ornament extraction to the first N pages (0 for all)")
    parser.add_argument("--greenman-limit", type=int, default=0,
                        help="Limit Greenman scan to the first N pages (0 for all)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Reuse existing artifacts and rebuild only the memo/summary")
    args = parser.parse_args()

    base_db = ROOT / args.base_db
    db_path = ROOT / args.db_path
    output_root = ROOT / args.output_dir
    output_dir = resolve_output_dir(output_root, args.control_source, args.namespace_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_base = Path(args.output_dir)
    if args.namespace_output:
        artifact_base = artifact_base / args.control_source

    if not args.skip_pipeline:
        shutil.copy2(base_db, db_path)

        extract_cmd = ["scripts/extract_characters.py", "--source", args.control_source, "--db-path", args.db_path]
        if args.extract_limit:
            extract_cmd.extend(["--limit", str(args.extract_limit)])
        run_command(extract_cmd)

        run_command(["scripts/match_character_sorts.py", "--db-path", args.db_path, "--output-dir", str(artifact_base / "character_sort_match")])
        run_command(["scripts/formal_stats.py", "--db-path", args.db_path, "--output-dir", str(artifact_base / "statistical_analysis")])

        ornaments_cmd = ["scripts/extract_ornaments.py", "--source", args.control_source, "--output-root", str(artifact_base / "ornaments")]
        if args.ornament_limit:
            ornaments_cmd.extend(["--limit", str(args.ornament_limit)])
        run_command(ornaments_cmd)

        greenman_cmd = ["scripts/scan_greenman_all.py", "--source", args.control_source, "--output-dir", str(artifact_base / "greenman_scan")]
        if args.greenman_limit:
            greenman_cmd.extend(["--limit", str(args.greenman_limit)])
        run_command(greenman_cmd)

    analysis = analyse_control(db_path, output_dir, args.control_source, extract_limit=args.extract_limit)
    build_memo(
        analysis=analysis,
        output_dir=output_dir,
        control_source=args.control_source,
        internal_only=args.control_source == "folger_iiif_aspley" and not args.namespace_output,
    )


if __name__ == "__main__":
    main()
