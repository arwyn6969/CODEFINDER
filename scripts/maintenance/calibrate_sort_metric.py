#!/usr/bin/env python3
"""Grid-search the sort metric against the fixed German/negative-control corpus."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.services.sort_metric import (
    SORT_METRIC_VERSION,
    SORT_METRIC_WEIGHTS,
    metric_descriptor,
    score_metrics,
)


STEP = 0.05
POSITIVE_SORT_PATH = ROOT / "reports" / "character_sort_match" / "sort_comparison.json"
OUTPUT_DIR = ROOT / "reports" / "negative_control"
REPORT_JSON = OUTPUT_DIR / "sort_metric_calibration.json"
REPORT_MD = OUTPUT_DIR / "sort_metric_calibration.md"

NEGATIVE_CONTROLS = (
    {
        "source_key": "folger_iiif_aspley",
        "label": "Folger Aspley stress test",
        "sort_path": ROOT / "reports" / "negative_control" / "character_sort_match" / "sort_comparison.json",
        "summary_path": ROOT / "reports" / "negative_control" / "negative_control_summary.json",
        "external": False,
    },
    {
        "source_key": "negative_control_bsb10222478",
        "label": "BSB candidate 10222478",
        "sort_path": ROOT / "reports" / "negative_control" / "negative_control_bsb10222478" / "character_sort_match" / "sort_comparison.json",
        "summary_path": ROOT / "reports" / "negative_control" / "negative_control_bsb10222478" / "negative_control_summary.json",
        "external": True,
    },
    {
        "source_key": "negative_control_bsb10326315",
        "label": "BSB candidate 10326315",
        "sort_path": ROOT / "reports" / "negative_control" / "negative_control_bsb10326315" / "character_sort_match" / "sort_comparison.json",
        "summary_path": ROOT / "reports" / "negative_control" / "negative_control_bsb10326315" / "negative_control_summary.json",
        "external": True,
    },
    {
        "source_key": "negative_control_bsb00090503",
        "label": "BSB candidate 00090503",
        "sort_path": ROOT / "reports" / "negative_control" / "negative_control_bsb00090503" / "character_sort_match" / "sort_comparison.json",
        "summary_path": ROOT / "reports" / "negative_control" / "negative_control_bsb00090503" / "negative_control_summary.json",
        "external": True,
    },
)


@dataclass(frozen=True)
class WeightEvaluation:
    weights: dict[str, float]
    external_pass_count: int
    weakest_positive_pair: float
    strongest_negative_pair: float
    margin: float
    positive_pair_averages: dict[str, float]
    negative_controls: list[dict]


def load_json(path: Path):
    return json.loads(path.read_text())


def weight_grid(step: float = STEP) -> Iterable[dict[str, float]]:
    total_steps = int(round(1.0 / step))
    for cosine_steps in range(total_steps + 1):
        cosine_weight = round(cosine_steps * step, 10)
        for avg_steps in range(total_steps - cosine_steps + 1):
            avg_weight = round(avg_steps * step, 10)
            dimension_weight = round(1.0 - cosine_weight - avg_weight, 10)
            yield {
                "cosine_similarity": cosine_weight,
                "avg_fingerprint_score": avg_weight,
                "dimension_similarity": dimension_weight,
            }


def compute_pair_averages(results: list[dict], weights: dict[str, float], filter_source: str | None = None) -> dict[str, float]:
    pair_scores: dict[str, list[float]] = {}
    for item in results:
        for pair, metrics in item.get("pairwise", {}).items():
            if filter_source and filter_source not in pair:
                continue
            pair_scores.setdefault(pair, []).append(score_metrics(metrics, weights=weights))
    return {
        pair: sum(scores) / len(scores)
        for pair, scores in sorted(pair_scores.items())
    }


def evaluate_negative_control(
    sort_results: list[dict],
    summary: dict,
    control: dict,
    weights: dict[str, float],
) -> dict:
    pair_averages = compute_pair_averages(
        sort_results,
        weights=weights,
        filter_source=control["source_key"],
    )
    median_pair_average = statistics.median(pair_averages.values()) if pair_averages else 0.0
    max_pair_average = max(pair_averages.values()) if pair_averages else 0.0
    greenman_ok = summary.get("greenman_matches", 0) == 0
    distributional_ok = bool(summary.get("all_distributional_verdicts_different"))
    pair_thresholds_ok = median_pair_average < 0.60 and max_pair_average <= 0.65
    return {
        "source_key": control["source_key"],
        "label": control["label"],
        "external": control["external"],
        "pair_averages": pair_averages,
        "median_pair_average": median_pair_average,
        "max_pair_average": max_pair_average,
        "greenman_ok": greenman_ok,
        "distributional_ok": distributional_ok,
        "accepted": greenman_ok and distributional_ok and pair_thresholds_ok,
    }


def evaluate_weights(
    positive_results: list[dict],
    positive_baseline: dict[str, float],
    negative_payloads: list[dict],
    weights: dict[str, float],
) -> WeightEvaluation | None:
    positive_pair_averages = compute_pair_averages(positive_results, weights=weights)
    if len(positive_pair_averages) != 6:
        return None
    if any(avg < 0.60 for avg in positive_pair_averages.values()):
        return None
    if any(
        positive_pair_averages[pair] < positive_baseline[pair] - 0.08
        for pair in positive_pair_averages
    ):
        return None

    negative_controls = [
        evaluate_negative_control(payload["sort_results"], payload["summary"], payload["control"], weights)
        for payload in negative_payloads
    ]
    external_controls = [row for row in negative_controls if row["external"]]
    external_pass_count = sum(1 for row in external_controls if row["accepted"])
    strongest_negative_pair = max(row["max_pair_average"] for row in negative_controls)
    weakest_positive_pair = min(positive_pair_averages.values())

    return WeightEvaluation(
        weights=weights,
        external_pass_count=external_pass_count,
        weakest_positive_pair=weakest_positive_pair,
        strongest_negative_pair=strongest_negative_pair,
        margin=weakest_positive_pair - strongest_negative_pair,
        positive_pair_averages=positive_pair_averages,
        negative_controls=negative_controls,
    )


def evaluate_baseline(
    positive_results: list[dict],
    negative_payloads: list[dict],
) -> dict:
    baseline_positive = compute_pair_averages(positive_results, weights=SORT_METRIC_WEIGHTS)
    negative_controls = [
        evaluate_negative_control(payload["sort_results"], payload["summary"], payload["control"], SORT_METRIC_WEIGHTS)
        for payload in negative_payloads
    ]
    return {
        "metric_version": SORT_METRIC_VERSION,
        "weights": dict(SORT_METRIC_WEIGHTS),
        "descriptor": metric_descriptor(),
        "positive_pair_averages": baseline_positive,
        "negative_controls": negative_controls,
    }


def report_row(control: dict) -> dict:
    return {
        "source_key": control["source_key"],
        "label": control["label"],
        "external": control["external"],
        "median_pair_average": round(control["median_pair_average"], 6),
        "max_pair_average": round(control["max_pair_average"], 6),
        "greenman_ok": control["greenman_ok"],
        "distributional_ok": control["distributional_ok"],
        "accepted": control["accepted"],
    }


def build_markdown(payload: dict) -> str:
    baseline = payload["baseline"]
    lines = [
        "# Sort Metric Calibration Report",
        "",
        "This report tests the production sort metric against the fixed German/Kempten positive corpus and the preserved negative-control corpus in `0.05` weight steps.",
        "",
        "## Baseline production metric",
        "",
        f"- Version: `{baseline['metric_version']}`",
        f"- Weights: `{baseline['descriptor']}`",
        "",
        "### Baseline external negative-control status",
        "",
        "| Control | External | Median pair avg | Max pair avg | Greenman | Distributional | Accepted |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]

    for row in baseline["negative_controls"]:
        lines.append(
            f"| `{row['source_key']}` | `{'yes' if row['external'] else 'no'}` | "
            f"{row['median_pair_average']:.3f} | {row['max_pair_average']:.3f} | "
            f"`{'PASS' if row['greenman_ok'] else 'FAIL'}` | "
            f"`{'PASS' if row['distributional_ok'] else 'FAIL'}` | "
            f"`{'PASS' if row['accepted'] else 'FAIL'}` |"
        )

    lines.extend(
        [
            "",
            "## Calibration outcome",
            "",
            f"- Weight combinations evaluated: `{payload['grid_size']}`",
            f"- Constraint-satisfying combinations: `{payload['eligible_count']}`",
            f"- External controls accepted by any combination: `{payload['best_external_pass_count']}`",
        ]
    )

    if payload["chosen_formula"]:
        chosen = payload["chosen_formula"]
        lines.extend(
            [
                f"- Chosen weights: `{chosen['descriptor']}`",
                f"- Weakest German pair after calibration: `{chosen['weakest_positive_pair']:.3f}`",
                f"- Strongest negative-control pair after calibration: `{chosen['strongest_negative_pair']:.3f}`",
                "",
                "This calibration found a valid replacement formula. The production metric may be updated to the chosen weights.",
            ]
        )
    else:
        best = payload["best_attempt"]
        lines.extend(
            [
                "- Chosen weights: `none`",
                "",
                "No eligible weight set accepted any external negative control while keeping all six German pairs above `0.60` and within `0.08` of baseline.",
                f"Best non-accepting attempt: `{best['descriptor']}`",
                f"- Weakest German pair under that attempt: `{best['weakest_positive_pair']:.3f}`",
                f"- Strongest negative-control pair under that attempt: `{best['strongest_negative_pair']:.3f}`",
                "",
                "Conclusion: source choice is not the only blocker. Under the current component set and acceptance rules, metric or threshold insufficiency remains.",
            ]
        )

    lines.extend(
        [
            "",
            "## Best-attempt negative-control table",
            "",
            "| Control | External | Median pair avg | Max pair avg | Accepted |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    best_rows = payload["chosen_formula"]["negative_controls"] if payload["chosen_formula"] else payload["best_attempt"]["negative_controls"]
    for row in best_rows:
        lines.append(
            f"| `{row['source_key']}` | `{'yes' if row['external'] else 'no'}` | "
            f"{row['median_pair_average']:.3f} | {row['max_pair_average']:.3f} | "
            f"`{'PASS' if row['accepted'] else 'FAIL'}` |"
        )

    return "\n".join(lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    positive_results = load_json(POSITIVE_SORT_PATH)
    negative_payloads = [
        {
            "control": control,
            "sort_results": load_json(control["sort_path"]),
            "summary": load_json(control["summary_path"]),
        }
        for control in NEGATIVE_CONTROLS
    ]

    baseline = evaluate_baseline(positive_results, negative_payloads)
    positive_baseline = baseline["positive_pair_averages"]

    evaluations = []
    for weights in weight_grid():
        result = evaluate_weights(
            positive_results=positive_results,
            positive_baseline=positive_baseline,
            negative_payloads=negative_payloads,
            weights=weights,
        )
        if result is not None:
            evaluations.append(result)

    evaluations.sort(
        key=lambda item: (
            item.external_pass_count,
            item.margin,
            item.weakest_positive_pair,
            -item.strongest_negative_pair,
            -abs(item.weights["cosine_similarity"] - SORT_METRIC_WEIGHTS["cosine_similarity"]),
            -abs(item.weights["avg_fingerprint_score"] - SORT_METRIC_WEIGHTS["avg_fingerprint_score"]),
            -abs(item.weights["dimension_similarity"] - SORT_METRIC_WEIGHTS["dimension_similarity"]),
        ),
        reverse=True,
    )

    best_attempt = evaluations[0]
    chosen_formula = best_attempt if best_attempt.external_pass_count > 0 else None

    payload = {
        "baseline": {
            **baseline,
            "negative_controls": [report_row(row) for row in baseline["negative_controls"]],
        },
        "grid_size": len(list(weight_grid())),
        "eligible_count": len(evaluations),
        "best_external_pass_count": best_attempt.external_pass_count,
        "chosen_formula": None,
        "best_attempt": {
            "weights": dict(best_attempt.weights),
            "descriptor": metric_descriptor(best_attempt.weights),
            "external_pass_count": best_attempt.external_pass_count,
            "weakest_positive_pair": best_attempt.weakest_positive_pair,
            "strongest_negative_pair": best_attempt.strongest_negative_pair,
            "margin": best_attempt.margin,
            "positive_pair_averages": best_attempt.positive_pair_averages,
            "negative_controls": [report_row(row) for row in best_attempt.negative_controls],
        },
    }

    if chosen_formula is not None:
        payload["chosen_formula"] = {
            "weights": dict(chosen_formula.weights),
            "descriptor": metric_descriptor(chosen_formula.weights),
            "external_pass_count": chosen_formula.external_pass_count,
            "weakest_positive_pair": chosen_formula.weakest_positive_pair,
            "strongest_negative_pair": chosen_formula.strongest_negative_pair,
            "margin": chosen_formula.margin,
            "positive_pair_averages": chosen_formula.positive_pair_averages,
            "negative_controls": [report_row(row) for row in chosen_formula.negative_controls],
        }

    REPORT_JSON.write_text(json.dumps(payload, indent=2))
    REPORT_MD.write_text(build_markdown(payload))

    if chosen_formula is None:
        print("No acceptable calibrated formula found.")
        print(f"Best attempt: {metric_descriptor(best_attempt.weights)}")
        return

    print("Calibration succeeded.")
    print(f"Chosen formula: {metric_descriptor(chosen_formula.weights)}")


if __name__ == "__main__":
    main()
