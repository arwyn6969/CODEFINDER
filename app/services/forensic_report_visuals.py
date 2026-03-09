from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PAIR_ORDER = (
    "bsb_munich_10057380 vs gdz_goettingen_ppn777246686",
    "bsb_munich_10057380 vs google_books_tractatus_brevis",
    "bsb_munich_10057380 vs hab_wolfenbuettel_178_1_theol_1s",
    "gdz_goettingen_ppn777246686 vs google_books_tractatus_brevis",
    "gdz_goettingen_ppn777246686 vs hab_wolfenbuettel_178_1_theol_1s",
    "google_books_tractatus_brevis vs hab_wolfenbuettel_178_1_theol_1s",
)

SOURCE_ABBREVIATIONS = {
    "bsb_munich_10057380": "BSB",
    "gdz_goettingen_ppn777246686": "GDZ",
    "google_books_tractatus_brevis": "Google",
    "hab_wolfenbuettel_178_1_theol_1s": "HAB",
}

CHART_FILES = {
    "pairwise_similarity": "chart_pairwise_similarity.png",
    "manual_review_outcomes": "chart_manual_review_outcomes.png",
    "bootstrap_forest": "chart_bootstrap_forest.png",
    "formal_test_matrix": "chart_formal_test_matrix.png",
}

NAVY = "#0f3460"
GOLD = "#c9a94e"
GREEN = "#4caf50"
ORANGE = "#ff9800"
RED = "#f44336"
LIGHT_GREY = "#e8ecef"
MID_GREY = "#9aa5ad"
DARK_GREY = "#333333"
WHITE = "#ffffff"


def pair_short_label(pair_key: str) -> str:
    left_source, right_source = pair_key.split(" vs ")
    return f"{SOURCE_ABBREVIATIONS[left_source]}-{SOURCE_ABBREVIATIONS[right_source]}"


def ordered_pair_keys(keys: set[str] | list[str] | tuple[str, ...]) -> list[str]:
    key_set = set(keys)
    ordered = [pair for pair in PAIR_ORDER if pair in key_set]
    extras = sorted(key_set - set(ordered))
    return ordered + extras


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def build_pairwise_similarity_rows(sort_results: list[dict]) -> list[dict]:
    pair_scores = defaultdict(list)
    for result in sort_results:
        for pair_key, metrics in result.get("pairwise", {}).items():
            pair_scores[pair_key].append(float(metrics["combined_score"]))

    rows = []
    for pair_key in ordered_pair_keys(pair_scores.keys()):
        scores = pair_scores[pair_key]
        avg_similarity = float(np.mean(scores))
        rows.append(
            {
                "pair": pair_key,
                "pair_key": pair_key,
                "label": pair_short_label(pair_key),
                "avg_similarity": avg_similarity,
                "characters_compared": len(scores),
                "verdict": "SIMILAR_FORMS" if avg_similarity > 0.60 else "UNCLEAR",
            }
        )
    return rows


def build_manual_review_rows(ledger_rows: list[dict]) -> list[dict]:
    counts_by_pair = defaultdict(Counter)
    for row in ledger_rows:
        counts_by_pair[row["source_pair"]][row["manual_assessment"]] += 1

    rows = []
    for pair_key in ordered_pair_keys(counts_by_pair.keys()):
        counts = counts_by_pair[pair_key]
        rows.append(
            {
                "pair_key": pair_key,
                "label": pair_short_label(pair_key),
                "same_design": counts.get("same design", 0),
                "possible_same_sort_block": counts.get("possible same sort/block", 0),
                "inconclusive": counts.get("inconclusive", 0),
                "total": sum(counts.values()),
            }
        )
    return rows


def build_bootstrap_rows(stats_results: dict) -> list[dict]:
    bootstrap_results = stats_results.get("bootstrap", {})
    rows = []
    for pair_key in ordered_pair_keys(bootstrap_results.keys()):
        payload = bootstrap_results[pair_key]
        rows.append(
            {
                "pair_key": pair_key,
                "label": pair_short_label(pair_key),
                "mean_similarity": float(payload["mean_similarity"]),
                "ci_low": float(payload["ci_95_lower"]),
                "ci_high": float(payload["ci_95_upper"]),
                "supports_threshold": float(payload["ci_95_lower"]) > 0.60,
            }
        )
    return rows


def build_formal_test_matrix_rows(stats_results: dict) -> list[dict]:
    ks_results = stats_results.get("ks_test", {})
    chi_results = stats_results.get("chi_squared", {})
    mw_results = stats_results.get("mann_whitney", {})

    rows = []
    keys = set(ks_results) | set(chi_results) | set(mw_results)
    for pair_key in ordered_pair_keys(keys):
        ks_payload = ks_results[pair_key]
        chi_payload = chi_results[pair_key]
        mw_payload = mw_results[pair_key]
        rows.append(
            {
                "pair_key": pair_key,
                "label": pair_short_label(pair_key),
                "ks_width_verdict": ks_payload["verdict_width"],
                "ks_height_verdict": ks_payload["verdict_height"],
                "chi_verdict": chi_payload["verdict"],
                "mw_effect_magnitude": mw_payload["effect_magnitude"].lower(),
                "mw_effect_size": float(mw_payload["effect_size"]),
            }
        )
    return rows


def render_pairwise_similarity_chart(rows: list[dict], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No pairwise similarity rows available for chart generation.")

    labels = [row["label"] for row in rows]
    scores = [row["avg_similarity"] for row in rows]
    colors = [GREEN if score > 0.60 else ORANGE for score in scores]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    y_positions = np.arange(len(rows))
    ax.barh(y_positions, scores, color=colors, edgecolor=NAVY, linewidth=0.8)
    ax.axvline(0.60, color=GOLD, linestyle="--", linewidth=1.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Average combined similarity score", fontsize=10)
    ax.set_title("Pairwise character-form similarity", fontsize=13, color=NAVY, pad=12)
    ax.set_xlim(min(0.55, min(scores) - 0.03), max(0.75, max(scores) + 0.04))
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    for idx, score in enumerate(scores):
        ax.text(score + 0.004, idx, f"{score:.3f}", va="center", ha="left", fontsize=9, color=DARK_GREY)

    ax.text(0.602, len(rows) - 0.35, "0.60 threshold", color=GOLD, fontsize=8, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_manual_review_chart(rows: list[dict], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No manual-review rows available for chart generation.")

    labels = [row["label"] for row in rows]
    same_design = np.array([row["same_design"] for row in rows])
    possible_same = np.array([row["possible_same_sort_block"] for row in rows])
    inconclusive = np.array([row["inconclusive"] for row in rows])

    fig, ax = plt.subplots(figsize=(10, 5.1), dpi=220)
    y_positions = np.arange(len(rows))
    ax.barh(y_positions, same_design, color=GREEN, edgecolor=WHITE, label="same design")
    ax.barh(y_positions, possible_same, left=same_design, color=GOLD, edgecolor=WHITE, label="possible same sort/block")
    ax.barh(y_positions, inconclusive, left=same_design + possible_same, color=LIGHT_GREY, edgecolor=WHITE, label="inconclusive")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Reviewed top matches per pair", fontsize=10)
    ax.set_title("Manual-review outcomes by source pair", fontsize=13, color=NAVY, pad=12)
    ax.set_xlim(0, max(row["total"] for row in rows))
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    for idx, row in enumerate(rows):
        cumulative = 0
        for value in (row["same_design"], row["possible_same_sort_block"], row["inconclusive"]):
            if value:
                ax.text(cumulative + value / 2, idx, str(value), va="center", ha="center", fontsize=9, color=DARK_GREY)
            cumulative += value

    fig.tight_layout()
    fig.savefig(output_path, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_bootstrap_chart(rows: list[dict], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No bootstrap rows available for chart generation.")

    labels = [row["label"] for row in rows]
    means = np.array([row["mean_similarity"] for row in rows])
    lowers = np.array([row["ci_low"] for row in rows])
    uppers = np.array([row["ci_high"] for row in rows])
    errors = np.vstack((means - lowers, uppers - means))
    colors = [GREEN if row["supports_threshold"] else ORANGE for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    y_positions = np.arange(len(rows))
    ax.errorbar(
        means,
        y_positions,
        xerr=errors,
        fmt="o",
        color=NAVY,
        ecolor=MID_GREY,
        elinewidth=2,
        capsize=4,
        markersize=6,
        markerfacecolor=WHITE,
        markeredgewidth=1.5,
    )
    for idx, (mean, color) in enumerate(zip(means, colors)):
        ax.scatter(mean, idx, s=50, color=color, zorder=3, edgecolors=NAVY, linewidths=0.8)
        ax.text(uppers[idx] + 0.006, idx, f"{mean:.3f}", va="center", ha="left", fontsize=9, color=DARK_GREY)

    ax.axvline(0.60, color=GOLD, linestyle="--", linewidth=1.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Bootstrap mean similarity with 95% CI", fontsize=10)
    ax.set_title("Bootstrap similarity intervals", fontsize=13, color=NAVY, pad=12)
    ax.set_xlim(min(0.70, lowers.min() - 0.03), max(1.01, uppers.max() + 0.05))
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.text(0.602, len(rows) - 0.35, "0.60 threshold", color=GOLD, fontsize=8, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _matrix_cell_value(metric: str, value: str) -> tuple[int, str]:
    if metric == "mw":
        label = value.upper().replace("NEGLIGIBLE", "NEG.")
        if value == "negligible":
            return 2, label
        if value == "small":
            return 1, label
        return 0, label

    if value == "SAME":
        return 2, value
    if value == "UNCLEAR":
        return 1, value
    return 0, "DIFF"


def render_formal_test_matrix(rows: list[dict], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No formal-test rows available for chart generation.")

    metrics = [
        ("ks_width_verdict", "KS width", "ks"),
        ("ks_height_verdict", "KS height", "ks"),
        ("chi_verdict", "Chi-sq.", "chi"),
        ("mw_effect_magnitude", "MW effect", "mw"),
    ]
    color_grid = np.zeros((len(rows), len(metrics)))
    label_grid: list[list[str]] = []
    for row_idx, row in enumerate(rows):
        label_row = []
        for col_idx, (key, _, metric_kind) in enumerate(metrics):
            cell_value, cell_label = _matrix_cell_value(metric_kind, row[key])
            color_grid[row_idx, col_idx] = cell_value
            label_row.append(cell_label)
        label_grid.append(label_row)

    cmap = ListedColormap([RED, GOLD, GREEN])
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    ax.imshow(color_grid, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label, _ in metrics], fontsize=10)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["label"] for row in rows], fontsize=10)
    ax.set_title("Formal-test verdict matrix", fontsize=13, color=NAVY, pad=12)

    for row_idx in range(len(rows)):
        for col_idx in range(len(metrics)):
            ax.text(col_idx, row_idx, label_grid[row_idx][col_idx], ha="center", va="center", fontsize=9, color=WHITE)

    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=WHITE, linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.text(0.14, 0.02, "Green = supportive, gold = mixed/provisional, red = divergent/caution", fontsize=9, color=DARK_GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, facecolor=WHITE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_report_charts(
    sort_results: list[dict],
    stats_results: dict,
    manual_review_rows: list[dict],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = build_pairwise_similarity_rows(sort_results)
    manual_rows = build_manual_review_rows(manual_review_rows)
    bootstrap_rows = build_bootstrap_rows(stats_results)
    matrix_rows = build_formal_test_matrix_rows(stats_results)

    outputs = {
        "pairwise_similarity": render_pairwise_similarity_chart(
            pairwise_rows,
            output_dir / CHART_FILES["pairwise_similarity"],
        ),
        "manual_review_outcomes": render_manual_review_chart(
            manual_rows,
            output_dir / CHART_FILES["manual_review_outcomes"],
        ),
        "bootstrap_forest": render_bootstrap_chart(
            bootstrap_rows,
            output_dir / CHART_FILES["bootstrap_forest"],
        ),
        "formal_test_matrix": render_formal_test_matrix(
            matrix_rows,
            output_dir / CHART_FILES["formal_test_matrix"],
        ),
    }
    return outputs
