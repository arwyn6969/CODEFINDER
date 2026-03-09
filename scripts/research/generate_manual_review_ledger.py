#!/usr/bin/env python3
"""
Generate a pair-level manual review ledger for the German/Kempten corpus.

This script turns the aggregate sort-comparison report into a reviewable artifact
with actual crop pairs, scores, page references, and contact sheets.
"""

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.services.block_fingerprinter import BlockFingerprinter


DISPLAY_NAMES = {
    "bsb_munich_10057380": "BSB Munich",
    "gdz_goettingen_ppn777246686": "GDZ Goettingen",
    "google_books_tractatus_brevis": "Google Books (Tractatus)",
    "hab_wolfenbuettel_178_1_theol_1s": "HAB Wolfenbuettel",
}

VALID_MANUAL_ASSESSMENTS = {
    "same design",
    "possible same sort/block",
    "inconclusive",
}


class ManualReviewLedgerBuilder:
    def __init__(
        self,
        db_path: str,
        sort_comparison_path: str,
        output_dir: str,
        per_pair: int = 10,
        annotations_path: str | None = None,
    ):
        self.db_path = db_path
        self.sort_comparison_path = Path(sort_comparison_path)
        self.output_dir = Path(output_dir)
        self.per_pair = per_pair
        self.annotations_path = Path(annotations_path) if annotations_path else self.output_dir / "manual_review_annotations.csv"
        self.fingerprinter = BlockFingerprinter()
        self.fp_cache = {}
        self.img_cache = {}

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_high_similarity_pairs(self):
        results = json.loads(self.sort_comparison_path.read_text())
        pair_scores = defaultdict(list)
        pair_characters = defaultdict(list)
        for item in results:
            for pair, metrics in item.get("pairwise", {}).items():
                pair_scores[pair].append(metrics.get("combined_score", 0.0))
                pair_characters[pair].append(
                    {
                        "character": item["character"],
                        "aggregate_score": metrics.get("combined_score", 0.0),
                    }
                )

        high_pairs = []
        for pair, scores in sorted(pair_scores.items()):
            avg = float(np.mean(scores)) if scores else 0.0
            if avg >= 0.6:
                left, right = pair.split(" vs ")
                allowed_chars = [
                    row["character"]
                    for row in sorted(
                        pair_characters[pair],
                        key=lambda row: row["aggregate_score"],
                        reverse=True,
                    )
                    if row["aggregate_score"] > 0
                ]
                high_pairs.append(
                    {
                        "pair": pair,
                        "source_left": left,
                        "source_right": right,
                        "average_score": avg,
                        "character_count": len(scores),
                        "allowed_characters": allowed_chars,
                    }
                )
        return high_pairs

    def gather_instances(self):
        conn = self.get_connection()
        rows = conn.execute(
            """
            SELECT s.name AS source_name,
                   p.page_number,
                   ci.id,
                   ci.character,
                   ci.x,
                   ci.y,
                   ci.width,
                   ci.height,
                   ci.confidence,
                   si.image_path AS crop_path
            FROM character_instances ci
            JOIN pages p ON ci.page_id = p.id
            JOIN sources s ON p.source_id = s.id
            JOIN sort_images si ON si.character_instance_id = ci.id
            WHERE si.image_path IS NOT NULL
            """
        ).fetchall()
        conn.close()

        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["character"]][row["source_name"]].append(dict(row))

        for char_sources in grouped.values():
            for source_name, items in list(char_sources.items()):
                char_sources[source_name] = sorted(
                    items, key=lambda item: item["confidence"], reverse=True
                )
        return grouped

    def _fingerprint(self, crop_rel_path: str):
        if crop_rel_path not in self.fp_cache:
            full_path = ROOT / "reports" / "crops" / crop_rel_path
            self.fp_cache[crop_rel_path] = self.fingerprinter.fingerprint(
                str(full_path), block_type="character"
            )
        return self.fp_cache[crop_rel_path]

    def _image(self, crop_rel_path: str):
        if crop_rel_path not in self.img_cache:
            full_path = ROOT / "reports" / "crops" / crop_rel_path
            self.img_cache[crop_rel_path] = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        return self.img_cache[crop_rel_path]

    def _choose_preliminary_assessment(self, score: float, left: dict, right: dict) -> str:
        dw = abs(left["width"] - right["width"]) / max(left["width"], right["width"], 1)
        dh = abs(left["height"] - right["height"]) / max(left["height"], right["height"], 1)
        if score >= 0.78 and dw <= 0.18 and dh <= 0.18:
            return "same design"
        if score >= 0.68:
            return "possible same sort/block"
        return "inconclusive"

    def build_pair_candidates(self, grouped_instances, pair_info):
        left_source = pair_info["source_left"]
        right_source = pair_info["source_right"]
        candidates = []
        allowed_characters = set(pair_info.get("allowed_characters", []))

        for character, source_map in grouped_instances.items():
            if allowed_characters and character not in allowed_characters:
                continue
            left_items = source_map.get(left_source, [])
            right_items = source_map.get(right_source, [])
            if len(left_items) < 3 or len(right_items) < 3:
                continue

            left_sample = left_items[:12]
            right_sample = right_items[:12]
            best = None

            for left in left_sample:
                fp_left = self._fingerprint(left["crop_path"])
                for right in right_sample:
                    fp_right = self._fingerprint(right["crop_path"])
                    scores = self.fingerprinter.compare(fp_left, fp_right)
                    aggregate = scores.get("aggregate", 0.0)
                    if not best or aggregate > best["combined_score"]:
                        best = {
                            "character": character,
                            "combined_score": aggregate,
                            "score_breakdown": scores,
                            "left": left,
                            "right": right,
                        }

            if best:
                best["assistant_preliminary_assessment"] = self._choose_preliminary_assessment(
                    best["combined_score"], best["left"], best["right"]
                )
                candidates.append(best)

        candidates.sort(key=lambda item: item["combined_score"], reverse=True)
        return candidates[: self.per_pair]

    def build_contact_sheet(self, pair_info, candidates, output_path: Path):
        row_h = 150
        img_box = 96
        width = 1120
        height = 100 + row_h * len(candidates)
        canvas = np.full((height, width, 3), 250, dtype=np.uint8)

        title = (
            f"{DISPLAY_NAMES.get(pair_info['source_left'], pair_info['source_left'])} vs "
            f"{DISPLAY_NAMES.get(pair_info['source_right'], pair_info['source_right'])}"
        )
        cv2.putText(canvas, title, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2)
        cv2.putText(
            canvas,
            "Top pair-level character matches for manual review",
            (24, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (70, 70, 70),
            1,
        )

        for idx, item in enumerate(candidates, start=1):
            y = 100 + (idx - 1) * row_h
            cv2.line(canvas, (20, y - 10), (width - 20, y - 10), (220, 220, 220), 1)

            left_img = self._image(item["left"]["crop_path"])
            right_img = self._image(item["right"]["crop_path"])
            left_resized = self._resize_for_box(left_img, img_box)
            right_resized = self._resize_for_box(right_img, img_box)

            self._paste_centered(canvas, left_resized, 30, y + 8, img_box, img_box)
            self._paste_centered(canvas, right_resized, 160, y + 8, img_box, img_box)

            label = (
                f"#{idx}  '{item['character']}'  score={item['combined_score']:.3f}  "
                f"prelim={item['assistant_preliminary_assessment']}"
            )
            cv2.putText(canvas, label, (300, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (10, 10, 10), 1)
            cv2.putText(
                canvas,
                f"L: p.{item['left']['page_number']} bbox=({item['left']['x']},{item['left']['y']},{item['left']['width']},{item['left']['height']})",
                (300, y + 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (60, 60, 60),
                1,
            )
            cv2.putText(
                canvas,
                f"R: p.{item['right']['page_number']} bbox=({item['right']['x']},{item['right']['y']},{item['right']['width']},{item['right']['height']})",
                (300, y + 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (60, 60, 60),
                1,
            )
            cv2.putText(
                canvas,
                "manual_assessment: ________   notes: ______________________________",
                (300, y + 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (90, 90, 90),
                1,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

    def _resize_for_box(self, image, box_size):
        if image is None:
            return np.full((box_size, box_size), 255, dtype=np.uint8)
        h, w = image.shape[:2]
        scale = min(box_size / max(w, 1), box_size / max(h, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _paste_centered(self, canvas, image, x, y, box_w, box_h):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        ox = x + (box_w - w) // 2
        oy = y + (box_h - h) // 2
        canvas[oy : oy + h, ox : ox + w] = image
        cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (180, 180, 180), 1)

    def _load_annotations(self):
        if not self.annotations_path.exists():
            return {}

        annotations = {}
        with open(self.annotations_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (row["source_pair"], row["rank"])
                if key in annotations:
                    raise ValueError(f"Duplicate manual annotation for {key[0]} rank {key[1]}")
                annotations[key] = row
        return annotations

    def _merge_annotations(self, ledger_rows):
        annotations = self._load_annotations()
        for row in ledger_rows:
            key = (row["source_pair"], str(row["rank"]))
            annotation = annotations.get(key)
            if not annotation:
                continue

            self._validate_annotation_identity(row, annotation)
            manual_assessment = annotation.get("manual_assessment", "").strip()
            manual_notes = annotation.get("manual_notes", "").strip()
            self._validate_manual_fields(manual_assessment, manual_notes, row["source_pair"], row["rank"])
            row["manual_assessment"] = manual_assessment
            row["manual_notes"] = manual_notes

        return ledger_rows

    def _validate_annotation_identity(self, row, annotation):
        for field in ("character", "left_crop", "right_crop"):
            expected = str(row[field])
            observed = str(annotation.get(field, "")).strip()
            if observed and observed != expected:
                raise ValueError(
                    f"Manual annotation mismatch for {row['source_pair']} rank {row['rank']}: "
                    f"{field} expected {expected!r}, got {observed!r}"
                )

    def _validate_manual_fields(self, manual_assessment, manual_notes, source_pair, rank):
        if not manual_assessment:
            return
        if manual_assessment not in VALID_MANUAL_ASSESSMENTS:
            allowed = ", ".join(sorted(VALID_MANUAL_ASSESSMENTS))
            raise ValueError(
                f"Invalid manual_assessment for {source_pair} rank {rank}: {manual_assessment!r}. "
                f"Allowed values: {allowed}"
            )
        if manual_assessment == "possible same sort/block" and not manual_notes:
            raise ValueError(
                f"Manual notes are required for {source_pair} rank {rank} when "
                "manual_assessment is 'possible same sort/block'"
            )

    def _load_greenman_status(self):
        greenman_json = self.output_dir / "greenman_review.json"
        if not greenman_json.exists():
            return None
        return json.loads(greenman_json.read_text())

    def render(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sheets_dir = self.output_dir / "sheets"
        grouped_instances = self.gather_instances()
        pair_info_list = self.load_high_similarity_pairs()

        ledger_rows = []
        for pair_info in pair_info_list:
            candidates = self.build_pair_candidates(grouped_instances, pair_info)
            slug = f"{pair_info['source_left']}__{pair_info['source_right']}".replace("/", "_")
            sheet_path = sheets_dir / f"{slug}.png"
            self.build_contact_sheet(pair_info, candidates, sheet_path)
            sheet_rel = str(sheet_path.resolve().relative_to(ROOT))

            for rank, item in enumerate(candidates, start=1):
                ledger_rows.append(
                    {
                        "source_pair": pair_info["pair"],
                        "pair_average_score": f"{pair_info['average_score']:.3f}",
                        "rank": rank,
                        "character": item["character"],
                        "combined_score": f"{item['combined_score']:.3f}",
                        "assistant_preliminary_assessment": item["assistant_preliminary_assessment"],
                        "manual_assessment": "",
                        "manual_notes": "",
                        "left_source": item["left"]["source_name"],
                        "left_page": item["left"]["page_number"],
                        "left_bbox": f"{item['left']['x']},{item['left']['y']},{item['left']['width']},{item['left']['height']}",
                        "left_crop": item["left"]["crop_path"],
                        "right_source": item["right"]["source_name"],
                        "right_page": item["right"]["page_number"],
                        "right_bbox": f"{item['right']['x']},{item['right']['y']},{item['right']['width']},{item['right']['height']}",
                        "right_crop": item["right"]["crop_path"],
                        "sheet_path": sheet_rel,
                    }
                )

        ledger_rows = self._merge_annotations(ledger_rows)
        self._write_csv(ledger_rows, self.output_dir / "manual_review_ledger.csv")
        (self.output_dir / "manual_review_ledger.json").write_text(json.dumps(ledger_rows, indent=2))
        self._write_markdown(pair_info_list, ledger_rows, self.output_dir / "manual_review_ledger.md")
        self._write_summary(ledger_rows, self.output_dir / "manual_review_summary.md")

    def _write_csv(self, rows, path: Path):
        fieldnames = [
            "source_pair",
            "pair_average_score",
            "rank",
            "character",
            "combined_score",
            "assistant_preliminary_assessment",
            "manual_assessment",
            "manual_notes",
            "left_source",
            "left_page",
            "left_bbox",
            "left_crop",
            "right_source",
            "right_page",
            "right_bbox",
            "right_crop",
            "sheet_path",
        ]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_markdown(self, pair_info_list, rows, path: Path):
        by_pair = defaultdict(list)
        for row in rows:
            by_pair[row["source_pair"]].append(row)

        lines = [
            "# German/Kempten Manual Review Ledger",
            "",
            "This ledger is a pair-level review artifact generated from the canonical German corpus.",
            "The `assistant_preliminary_assessment` column is only a sorting aid. Final judgments belong in `manual_assessment`.",
            "",
        ]
        for pair_info in pair_info_list:
            pair = pair_info["pair"]
            lines.append(f"## {pair}")
            lines.append("")
            lines.append(f"- Pair average score: `{pair_info['average_score']:.3f}`")
            lines.append(f"- Characters compared in aggregate report: `{pair_info['character_count']}`")
            first = by_pair.get(pair, [])
            if first:
                lines.append(f"- Contact sheet: `{first[0]['sheet_path']}`")
            lines.append("")
            lines.append("| Rank | Char | Score | Preliminary | Manual | Notes | Left | Right |")
            lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
            for row in first:
                lines.append(
                    f"| {row['rank']} | `{row['character']}` | {row['combined_score']} | "
                    f"{row['assistant_preliminary_assessment']} | "
                    f"{row['manual_assessment'] or '-'} | "
                    f"{row['manual_notes'] or '-'} | "
                    f"p.{row['left_page']} `{row['left_bbox']}` | "
                    f"p.{row['right_page']} `{row['right_bbox']}` |"
                )
            lines.append("")
        path.write_text("\n".join(lines))

    def _write_summary(self, rows, path: Path):
        by_pair = defaultdict(list)
        overall_counts = Counter()
        for row in rows:
            by_pair[row["source_pair"]].append(row)
            if row["manual_assessment"]:
                overall_counts[row["manual_assessment"]] += 1

        reviewed_rows = [row for row in rows if row["manual_assessment"]]
        same_design_rows = sorted(
            (row for row in reviewed_rows if row["manual_assessment"] == "same design"),
            key=lambda row: float(row["combined_score"]),
            reverse=True,
        )
        possible_rows = sorted(
            (row for row in reviewed_rows if row["manual_assessment"] == "possible same sort/block"),
            key=lambda row: float(row["combined_score"]),
            reverse=True,
        )
        greenman_status = self._load_greenman_status()

        lines = [
            "# German/Kempten Manual Review Summary",
            "",
            f"- Reviewed rows: `{len(reviewed_rows)}` / `{len(rows)}`",
            f"- `same design`: `{overall_counts['same design']}`",
            f"- `possible same sort/block`: `{overall_counts['possible same sort/block']}`",
            f"- `inconclusive`: `{overall_counts['inconclusive']}`",
            "",
            "## Counts by source pair",
            "",
            "| Pair | same design | possible same sort/block | inconclusive |",
            "| --- | ---: | ---: | ---: |",
        ]
        for pair in sorted(by_pair):
            pair_counts = Counter(row["manual_assessment"] for row in by_pair[pair])
            lines.append(
                f"| `{pair}` | {pair_counts['same design']} | "
                f"{pair_counts['possible same sort/block']} | {pair_counts['inconclusive']} |"
            )

        lines.extend(
            [
                "",
                "## Strongest `same design` examples",
                "",
            ]
        )
        if same_design_rows:
            for row in same_design_rows[:8]:
                lines.append(
                    f"- `{row['source_pair']}` rank `{row['rank']}` char `{row['character']}` "
                    f"score `{row['combined_score']}`: page `{row['left_page']}` vs `{row['right_page']}`"
                )
        else:
            lines.append("- None recorded.")

        lines.extend(
            [
                "",
                "## `possible same sort/block` examples",
                "",
            ]
        )
        if possible_rows:
            for row in possible_rows:
                lines.append(
                    f"- `{row['source_pair']}` rank `{row['rank']}` char `{row['character']}` "
                    f"score `{row['combined_score']}`: {row['manual_notes']}"
                )
        else:
            lines.append("- None recorded.")

        lines.extend(["", "## Greenman review status", ""])
        if greenman_status:
            verified = greenman_status["verified_match"]
            control = greenman_status["below_threshold_control"]
            lines.extend(
                [
                    f"- Verified candidate remains `{verified['source']}` page `{verified['page']}` with score `{verified['aggregate_score']:.3f}`.",
                    f"- Below-threshold control remains `{control['source']}` page `{control['page']}` with score `{control['aggregate_score']:.3f}`.",
                    f"- Interpretation: {greenman_status['interpretation']}",
                ]
            )
        else:
            lines.append("- `greenman_review.json` not found; regenerate the Greenman review sheet before using this summary externally.")

        path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Generate a manual review ledger for German character matches")
    parser.add_argument("--db-path", default="data/forensic.db", help="SQLite database path")
    parser.add_argument("--sort-comparison", default="reports/character_sort_match/sort_comparison.json",
                        help="Aggregate sort comparison JSON path")
    parser.add_argument("--output-dir", default="reports/manual_review",
                        help="Output directory for ledger artifacts")
    parser.add_argument("--per-pair", type=int, default=10,
                        help="Number of top pair-level items to select per high-similarity source pair")
    parser.add_argument("--annotations",
                        help="CSV file with manual_assessment/manual_notes keyed by source_pair and rank")
    args = parser.parse_args()

    builder = ManualReviewLedgerBuilder(
        db_path=args.db_path,
        sort_comparison_path=args.sort_comparison,
        output_dir=args.output_dir,
        per_pair=args.per_pair,
        annotations_path=args.annotations,
    )
    builder.render()


if __name__ == "__main__":
    main()
