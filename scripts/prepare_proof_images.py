#!/usr/bin/env python3
import json
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from app.services.forensic_report_visuals import generate_report_charts

proof_dir = base_dir / "reports" / "proof_images"
greenman_dir = base_dir / "reports" / "greenman_scan"

SOURCE_PAGE_SELECTIONS = {
    "bsb_munich": base_dir / "data/sources/bsb_10057380/000_0009.jpg",
    "gdz_gottingen": base_dir / "data/sources/gdz_ppn777246686/009_1.jpg",
    "hab_wolfenbuttel": base_dir / "data/sources/hab_178_1_theol_1s/00005.jpg",
    "google_books": base_dir / "data/sources/google_books_tractatus/page-07.jpg",
}

SOURCE_IMAGE_DIRS = {
    "bsb_munich_10057380": base_dir / "data/sources/bsb_10057380",
    "gdz_goettingen_ppn777246686": base_dir / "data/sources/gdz_ppn777246686",
    "hab_wolfenbuettel_178_1_theol_1s": base_dir / "data/sources/hab_178_1_theol_1s",
    "google_books_tractatus_brevis": base_dir / "data/sources/google_books_tractatus",
}

DISPLAY_NAMES = {
    "bsb_munich_10057380": "BSB Munich",
    "gdz_goettingen_ppn777246686": "GDZ Gottingen",
    "hab_wolfenbuettel_178_1_theol_1s": "HAB Wolfenbuttel",
    "google_books_tractatus_brevis": "Google Books (Tractatus)",
}

REVIEWED_PAIR_SELECTIONS = [
    {
        "pair": "bsb_munich_10057380 vs gdz_goettingen_ppn777246686",
        "rank": 4,
        "output_name": "reviewed_pair_possible_d.png",
        "caption": "Possible same sort/block exemplar: 'd' (BSB p.286 vs GDZ p.33).",
    },
    {
        "pair": "bsb_munich_10057380 vs gdz_goettingen_ppn777246686",
        "rank": 6,
        "output_name": "reviewed_pair_possible_b.png",
        "caption": "Possible same sort/block exemplar: 'b' (BSB p.559 vs GDZ p.50).",
    },
    {
        "pair": "gdz_goettingen_ppn777246686 vs hab_wolfenbuettel_178_1_theol_1s",
        "rank": 1,
        "output_name": "reviewed_pair_same_design_s.png",
        "caption": "Same-design control exemplar: 'S' (GDZ p.58 vs HAB p.153).",
    },
]


def _clear_proof_dir():
    proof_dir.mkdir(parents=True, exist_ok=True)
    for file_path in proof_dir.glob("*"):
        file_path.unlink()


def _copy_selected_source_pages():
    for key, source_path in SOURCE_PAGE_SELECTIONS.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing selected source page: {source_path}")
        destination = proof_dir / f"source_page_{key}.jpg"
        shutil.copyfile(source_path, destination)
        print(f"Copied selected source page for {key}: {source_path.name}")


def _draw_greenman_context():
    matches_path = greenman_dir / "matches.json"
    matches = json.loads(matches_path.read_text())
    if not matches:
        raise RuntimeError("No Greenman matches found to build proof images.")

    verified = matches[0]
    page_path = SOURCE_IMAGE_DIRS[verified["source"]] / verified["filename"]
    crop_path = greenman_dir / f"crop_{verified['match_id']}.jpg"
    viz_path = greenman_dir / f"match_viz_{verified['match_id']}.jpg"

    page_img = cv2.imread(str(page_path))
    if page_img is None:
        raise RuntimeError(f"Could not load Greenman page image: {page_path}")

    x = int(verified["bbox"]["x"])
    y = int(verified["bbox"]["y"])
    w = int(verified["bbox"]["w"])
    h = int(verified["bbox"]["h"])

    annotated = page_img.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 140, 255), 6)
    label = "Verified GDZ candidate"
    label_y = max(36, y - 16)
    cv2.rectangle(
        annotated,
        (x, max(0, label_y - 28)),
        (x + 320, label_y + 8),
        (255, 255, 255),
        thickness=-1,
    )
    cv2.putText(
        annotated,
        label,
        (x + 10, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(proof_dir / "greenman_context_gdz.jpg"), annotated)
    shutil.copyfile(crop_path, proof_dir / "greenman_crop_gdz.jpg")
    print(f"Created Greenman context image from {page_path.name}")

    if viz_path.exists():
        shutil.copyfile(viz_path, proof_dir / f"sift_match_{verified['match_id']}.jpg")
        print("Copied Greenman SIFT visualization for supplemental use")


def _resize_with_padding(img, box_w, box_h):
    h, w = img.shape[:2]
    scale = min(box_w / max(w, 1), box_h / max(h, 1))
    resized = cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    canvas = np.full((box_h, box_w, 3), 255, dtype=np.uint8)
    y = (box_h - resized.shape[0]) // 2
    x = (box_w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _panel_title(row):
    assessment = row["manual_assessment"].replace("_", " ")
    char = row["character"]
    score = float(row["combined_score"])
    return f"{assessment.title()} | '{char}' | score {score:.3f}"


def _panel_note(row):
    if row["manual_notes"]:
        return row["manual_notes"]
    if row["manual_assessment"] == "possible same sort/block":
        return "Manual review marked this pair as provisional, not dispositive."
    return "Manual review marked this pair as same design, not same object."


def _build_reviewed_pair_panel(row, output_name):
    left_path = base_dir / "reports" / "crops" / row["left_crop"]
    right_path = base_dir / "reports" / "crops" / row["right_crop"]
    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    if left_img is None or right_img is None:
        raise RuntimeError(f"Could not load reviewed pair crops: {left_path} / {right_path}")

    canvas = np.full((320, 980, 3), 248, dtype=np.uint8)
    cv2.putText(
        canvas,
        _panel_title(row),
        (28, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )

    note = _panel_note(row)
    cv2.putText(
        canvas,
        note[:120],
        (28, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    left_box = _resize_with_padding(left_img, 320, 180)
    right_box = _resize_with_padding(right_img, 320, 180)

    canvas[92:272, 60:380] = left_box
    canvas[92:272, 600:920] = right_box

    cv2.rectangle(canvas, (60, 92), (380, 272), (185, 185, 185), 1)
    cv2.rectangle(canvas, (600, 92), (920, 272), (185, 185, 185), 1)

    left_label = f"{DISPLAY_NAMES[row['left_source']]} p.{row['left_page']} bbox {row['left_bbox']}"
    right_label = f"{DISPLAY_NAMES[row['right_source']]} p.{row['right_page']} bbox {row['right_bbox']}"

    cv2.putText(
        canvas,
        left_label[:70],
        (60, 292),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        right_label[:70],
        (600, 292),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )

    output_path = proof_dir / output_name
    cv2.imwrite(str(output_path), canvas)
    return output_path


def _build_reviewed_pairs():
    ledger_rows = json.loads((base_dir / "reports" / "manual_review" / "manual_review_ledger.json").read_text())
    lookup = {
        (row["source_pair"], int(row["rank"])): row
        for row in ledger_rows
    }

    manifest = []
    for item in REVIEWED_PAIR_SELECTIONS:
        row = lookup.get((item["pair"], item["rank"]))
        if row is None:
            raise RuntimeError(f"Missing reviewed pair row for {item['pair']} rank {item['rank']}")
        output_path = _build_reviewed_pair_panel(row, item["output_name"])
        manifest.append(
            {
                "file": output_path.name,
                "caption": item["caption"],
                "assessment": row["manual_assessment"],
                "character": row["character"],
                "score": float(row["combined_score"]),
                "left_source": row["left_source"],
                "right_source": row["right_source"],
                "left_page": int(row["left_page"]),
                "right_page": int(row["right_page"]),
            }
        )

    (proof_dir / "reviewed_pairs_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Created reviewed pair proof images and manifest")


def _build_report_charts():
    sort_results = json.loads((base_dir / "reports" / "character_sort_match" / "sort_comparison.json").read_text())
    stats_results = json.loads((base_dir / "reports" / "statistical_analysis" / "formal_stats.json").read_text())
    manual_review_rows = json.loads((base_dir / "reports" / "manual_review" / "manual_review_ledger.json").read_text())
    generate_report_charts(sort_results, stats_results, manual_review_rows, proof_dir)
    print("Created chart suite for forensic PDF")


def prepare_proof_images():
    print(f"Preparing proof images in {proof_dir}")
    _clear_proof_dir()
    _copy_selected_source_pages()
    _draw_greenman_context()
    _build_reviewed_pairs()
    _build_report_charts()
    print("Proof images prepared successfully.")


if __name__ == "__main__":
    prepare_proof_images()
