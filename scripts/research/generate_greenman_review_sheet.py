#!/usr/bin/env python3
"""
Generate a Greenman review artifact with one verified match and one below-threshold control.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "scripts"))

from app.services.block_fingerprinter import BlockFingerprinter
from scan_greenman_all import GreenmanScanner


DISPLAY_NAMES = {
    "bsb_munich_10057380": "BSB Munich",
    "gdz_goettingen_ppn777246686": "GDZ Goettingen",
    "hab_wolfenbuettel_178_1_theol_1s": "HAB Wolfenbuettel",
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_jsonl(path: Path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _collect_best_control(scanner: GreenmanScanner, ornaments_root: Path, verified_match_id: str):
    best = None
    for candidates_path in sorted(ornaments_root.glob("*/candidates.jsonl")):
        source_name = candidates_path.parent.name
        image_dir = candidates_path.parent / "images"
        for candidate in _load_jsonl(candidates_path):
            if not scanner._candidate_area_is_viable(candidate.get("area", 0)):
                continue
            match_id = f"{source_name}_p{candidate.get('page_number', 0)}_{candidate.get('x', 0)}_{candidate.get('y', 0)}"
            if match_id == verified_match_id:
                continue

            crop_path = image_dir / candidate["filename"]
            crop = cv2.imread(str(crop_path))
            if crop is None:
                continue

            fingerprint = scanner.fingerprinter.fingerprint(crop, block_type="device")
            scores = scanner.fingerprinter.compare(scanner.reference_fp, fingerprint)
            aggregate = float(scores.get("aggregate", 0.0))

            if aggregate >= scanner.FINGERPRINT_THRESHOLD:
                continue

            item = {
                "source": source_name,
                "page": int(candidate.get("page_number", 0)),
                "bbox": {
                    "x": int(candidate.get("x", 0)),
                    "y": int(candidate.get("y", 0)),
                    "w": int(candidate.get("w", 0)),
                    "h": int(candidate.get("h", 0)),
                },
                "candidate_filename": candidate["filename"],
                "candidate_crop_path": str(crop_path.relative_to(ROOT)),
                "fingerprint_scores": scores,
                "aggregate_score": aggregate,
            }
            if not best or aggregate > best["aggregate_score"]:
                best = item

    if best:
        control_crop = cv2.imread(str(ROOT / best["candidate_crop_path"]))
        sift_score, _ = scanner._sift_match(control_crop)
        best["sift_matches"] = int(sift_score)
        best["passes_threshold"] = bool(
            sift_score >= scanner.MIN_SIFT_MATCHES and
            best["aggregate_score"] >= scanner.FINGERPRINT_THRESHOLD
        )
    return best


def _resize(image, max_w, max_h):
    if image is None:
        return np.full((max_h, max_w, 3), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return resized


def _paste_centered(canvas, image, x, y, box_w, box_h):
    h, w = image.shape[:2]
    ox = x + (box_w - w) // 2
    oy = y + (box_h - h) // 2
    canvas[oy:oy + h, ox:ox + w] = image
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (180, 180, 180), 1)


def build_sheet(reference_path: Path, verified_crop_path: Path, control_crop_path: Path, output_path: Path):
    ref = cv2.imread(str(reference_path))
    verified = cv2.imread(str(verified_crop_path))
    control = cv2.imread(str(control_crop_path))

    canvas = np.full((560, 1380, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, "Greenman Manual Review", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(
        canvas,
        "Reference, verified candidate, and strongest below-threshold control",
        (28, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
    )

    columns = [
        ("Reference", ref, 28),
        ("Verified match", verified, 472),
        ("Below-threshold control", control, 916),
    ]
    for label, image, x in columns:
        cv2.putText(canvas, label, (x, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2)
        _paste_centered(canvas, _resize(image, 380, 300), x, 150, 380, 300)

    cv2.imwrite(str(output_path), canvas)


def main():
    parser = argparse.ArgumentParser(description="Generate Greenman review sheet and control note")
    parser.add_argument("--reference", default="data/greenman_reference.jpg", help="Reference crop path")
    parser.add_argument("--matches-path", default="reports/greenman_scan/matches.json", help="Verified matches JSON")
    parser.add_argument("--ornaments-root", default="reports/ornaments", help="Cached ornament candidate root")
    parser.add_argument("--output-dir", default="reports/manual_review", help="Output directory")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matches = json.loads((ROOT / args.matches_path).read_text())
    if not matches:
        raise RuntimeError("No verified Greenman matches found.")

    verified = matches[0]
    verified_crop_path = ROOT / "reports" / "greenman_scan" / f"crop_{verified['match_id']}.jpg"

    scanner = GreenmanScanner(str(ROOT / args.reference))
    control = _collect_best_control(
        scanner,
        ROOT / args.ornaments_root,
        verified["match_id"],
    )
    if control is None:
        raise RuntimeError("No below-threshold Greenman control candidate could be identified.")

    sheet_path = output_dir / "greenman_review_sheet.png"
    build_sheet(
        ROOT / args.reference,
        verified_crop_path,
        ROOT / control["candidate_crop_path"],
        sheet_path,
    )

    review = {
        "verified_match": verified,
        "below_threshold_control": control,
        "sheet_path": str(sheet_path.relative_to(ROOT)),
        "interpretation": (
            "The verified candidate remains source-specific. The control candidate is visually similar enough "
            "to justify review, but it stays below the match threshold and should be treated as a negative control."
        ),
    }
    (output_dir / "greenman_review.json").write_text(
        json.dumps(review, indent=2, default=_json_default)
    )

    lines = [
        "# Greenman Review Note",
        "",
        f"- Review sheet: `{sheet_path.relative_to(ROOT)}`",
        f"- Verified match: `{DISPLAY_NAMES.get(verified['source'], verified['source'])}` page `{verified['page']}` "
        f"bbox `{verified['bbox']['x']},{verified['bbox']['y']},{verified['bbox']['w']},{verified['bbox']['h']}` "
        f"score `{verified['aggregate_score']:.3f}`",
        f"- Below-threshold control: `{DISPLAY_NAMES.get(control['source'], control['source'])}` page `{control['page']}` "
        f"bbox `{control['bbox']['x']},{control['bbox']['y']},{control['bbox']['w']},{control['bbox']['h']}` "
        f"score `{control['aggregate_score']:.3f}` with `{control['sift_matches']}` SIFT matches",
        "",
        "This sheet is meant to accompany the manual review ledger. The control candidate remains below the current threshold and should not be promoted into a match claim.",
    ]
    (output_dir / "greenman_review.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
