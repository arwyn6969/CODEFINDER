"""Shared geometry analysis for the Legacy Exploratory Lab."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from app.models.database_models import Document, Pattern
from app.services.bardcode_analyzer import BardCodeAnalyzer, Point as BardPoint
from app.services.geometric_analyzer import GeometricAnalyzer, Point as GeometricPoint

MAX_POINTS = 24
MAX_DISTANCE_MEASUREMENTS = 120
MAX_RATIO_MEASUREMENTS = 60
MAX_SACRED_GEOMETRY_POINTS = 12


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _as_point(raw: Any) -> Optional[GeometricPoint]:
    """Coerce a stored coordinate payload into a point."""
    if isinstance(raw, dict):
        if {"x", "y"} <= set(raw):
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw.get("width", 0) or 0)
            height = float(raw.get("height", 0) or 0)
            if width or height:
                return GeometricPoint(x + width / 2.0, y + height / 2.0)
            return GeometricPoint(x, y)
        if {"left", "right", "top", "bottom"} <= set(raw):
            return GeometricPoint(
                (float(raw["left"]) + float(raw["right"])) / 2.0,
                (float(raw["top"]) + float(raw["bottom"])) / 2.0,
            )
        return None

    if isinstance(raw, (list, tuple)):
        if len(raw) == 2:
            return GeometricPoint(float(raw[0]), float(raw[1]))
        if len(raw) == 4:
            x1, y1, x2, y2 = map(float, raw)
            return GeometricPoint((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    return None


def _extract_pattern_points(patterns: List[Pattern]) -> List[GeometricPoint]:
    points: List[GeometricPoint] = []
    seen = set()

    for pattern in patterns:
        raw = getattr(pattern, "coordinates", None)
        if not raw:
            continue

        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            point = _as_point(candidate)
            if point is None:
                continue
            key = (round(point.x, 4), round(point.y, 4))
            if key in seen:
                continue
            seen.add(key)
            points.append(point)

    return points


def _build_ratio_measurements(distance_values: List[float]) -> List[Dict[str, float]]:
    non_zero = sorted({round(value, 6) for value in distance_values if value > 0})
    if not non_zero:
        return []

    base = non_zero[0]
    ratios = []
    for value in non_zero[1 : MAX_RATIO_MEASUREMENTS + 1]:
        ratios.append({"type": "ratio", "value": value / base})
    return ratios


def _trim_points(points: List[GeometricPoint], warnings: List[str]) -> List[GeometricPoint]:
    if len(points) <= MAX_POINTS:
        return points

    warnings.append(
        f"Using the first {MAX_POINTS} unique pattern points out of {len(points)} available for legacy geometry analysis."
    )
    return points[:MAX_POINTS]


def analyze_document_geometry(db: Session, document_id: int) -> Dict[str, Any]:
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise LookupError(f"Document {document_id} not found")

    patterns = db.query(Pattern).filter(Pattern.document_id == document_id).all()
    warnings: List[str] = []
    points = _extract_pattern_points(patterns)

    if len(points) < 3:
        return {
            "document_id": document_id,
            "status": "no_data",
            "warnings": ["Not enough stored pattern coordinates are available for geometry analysis."],
            "total_measurements": 0,
            "angle_measurements": 0,
            "distance_measurements": 0,
            "ratio_measurements": 0,
            "sacred_geometry_patterns": 0,
            "mathematical_constants_found": [],
            "significance_scores": {},
            "potential_coordinates": [],
            "coordinate_pairs": [],
            "historical_sites": [],
        }

    points = _trim_points(points, warnings)
    geometric_analyzer = GeometricAnalyzer()
    bardcode_analyzer = BardCodeAnalyzer()

    angle_measurements = geometric_analyzer.measure_angles(points)
    distance_measurements = []
    for point_a, point_b in combinations(points, 2):
        distance_measurements.append(geometric_analyzer.calculate_distances(point_a, point_b))
        if len(distance_measurements) >= MAX_DISTANCE_MEASUREMENTS:
            warnings.append(
                f"Distance measurements truncated at {MAX_DISTANCE_MEASUREMENTS} pairs to keep legacy geometry analysis responsive."
            )
            break

    distance_values = [measurement.distance for measurement in distance_measurements]
    ratio_measurements = _build_ratio_measurements(distance_values)

    measurement_payloads: List[Dict[str, float]] = [
        {"type": "angle", "value": measurement.angle_degrees}
        for measurement in angle_measurements
    ]
    measurement_payloads.extend(
        {"type": "distance", "value": measurement.distance}
        for measurement in distance_measurements
    )
    measurement_payloads.extend(ratio_measurements)

    raw_values = [payload["value"] for payload in measurement_payloads]
    constants = bardcode_analyzer.detect_mathematical_constants(raw_values)
    coordinate_analysis = bardcode_analyzer.extract_geographic_coordinates_advanced(measurement_payloads)

    sacred_geometry_patterns = 0
    if len(points) <= MAX_SACRED_GEOMETRY_POINTS:
        constructions = bardcode_analyzer.find_sacred_geometry(
            [BardPoint(point.x, point.y) for point in points]
        )
        sacred_geometry_patterns = len(constructions)
    else:
        warnings.append(
            f"Sacred geometry construction search skipped because it becomes too expensive above {MAX_SACRED_GEOMETRY_POINTS} points."
        )

    coordinate_pairs = coordinate_analysis.get("coordinate_pairs", [])
    historical_sites = coordinate_analysis.get("historical_sites", [])

    if not coordinate_pairs:
        warnings.append("No plausible latitude/longitude pairs were derived from the available measurements.")

    status = "ok"
    if warnings and measurement_payloads:
        status = "warning"

    significance_scores = {
        "mean_angle_confidence": round(_mean(m.confidence for m in angle_measurements), 4),
        "mean_angle_significance": round(_mean(m.significance_score for m in angle_measurements), 4),
        "mean_distance_significance": round(
            _mean(m.significance_score for m in distance_measurements), 4
        ),
        "mean_coordinate_confidence": round(
            _mean(pair.get("combined_confidence", 0.0) for pair in coordinate_pairs), 4
        ),
    }

    return {
        "document_id": document_id,
        "status": status,
        "warnings": warnings,
        "total_measurements": len(measurement_payloads),
        "angle_measurements": len(angle_measurements),
        "distance_measurements": len(distance_measurements),
        "ratio_measurements": len(ratio_measurements),
        "sacred_geometry_patterns": sacred_geometry_patterns,
        "mathematical_constants_found": sorted({constant.name for constant in constants}),
        "significance_scores": significance_scores,
        "potential_coordinates": coordinate_analysis.get("potential_coordinates", []),
        "coordinate_pairs": coordinate_pairs,
        "historical_sites": historical_sites,
    }
