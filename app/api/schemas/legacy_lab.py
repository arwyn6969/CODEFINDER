"""Typed schema models for the Legacy Exploratory Lab."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GematriaCipherResponse(BaseModel):
    score: int
    breakdown: List[int]
    significance: Optional[str] = None


class GematriaAnalysisResponse(BaseModel):
    results: Dict[str, GematriaCipherResponse]
    persisted_patterns: int = 0


class TransliterationCandidateResponse(BaseModel):
    hebrew: str
    description: str


class ELSMatchResponse(BaseModel):
    term: str
    skip: int
    start_index: int
    end_index: int
    location: List[int]
    direction: str


class ELSAnalysisResponse(BaseModel):
    total_length: int
    found_count: int
    matches: List[ELSMatchResponse]
    persisted_patterns: int = 0


class ELSGridCellResponse(BaseModel):
    char: str
    index: int
    row: int


class ELSGridDimensionsResponse(BaseModel):
    rows: int
    cols: int
    row_width: int


class ELSViewportResponse(BaseModel):
    start_row: int
    start_col: int


class ELSHighlightResponse(BaseModel):
    index: int
    grid_row: int
    grid_col: int
    visible: bool


class ELSVisualizationResponse(BaseModel):
    grid: List[List[ELSGridCellResponse]]
    dimensions: ELSGridDimensionsResponse
    center_index: int
    skip: int
    viewport: ELSViewportResponse
    highlights: List[ELSHighlightResponse] = Field(default_factory=list)


class CipherSolveResponse(BaseModel):
    original: str
    method: str
    key: Optional[object] = None
    result: str


class PropheticZoneTermResponse(BaseModel):
    name: str
    term: str
    skip: int
    start_index: int


class PropheticZoneResponse(BaseModel):
    center_index: int
    spread: int
    book: str
    position_percentage: float
    terms: List[PropheticZoneTermResponse]
    visualization_svg: Optional[str] = None


class PropheticConvergenceResponse(BaseModel):
    total_zones_found: int
    top_zones: List[PropheticZoneResponse]


class GeometryPotentialCoordinateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    value: float
    method: str
    confidence: float


class GeometryCoordinatePairResponse(BaseModel):
    latitude: float
    longitude: float
    combined_confidence: float
    methods: List[str]


class GeometryHistoricalSiteResponse(BaseModel):
    name: str
    lat: float
    lon: float
    significance: str


class GeometryAccuracyResponse(BaseModel):
    latitude_error: float
    longitude_error: float
    total_error: float


class GeometryHistoricalMatchResponse(BaseModel):
    site: GeometryHistoricalSiteResponse
    detected_coordinates: GeometryCoordinatePairResponse
    accuracy: GeometryAccuracyResponse
    match_confidence: float


class GeometryAnalysisResponse(BaseModel):
    document_id: int
    status: str
    warnings: List[str] = Field(default_factory=list)
    total_measurements: int
    angle_measurements: int
    distance_measurements: int
    ratio_measurements: int = 0
    sacred_geometry_patterns: int
    mathematical_constants_found: List[str] = Field(default_factory=list)
    significance_scores: Dict[str, float] = Field(default_factory=dict)
    potential_coordinates: List[GeometryPotentialCoordinateResponse] = Field(default_factory=list)
    coordinate_pairs: List[GeometryCoordinatePairResponse] = Field(default_factory=list)
    historical_sites: List[GeometryHistoricalMatchResponse] = Field(default_factory=list)
