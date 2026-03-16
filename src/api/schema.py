"""
Pydantic models for the BrainMetScan API request/response schema.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LesionDetail(BaseModel):
    """Individual lesion measurement."""
    id: int
    volume_voxels: int
    volume_mm3: float
    centroid: List[float] = Field(description="[h, w, d] coordinates")
    confidence: float = Field(ge=0.0, le=1.0)
    max_diameter_mm: float
    bounding_box: Dict[str, List[int]] = Field(description="{'min': [h,w,d], 'max': [h,w,d]}")


class SegmentationResult(BaseModel):
    """Result from a single segmentation prediction."""
    case_id: str
    lesion_count: int
    total_volume_voxels: int = 0
    total_volume_mm3: float = 0.0
    lesions: List[LesionDetail] = []
    model_version: str = "1.23.0"
    processing_time_seconds: float = 0.0
    job_id: str = ""


class PredictionResponse(BaseModel):
    """Response wrapper for prediction endpoints."""
    status: str = "success"
    result: Optional[SegmentationResult] = None
    rag_report: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    gpu_available: bool
    models_loaded: int
    model_names: List[str]


class ModelInfo(BaseModel):
    """Information about a registered model."""
    name: str
    patch_size: int
    threshold: float
    architecture: str
    exists: bool = True


class ComparisonLesionMatch(BaseModel):
    """A matched lesion between two timepoints."""
    baseline_id: int
    followup_id: int
    volume_change_percent: float
    baseline_volume_mm3: float
    followup_volume_mm3: float


class ComparisonResponse(BaseModel):
    """Response for longitudinal comparison endpoint."""
    status: str = "success"
    baseline_case_id: str = ""
    followup_case_id: str = ""
    response_category: str = Field(default="", description="CR, PR, SD, or PD per RECIST 1.1")
    matched_lesions: List[ComparisonLesionMatch] = []
    new_lesions: int = 0
    resolved_lesions: int = 0
    sum_of_diameters_baseline_mm: float = 0.0
    sum_of_diameters_followup_mm: float = 0.0
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction endpoint."""
    status: str = "success"
    total_cases: int = 0
    completed: int = 0
    failed: int = 0
    results: List[PredictionResponse] = []
