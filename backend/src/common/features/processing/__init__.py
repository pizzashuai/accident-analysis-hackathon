# Processing feature module

from .schemas import (
    ProcessingRunCreate,
    ProcessingRunPublic,
    ProcessingRunsPublic,
    DetectionPublic,
    DetectionsPublic,
    ArtifactPublic,
    ArtifactsPublic,
    ProcessingProgress,
)

from .crud import (
    create_processing_run,
    get_processing_run,
    list_processing_runs,
    update_run_status,
    update_run_progress,
    bulk_insert_detections,
    get_detections_by_run,
    get_detections_by_frame,
    create_artifact,
    list_artifacts,
    get_artifact,
)

__all__ = [
    # Schemas
    "ProcessingRunCreate",
    "ProcessingRunPublic", 
    "ProcessingRunsPublic",
    "DetectionPublic",
    "DetectionsPublic",
    "ArtifactPublic",
    "ArtifactsPublic",
    "ProcessingProgress",
    # CRUD functions
    "create_processing_run",
    "get_processing_run",
    "list_processing_runs",
    "update_run_status",
    "update_run_progress",
    "bulk_insert_detections",
    "get_detections_by_run",
    "get_detections_by_frame",
    "create_artifact",
    "list_artifacts",
    "get_artifact",
]
