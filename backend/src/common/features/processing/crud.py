from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from uuid import UUID
import uuid
from datetime import datetime

from src.common.database.models.processing_run_table import ProcessingRun
from src.common.database.models.detection_table import Detection
from src.common.database.models.artifact_table import Artifact
from src.common.features.processing.schemas import (
    ProcessingRunCreate,
    ProcessingRunPublic,
    ProcessingRunsPublic,
    DetectionPublic,
    DetectionsPublic,
    ArtifactPublic,
    ArtifactsPublic,
    ProcessingProgress,
)


def create_processing_run(
    db: Session, 
    project_id: UUID, 
    homography_session_id: Optional[UUID] = None, 
    params: Optional[Dict[str, Any]] = None
) -> ProcessingRun:
    """Create a new processing run."""
    if params is None:
        params = {}
    
    db_run = ProcessingRun(
        project_id=project_id,
        homography_session_id=homography_session_id,
        params=params,
        status="pending",
        progress={},
        started_at=datetime.utcnow()
    )
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run


def get_processing_run(db: Session, run_id: UUID) -> Optional[ProcessingRun]:
    """Get a processing run by ID."""
    return db.query(ProcessingRun).filter(ProcessingRun.id == run_id).first()


def list_processing_runs(db: Session, project_id: UUID) -> List[ProcessingRun]:
    """List all processing runs for a project."""
    return (
        db.query(ProcessingRun)
        .filter(ProcessingRun.project_id == project_id)
        .order_by(ProcessingRun.started_at.desc())
        .all()
    )


def update_run_status(
    db: Session, 
    run_id: UUID, 
    status: str, 
    progress: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> Optional[ProcessingRun]:
    """Update processing run status and optionally progress/error."""
    db_run = get_processing_run(db, run_id)
    if not db_run:
        return None
    
    db_run.status = status
    if progress is not None:
        db_run.progress = progress
    if error_message is not None:
        db_run.error_message = error_message
    if status in ["completed", "failed"]:
        db_run.finished_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_run)
    return db_run


def update_run_progress(
    db: Session, 
    run_id: UUID, 
    stage: str, 
    percent: int, 
    message: str
) -> Optional[ProcessingRun]:
    """Update processing run progress."""
    progress = {
        "stage": stage,
        "percent": percent,
        "message": message
    }
    return update_run_status(db, run_id, "running", progress)


def bulk_insert_detections(db: Session, run_id: UUID, detections_list: List[Dict[str, Any]]) -> int:
    """Efficiently bulk insert detections."""
    if not detections_list:
        return 0
    
    # Prepare detection objects
    detection_objects = []
    for detection_data in detections_list:
        detection_obj = Detection(
            project_id=detection_data["project_id"],
            run_id=run_id,
            frame_idx=detection_data["frame_idx"],
            t_ms=detection_data["t_ms"],
            track_id=detection_data.get("track_id"),
            cls=detection_data["cls"],
            conf=detection_data.get("conf"),
            x=detection_data["x"],
            y=detection_data["y"],
            w=detection_data["w"],
            h=detection_data["h"],
            wx=detection_data.get("wx"),
            wy=detection_data.get("wy"),
            extra=detection_data.get("extra", {})
        )
        detection_objects.append(detection_obj)
    
    # Bulk insert
    db.bulk_save_objects(detection_objects)
    db.commit()
    return len(detection_objects)


def get_detections_by_run(
    db: Session, 
    run_id: UUID, 
    skip: int = 0, 
    limit: int = 100
) -> List[Detection]:
    """Get paginated detections for a run."""
    return (
        db.query(Detection)
        .filter(Detection.run_id == run_id)
        .order_by(Detection.frame_idx, Detection.t_ms)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_detections_by_frame(db: Session, run_id: UUID, frame_idx: int) -> List[Detection]:
    """Get detections for a specific frame."""
    return (
        db.query(Detection)
        .filter(and_(Detection.run_id == run_id, Detection.frame_idx == frame_idx))
        .all()
    )


def create_artifact(
    db: Session, 
    project_id: UUID, 
    run_id: Optional[UUID], 
    kind: str, 
    uri: str, 
    meta: Optional[Dict[str, Any]] = None
) -> Artifact:
    """Create an artifact record."""
    if meta is None:
        meta = {}
    
    db_artifact = Artifact(
        project_id=project_id,
        run_id=run_id,
        kind=kind,
        uri=uri,
        meta=meta,
        created_at=datetime.utcnow()
    )
    db.add(db_artifact)
    db.commit()
    db.refresh(db_artifact)
    return db_artifact


def list_artifacts(db: Session, run_id: UUID) -> List[Artifact]:
    """List artifacts for a run."""
    return (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id)
        .order_by(Artifact.created_at.desc())
        .all()
    )


def get_artifact(db: Session, artifact_id: UUID) -> Optional[Artifact]:
    """Get an artifact by ID."""
    return db.query(Artifact).filter(Artifact.id == artifact_id).first()
