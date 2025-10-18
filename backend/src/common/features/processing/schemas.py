from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, UUID4

# Processing Run schemas
class ProcessingRunCreate(BaseModel):
    params: Optional[Dict[str, Any]] = {}

class ProcessingRunPublic(BaseModel):
    model_config = {"from_attributes": True}
    
    id: UUID4
    project_id: UUID4
    homography_session_id: Optional[UUID4]
    params: Dict[str, Any]
    status: str
    progress: Dict[str, Any]
    started_at: datetime
    finished_at: Optional[datetime]
    error_message: Optional[str]

class ProcessingRunsPublic(BaseModel):
    model_config = {"from_attributes": True}
    
    data: List[ProcessingRunPublic]
    count: int

# Detection schemas
class DetectionPublic(BaseModel):
    model_config = {"from_attributes": True}
    
    id: int
    frame_idx: int
    t_ms: int
    track_id: Optional[int]
    cls: str
    conf: Optional[float]
    x: float
    y: float
    w: float
    h: float
    wx: Optional[float]
    wy: Optional[float]
    extra: Dict[str, Any]

class DetectionsPublic(BaseModel):
    model_config = {"from_attributes": True}
    
    data: List[DetectionPublic]
    count: int

# Artifact schemas
class ArtifactPublic(BaseModel):
    model_config = {"from_attributes": True}
    
    id: UUID4
    kind: str
    uri: str
    meta: Dict[str, Any]
    created_at: datetime

class ArtifactsPublic(BaseModel):
    model_config = {"from_attributes": True}
    
    data: List[ArtifactPublic]
    count: int

# Progress update schema
class ProcessingProgress(BaseModel):
    stage: str
    percent: int
    message: str
