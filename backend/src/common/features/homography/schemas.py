import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class HomographyPairCreate(BaseModel):
    """Schema for creating a homography point pair"""
    image_x_norm: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate (0-1)")
    image_y_norm: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate (0-1)")
    map_lat: float = Field(..., description="Latitude coordinate")
    map_lng: float = Field(..., description="Longitude coordinate")
    order_idx: int = Field(default=0, description="Display order index")


class HomographyPairPublic(BaseModel):
    """Schema for homography point pair response"""
    id: uuid.UUID
    session_id: uuid.UUID
    image_x_norm: float
    image_y_norm: float
    map_lat: float
    map_lng: float
    order_idx: int

    class Config:
        from_attributes = True


class HomographySessionCreate(BaseModel):
    """Schema for creating a homography session"""
    project_id: uuid.UUID
    screenshot_asset_id: Optional[uuid.UUID] = None


class HomographySessionUpdate(BaseModel):
    """Schema for updating a homography session"""
    status: Optional[str] = Field(None, description="Session status: draft, ready_to_solve, solved, error")
    screenshot_asset_id: Optional[uuid.UUID] = None


class HomographyModelPublic(BaseModel):
    """Schema for homography model response"""
    id: uuid.UUID
    session_id: uuid.UUID
    matrix_data: List[List[float]]
    reprojection_error: Optional[float]
    created_at: datetime
    meta: Optional[dict]

    class Config:
        from_attributes = True


class HomographySessionPublic(BaseModel):
    """Schema for homography session response"""
    id: uuid.UUID
    project_id: uuid.UUID
    screenshot_asset_id: Optional[uuid.UUID]
    status: str
    created_at: datetime
    solved_at: Optional[datetime]
    pairs: List[HomographyPairPublic] = []
    model: Optional[HomographyModelPublic] = None

    class Config:
        from_attributes = True


class HomographySolveResponse(BaseModel):
    """Schema for homography solve response"""
    success: bool
    model: Optional[HomographyModelPublic] = None
    error_message: Optional[str] = None


class HomographyExportData(BaseModel):
    """Schema for exporting homography data in process-video compatible format"""
    pairs: List[dict]
    imagesMeta: dict
    mapMeta: dict
