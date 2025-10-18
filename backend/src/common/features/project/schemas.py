import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class ProjectBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)


class MediaAssetPublic(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    kind: str
    uri: str
    bytes: int | None
    meta: dict[str, Any]
    created_at: datetime
    is_processing: bool = False
    processing_error: str | None = None

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def presigned_url(self) -> str | None:
        """Generate presigned URL for S3 objects."""
        if not self.uri.startswith('s3://'):
            return None
        
        try:
            from src.common.features.storage.s3_service import generate_presigned_url, parse_s3_uri
            bucket, key = parse_s3_uri(self.uri)
            return generate_presigned_url(bucket, key)
        except Exception:
            # Return None if presigned URL generation fails
            return None


class ProjectLocationPublic(BaseModel):
    project_id: uuid.UUID
    addr_line: str | None
    lat: float | None
    lon: float | None
    source: str

    model_config = {"from_attributes": True}


class ProjectLocationCreate(BaseModel):
    addr_line: str | None = Field(None, max_length=500)
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    source: str = Field(default="user", max_length=50)


class ProjectPublic(ProjectBase):
    id: uuid.UUID
    user_id: uuid.UUID
    video_id: uuid.UUID | None
    created_at: datetime
    video: MediaAssetPublic | None = None
    location: ProjectLocationPublic | None = None
    media_assets: list[MediaAssetPublic] = []

    model_config = {"from_attributes": True}


class ProjectsPublic(BaseModel):
    data: list[ProjectPublic]
    count: int


class Message(BaseModel):
    message: str
