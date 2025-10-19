import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.api.deps import CurrentUser, get_db
from src.common.database.models.project_table import Project
from src.common.database.models.artifact_table import Artifact
from src.common.features.storage import (
    download_file_from_s3,
    generate_presigned_url,
    parse_s3_uri,
    upload_file_to_s3,
)
from src.common.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video-annotation", tags=["video-annotation"])


class FilterDetectionsRequest(BaseModel):
    """Request model for filtering detections by track IDs."""
    track_ids: List[int]
    artifact_id: str
    filename: Optional[str] = None


class FilterDetectionsResponse(BaseModel):
    """Response model for filtered detections."""
    artifact_id: str
    filename: str
    track_count: int
    detection_count: int
    message: str


@router.post("/filter-detections", response_model=FilterDetectionsResponse)
def filter_detections_by_tracks(
    *,
    request: FilterDetectionsRequest,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> FilterDetectionsResponse:
    """
    Filter detection JSONL file by track IDs and save as new artifact.
    
    This endpoint:
    1. Downloads the original JSONL artifact from S3
    2. Filters detections to only include specified track IDs
    3. Creates a new filtered JSONL file
    4. Uploads the filtered file to S3 as a new artifact
    5. Returns metadata about the filtered file
    """
    try:
        artifact_uuid = uuid.UUID(request.artifact_id)
        
        # Get the original artifact
        artifact = session.get(Artifact, artifact_uuid)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Validate ownership through project
        project = session.get(Project, artifact.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Validate artifact is JSONL detections
        if artifact.kind != "jsonl_detections":
            raise HTTPException(status_code=400, detail="Artifact must be a JSONL detections file")
        
        # Download original JSONL file from S3
        bucket, key = parse_s3_uri(artifact.uri)
        presigned_url = generate_presigned_url(bucket, key)
        
        import requests
        response = requests.get(presigned_url)
        response.raise_for_status()
        
        # Parse JSONL content
        jsonl_content = response.text
        lines = jsonl_content.strip().split('\n')
        
        # Filter detections by track IDs
        filtered_lines = []
        track_ids_set = set(request.track_ids)
        
        for line in lines:
            if not line.strip():
                continue
                
            try:
                import json
                detection = json.loads(line)
                
                # Check if detection has a track_id that matches our filter
                track_id = detection.get('track_id')
                if track_id is not None and track_id in track_ids_set:
                    filtered_lines.append(line)
                    
            except json.JSONDecodeError:
                # Skip malformed lines
                logger.warning(f"Skipping malformed JSON line: {line[:100]}...")
                continue
        
        if not filtered_lines:
            raise HTTPException(
                status_code=400, 
                detail=f"No detections found for track IDs: {request.track_ids}"
            )
        
        # Create filtered JSONL content
        filtered_jsonl_content = '\n'.join(filtered_lines) + '\n'
        
        # Generate filename for filtered file
        original_filename = request.filename or f"filtered_detections_tracks_{'_'.join(map(str, request.track_ids))}.jsonl"
        if not original_filename.endswith('.jsonl'):
            original_filename += '.jsonl'
        
        # Replace the existing artifact with filtered content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            temp_file.write(filtered_jsonl_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Upload filtered content to the same S3 location (replacing the original)
            with open(temp_file_path, 'rb') as f:
                upload_file_to_s3(
                    f,
                    settings.AWS_S3_BUCKET,
                    key  # Use the same key as the original artifact
                )
            
            # Update the existing artifact record
            artifact.bytes = len(filtered_jsonl_content.encode('utf-8'))
            artifact.meta = {
                **artifact.meta,  # Preserve existing metadata
                "filtered_track_ids": request.track_ids,
                "filter_type": "track_id_filter",
                "filtered_at": datetime.utcnow().isoformat(),
                "original_detection_count": len(lines),
                "filtered_detection_count": len(filtered_lines),
            }
            session.commit()
            
            return FilterDetectionsResponse(
                artifact_id=str(artifact.id),
                filename=original_filename,
                track_count=len(request.track_ids),
                detection_count=len(filtered_lines),
                message=f"Successfully filtered {len(filtered_lines)} detections for {len(request.track_ids)} track(s) and replaced the original file"
            )
            
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error filtering detections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/artifacts/{artifact_id}/download")
def get_filtered_artifact_download_url(
    *,
    artifact_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> dict:
    """Get presigned download URL for a filtered artifact."""
    try:
        artifact_uuid = uuid.UUID(artifact_id)
        
        artifact = session.get(Artifact, artifact_uuid)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Validate ownership through project
        project = session.get(Project, artifact.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Generate presigned URL
        bucket, key = parse_s3_uri(artifact.uri)
        presigned_url = generate_presigned_url(bucket, key)
        
        return {"url": presigned_url}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting filtered artifact download URL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
