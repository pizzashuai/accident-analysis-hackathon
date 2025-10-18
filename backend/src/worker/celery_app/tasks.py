import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import requests
from celery import Celery
from sqlalchemy.orm import Session

from src.common.database.db import engine
from src.common.features.storage import (
    extract_first_frame,
    generate_presigned_url,
    parse_s3_uri,
    upload_file_to_s3,
)
from src.common.features.project import create_media_asset, get_project_with_relations
from src.common.database.models.media_asset_table import MediaAsset
from src.common.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
from src.worker.celery_app.app import app as celery_app


def get_db_session():
    """Get database session for worker tasks."""
    with Session(engine) as session:
        yield session


@celery_app.task(bind=True)
def extract_video_frame_task(self, project_id: str, video_asset_id: str):
    """
    Extract first frame from video and upload to S3.
    
    Args:
        project_id: Project UUID string
        video_asset_id: Video MediaAsset UUID string
    """
    project_uuid = uuid.UUID(project_id)
    video_uuid = uuid.UUID(video_asset_id)
    
    try:
        # Get database session
        with Session(engine) as session:
            # Query video MediaAsset
            video_asset = session.get(MediaAsset, video_uuid)
            
            if not video_asset:
                raise ValueError(f"Video asset {video_asset_id} not found")
            
            # Update processing status
            video_asset.is_processing = True
            video_asset.processing_error = None
            session.commit()
            
            # Generate presigned URL to download video from S3
            bucket, key = parse_s3_uri(video_asset.uri)
            presigned_url = generate_presigned_url(bucket, key)
            
            # Download video to temporary location
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)
                
                temp_video_path = Path(temp_video.name)
            
            # Extract first frame
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_frame:
                temp_frame_path = Path(temp_frame.name)
            
            extract_first_frame(temp_video_path, temp_frame_path)
            
            # Upload frame to S3
            frame_key = f"projects/{project_id}/frames/{uuid.uuid4()}.png"
            
            with open(temp_frame_path, 'rb') as frame_file:
                upload_result = upload_file_to_s3(
                    frame_file,
                    settings.AWS_S3_BUCKET,
                    frame_key
                )
            
            # Create MediaAsset record for frame
            frame_asset = create_media_asset(
                session=session,
                project_id=project_uuid,
                kind="image",
                uri=upload_result["uri"],
                bytes=temp_frame_path.stat().st_size,
                meta={
                    "source": "auto_extracted",
                    "frame_number": 0,
                    "parent_video_id": video_asset_id,
                    "filename": f"frame_{uuid.uuid4()}.png",
                    "content_type": "image/png",
                },
            )
            
            # Update video asset processing status
            video_asset.is_processing = False
            video_asset.processing_error = None
            session.commit()
            
            logger.info(f"Successfully extracted frame for project {project_id}")
            
            # Clean up temporary files
            temp_video_path.unlink(missing_ok=True)
            temp_frame_path.unlink(missing_ok=True)
            
            return {
                "success": True,
                "frame_asset_id": str(frame_asset.id),
                "frame_uri": frame_asset.uri,
            }
            
    except Exception as e:
        logger.error(f"Failed to extract frame for project {project_id}: {e}")
        
        # Update processing error status
        try:
            with Session(engine) as session:
                video_asset = session.get(MediaAsset, video_uuid)
                if video_asset:
                    video_asset.is_processing = False
                    video_asset.processing_error = str(e)
                    session.commit()
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")
        
        # Clean up temporary files
        try:
            temp_video_path.unlink(missing_ok=True)
            temp_frame_path.unlink(missing_ok=True)
        except:
            pass
        
        # Re-raise the exception to mark task as failed
        raise


# Example task - replace with your actual tasks
def example_task():
    """Example task that uses the database."""
    with Session(engine) as session:
        # Your task logic here
        pass
