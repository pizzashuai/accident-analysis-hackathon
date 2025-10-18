import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.deps import CurrentUser, get_db
from src.common.features.project import (
    MediaAssetPublic,
    Message,
    ProjectCreate,
    ProjectLocationCreate,
    ProjectLocationPublic,
    ProjectPublic,
    ProjectsPublic,
    ProjectUpdate,
    create_media_asset,
    create_project,
    delete_project,
    get_project_with_relations,
    list_projects,
    update_project,
    upsert_project_location,
)
from src.common.features.storage import (
    extract_video_metadata,
    generate_presigned_url,
    parse_s3_uri,
    upload_file_to_s3,
    validate_video_file,
)
from src.common.config import settings
from src.worker.celery_app.tasks import extract_video_frame_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])

# Ensure uploads directory exists
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


@router.post("/", response_model=ProjectPublic)
def create_project_route(
    *,
    session: Session = Depends(get_db),
    project_in: ProjectCreate,
    current_user: CurrentUser,
) -> ProjectPublic:
    """Create a new project."""
    try:
        project = create_project(
            session=session, project_create=project_in, user_id=current_user.id
        )
        return ProjectPublic.model_validate(project)
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=ProjectsPublic)
def read_projects(
    current_user: CurrentUser,
    session: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
) -> ProjectsPublic:
    """Retrieve user's projects."""
    try:
        projects, count = list_projects(
            session=session, user_id=current_user.id, skip=skip, limit=limit
        )
        return ProjectsPublic(
            data=[ProjectPublic.model_validate(project) for project in projects],
            count=count,
        )
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{project_id}", response_model=ProjectPublic)
def read_project(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> ProjectPublic:
    """Get a specific project by id."""
    try:
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectPublic.model_validate(project)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{project_id}", response_model=ProjectPublic)
def update_project_route(
    project_id: uuid.UUID,
    project_in: ProjectUpdate,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> ProjectPublic:
    """Update a project."""
    try:
        project = update_project(
            session=session,
            project_id=project_id,
            user_id=current_user.id,
            project_update=project_in,
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectPublic.model_validate(project)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{project_id}", response_model=Message)
def delete_project_route(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> Message:
    """Delete a project."""
    try:
        success = delete_project(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        return Message(message="Project deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{project_id}/upload-video", response_model=MediaAssetPublic)
def upload_video(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
) -> MediaAssetPublic:
    """Upload a video file for a project."""
    temp_file_path = None
    
    try:
        # Verify project exists and belongs to user
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Validate file type
        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="File must be a video")

        # Save file temporarily for metadata extraction
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix if file.filename else ".mp4", delete=False) as temp_file:
            content = file.file.read()
            temp_file.write(content)
            temp_file_path = Path(temp_file.name)

        # Validate video file
        if not validate_video_file(temp_file_path):
            raise HTTPException(status_code=400, detail="Invalid video file")

        # Extract video metadata
        metadata = extract_video_metadata(temp_file_path)
        
        # Generate S3 key
        file_extension = Path(file.filename).suffix if file.filename else ".mp4"
        s3_key = f"projects/{project_id}/videos/{uuid.uuid4()}{file_extension}"

        # Upload to S3
        with open(temp_file_path, "rb") as video_file:
            upload_result = upload_file_to_s3(
                video_file,
                settings.AWS_S3_BUCKET,
                s3_key
            )

        # Create media asset record
        media_asset = create_media_asset(
            session=session,
            project_id=project_id,
            kind="video",
            uri=upload_result["uri"],
            bytes=len(content),
            meta={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(content),
                **metadata,  # Include fps, duration, width, height, frame_count
            },
        )

        # Update project's video_id
        project.video_id = media_asset.id
        session.commit()

        # Trigger Celery task for frame extraction
        extract_video_frame_task.delay(str(project_id), str(media_asset.id))

        return MediaAssetPublic.model_validate(media_asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)


@router.get("/{project_id}/media/{media_asset_id}/url")
def get_media_presigned_url(
    project_id: uuid.UUID,
    media_asset_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
):
    """Generate presigned URL for media asset streaming."""
    try:
        # Verify project exists and belongs to user
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Query MediaAsset
        from src.common.database.models.media_asset_table import MediaAsset
        media_asset = session.get(MediaAsset, media_asset_id)
        
        if not media_asset or media_asset.project_id != project_id:
            raise HTTPException(status_code=404, detail="Media asset not found")

        # Parse S3 URI and generate presigned URL
        bucket, key = parse_s3_uri(media_asset.uri)
        presigned_url = generate_presigned_url(bucket, key)

        return {
            "url": presigned_url,
            "expires_in": settings.S3_PRESIGNED_URL_EXPIRATION,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{project_id}/location", response_model=ProjectLocationPublic)
def set_project_location(
    project_id: uuid.UUID,
    location_data: ProjectLocationCreate,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> ProjectLocationPublic:
    """Set or update project location."""
    try:
        # Verify project exists and belongs to user
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        location = upsert_project_location(
            session=session, project_id=project_id, location_data=location_data
        )
        return ProjectLocationPublic.model_validate(location)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting project location: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
