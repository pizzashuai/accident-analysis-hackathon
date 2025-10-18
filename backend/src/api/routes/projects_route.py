import logging
import tempfile
import uuid
from pathlib import Path
import cv2
import numpy as np
from src.common.database.models.media_asset_table import MediaAsset

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
from src.common.features.homography import (
    HomographyPairCreate,
    HomographyPairPublic,
    HomographySessionCreate,
    HomographySessionPublic,
    HomographySolveResponse,
    HomographyModelPublic,
    create_session,
    get_or_create_session_for_project,
    get_session_with_relations,
    add_pair,
    update_pairs,
    delete_pair,
    solve_homography,
    export_homography_data,
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


@router.post("/{project_id}/extract-frame", response_model=MediaAssetPublic)
def extract_video_frame(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> MediaAssetPublic:
    """Extract first frame from project video and save as media asset."""
    temp_file_path = None
    
    try:
        # Verify project exists and belongs to user
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if not project.video_id:
            raise HTTPException(status_code=400, detail="No video uploaded for this project")

        # Get video media asset
        video_asset = session.get(MediaAsset, project.video_id)
        if not video_asset:
            raise HTTPException(status_code=404, detail="Video asset not found")

        # Parse S3 URI and download video temporarily
        bucket, key = parse_s3_uri(video_asset.uri)
        
        # For now, we'll use a placeholder approach since we need S3 download functionality
        # In a real implementation, you'd download the video from S3 first
        raise HTTPException(status_code=501, detail="Frame extraction not yet implemented - requires S3 download functionality")

        # TODO: Implement actual frame extraction:
        # 1. Download video from S3 to temp file
        # 2. Use OpenCV to extract first frame
        # 3. Save frame as image
        # 4. Upload image to S3
        # 5. Create media asset record
        # 6. Link to homography session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting video frame: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up temporary file
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)


# Homography endpoints

@router.post("/{project_id}/homography/session", response_model=HomographySessionPublic)
def create_homography_session(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> HomographySessionPublic:
    """Create or get active homography session for project."""
    try:
        # Verify project exists and belongs to user
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        homography_session = get_or_create_session_for_project(session, project_id)
        return HomographySessionPublic.model_validate(homography_session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating homography session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{project_id}/homography/session", response_model=HomographySessionPublic)
def get_homography_session(
    project_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> HomographySessionPublic:
    """Get current homography session for project."""
    try:
        # Verify project exists and belongs to user
        project = get_project_with_relations(
            session=session, project_id=project_id, user_id=current_user.id
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        homography_session = get_session_with_relations(session, project.homography_session.id) if project.homography_session else None
        if not homography_session:
            raise HTTPException(status_code=404, detail="No homography session found")

        return HomographySessionPublic.model_validate(homography_session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting homography session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/homography/sessions/{session_id}/pairs", response_model=HomographyPairPublic)
def add_homography_pair(
    session_id: uuid.UUID,
    pair_data: HomographyPairCreate,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> HomographyPairPublic:
    """Add point pair to homography session."""
    try:
        pair = add_pair(session, session_id, pair_data)
        return HomographyPairPublic.model_validate(pair)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding homography pair: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/homography/sessions/{session_id}/pairs", response_model=list[HomographyPairPublic])
def update_homography_pairs(
    session_id: uuid.UUID,
    pairs_data: list[HomographyPairCreate],
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> list[HomographyPairPublic]:
    """Replace all pairs in homography session (bulk update)."""
    try:
        pairs = update_pairs(session, session_id, pairs_data)
        return [HomographyPairPublic.model_validate(pair) for pair in pairs]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating homography pairs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/homography/pairs/{pair_id}")
def delete_homography_pair(
    pair_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
):
    """Delete specific homography pair."""
    try:
        success = delete_pair(session, pair_id)
        if not success:
            raise HTTPException(status_code=404, detail="Pair not found")
        return {"message": "Pair deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting homography pair: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/homography/sessions/{session_id}/solve", response_model=HomographySolveResponse)
def solve_homography_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> HomographySolveResponse:
    """Solve homography for session."""
    try:
        model = solve_homography(session, session_id)
        return HomographySolveResponse(
            success=True,
            model=HomographyModelPublic.model_validate(model)
        )
    except ValueError as e:
        return HomographySolveResponse(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        logger.error(f"Error solving homography: {e}")
        return HomographySolveResponse(
            success=False,
            error_message="Internal server error"
        )


@router.get("/homography/sessions/{session_id}/model", response_model=HomographyModelPublic)
def get_homography_model(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> HomographyModelPublic:
    """Get solved homography model."""
    try:
        homography_session = get_session_with_relations(session, session_id)
        if not homography_session:
            raise HTTPException(status_code=404, detail="Homography session not found")
        
        if not homography_session.model:
            raise HTTPException(status_code=404, detail="No solved model found")

        return HomographyModelPublic.model_validate(homography_session.model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting homography model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/homography/sessions/{session_id}/export")
def export_homography_session(
    session_id: uuid.UUID,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
):
    """Export homography data in process-video compatible format."""
    try:
        export_data = export_homography_data(session, session_id)
        return export_data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting homography data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
