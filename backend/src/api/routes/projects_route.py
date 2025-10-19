import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import requests
from typing import Optional
from src.common.database.models.media_asset_table import MediaAsset
from src.common.database.models.project_table import Project
from src.common.database.models.homography_session_table import HomographySession
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Response
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
from src.common.features.processing import (
    ProcessingRunCreate,
    ProcessingRunPublic,
    ProcessingRunsPublic,
    DetectionPublic,
    DetectionsPublic,
    ArtifactPublic,
    ArtifactsPublic,
    create_processing_run,
    get_processing_run,
    list_processing_runs,
    get_detections_by_run,
    get_detections_by_frame,
    list_artifacts,
    get_artifact,
)
from src.worker.celery_app.tasks import process_video_task, generate_annotated_video_task
from src.common.features.storage import (
    download_file_from_s3,
    extract_first_frame,
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
    video_start_time: str | None = Form(None),
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
        
        # Parse video start time if provided
        parsed_start_time = None
        if video_start_time and video_start_time.strip():
            try:
                parsed_start_time = datetime.fromisoformat(video_start_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid video_start_time format. Use ISO format (e.g., 2024-01-01T12:00:00Z)")
        
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
            video_start_time=parsed_start_time,
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


@router.patch("/{project_id}/media/{media_asset_id}/video-start-time", response_model=MediaAssetPublic)
def update_video_start_time(
    project_id: uuid.UUID,
    media_asset_id: uuid.UUID,
    current_user: CurrentUser,
    video_start_time: str | None = Form(None),
    session: Session = Depends(get_db),
) -> MediaAssetPublic:
    """Update video start time for a media asset."""
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

        # Only allow updating video assets
        if media_asset.kind != "video":
            raise HTTPException(status_code=400, detail="Can only update video start time for video assets")

        # Parse video start time if provided
        parsed_start_time = None
        if video_start_time and video_start_time.strip():
            try:
                parsed_start_time = datetime.fromisoformat(video_start_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid video_start_time format. Use ISO format (e.g., 2024-01-01T12:00:00Z)")

        # Update the video start time
        media_asset.video_start_time = parsed_start_time
        session.commit()
        session.refresh(media_asset)

        return MediaAssetPublic.model_validate(media_asset)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating video start time: {e}")
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
        
        # Create temporary file for video download
        temp_video_path = Path(tempfile.mktemp(suffix=".mp4"))
        temp_file_path = temp_video_path
        
        # 1. Download video from S3 to temp file
        download_file_from_s3(bucket, key, str(temp_video_path))
        
        # 2. Use OpenCV to extract first frame
        temp_frame_path = Path(tempfile.mktemp(suffix=".png"))
        extract_first_frame(temp_video_path, temp_frame_path)
        
        # 3. Save frame as image (already done by extract_first_frame)
        # 4. Upload image to S3
        frame_key = f"frames/{project_id}/{uuid.uuid4()}.png"
        with open(temp_frame_path, 'rb') as frame_file:
            upload_result = upload_file_to_s3(frame_file, bucket, frame_key)
        
        # 5. Create media asset record
        frame_asset = MediaAsset(
            project_id=project_id,
            kind="image",
            uri=f"s3://{bucket}/{frame_key}",
            bytes=temp_frame_path.stat().st_size,
            meta={
                "extracted_from_video": str(video_asset.id),
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "frame_type": "first_frame"
            }
        )
        session.add(frame_asset)
        session.flush()  # Get the ID
        
        # 6. Link to homography session
        homography_session = get_or_create_session_for_project(session, project_id)
        homography_session.screenshot_asset_id = frame_asset.id
        session.commit()
        
        # Clean up temporary frame file
        if temp_frame_path.exists():
            temp_frame_path.unlink(missing_ok=True)
        
        return MediaAssetPublic.model_validate(frame_asset)
        
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


# Processing endpoints

@router.post("/{project_id}/processing/start", response_model=ProcessingRunPublic)
def start_processing(
    *,
    project_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
    processing_in: ProcessingRunCreate,
) -> ProcessingRunPublic:
    """Start video processing for a project."""
    try:
        project_uuid = uuid.UUID(project_id)
        
        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Validate homography is solved
        if not project.homography_session:
            raise HTTPException(status_code=400, detail="No homography session found")
        
        homography_session = project.homography_session
        if not homography_session or homography_session.status != "solved":
            raise HTTPException(status_code=400, detail="Homography must be solved before processing")
        
        # Validate video exists and duration
        if not project.video_id:
            raise HTTPException(status_code=400, detail="Project has no video")
        
        video_asset = session.get(MediaAsset, project.video_id)
        if not video_asset:
            raise HTTPException(status_code=400, detail="Video asset not found")
        
        # Check video duration (< 10 seconds)
        bucket, key = parse_s3_uri(video_asset.uri)
        presigned_url = generate_presigned_url(bucket, key)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            response = requests.get(presigned_url, stream=True)
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=8192):
                temp_video.write(chunk)
            
            temp_video_path = Path(temp_video.name)
        
        cap = cv2.VideoCapture(str(temp_video_path))
        if not cap.isOpened():
            temp_video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Failed to open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()
        temp_video_path.unlink(missing_ok=True)
        
        if duration_sec > 10:
            raise HTTPException(status_code=400, detail=f"Video duration ({duration_sec:.1f}s) exceeds 10 second limit")
        
        # Create processing run
        processing_run = create_processing_run(
            db=session,
            project_id=project_uuid,
            homography_session_id=project.homography_session.id,
            params=processing_in.params
        )
        
        # Enqueue processing task
        process_video_task.delay(project_id, str(processing_run.id))
        
        return ProcessingRunPublic.model_validate(processing_run)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{project_id}/processing/runs", response_model=ProcessingRunsPublic)
def list_processing_runs_route(
    *,
    project_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> ProcessingRunsPublic:
    """List all processing runs for a project."""
    try:
        project_uuid = uuid.UUID(project_id)
        
        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        runs = list_processing_runs(db=session, project_id=project_uuid)
        
        return ProcessingRunsPublic(
            data=[ProcessingRunPublic.model_validate(run) for run in runs],
            count=len(runs)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing processing runs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing/runs/{run_id}", response_model=ProcessingRunPublic)
def get_processing_run_route(
    *,
    run_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> ProcessingRunPublic:
    """Get a single processing run."""
    try:
        run_uuid = uuid.UUID(run_id)
        
        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")
        
        # Validate ownership through project
        project = session.get(Project, run.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        return ProcessingRunPublic.model_validate(run)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing run: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing/runs/{run_id}/detections", response_model=DetectionsPublic)
def get_detections_route(
    *,
    run_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
    frame_idx: Optional[int] = None,
) -> DetectionsPublic:
    """Get paginated detections for a run."""
    try:
        run_uuid = uuid.UUID(run_id)
        
        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")
        
        # Validate ownership through project
        project = session.get(Project, run.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        if frame_idx is not None:
            detections = get_detections_by_frame(db=session, run_id=run_uuid, frame_idx=frame_idx)
        else:
            detections = get_detections_by_run(db=session, run_id=run_uuid, skip=skip, limit=limit)
        
        return DetectionsPublic(
            data=[DetectionPublic.model_validate(detection) for detection in detections],
            count=len(detections)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing/runs/{run_id}/detections/frames/{frame_idx}", response_model=DetectionsPublic)
def get_detections_by_frame_route(
    *,
    run_id: str,
    frame_idx: int,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> DetectionsPublic:
    """Get detections for a specific frame."""
    try:
        run_uuid = uuid.UUID(run_id)
        
        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")
        
        # Validate ownership through project
        project = session.get(Project, run.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        detections = get_detections_by_frame(db=session, run_id=run_uuid, frame_idx=frame_idx)
        
        return DetectionsPublic(
            data=[DetectionPublic.model_validate(detection) for detection in detections],
            count=len(detections)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detections by frame: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/processing/runs/{run_id}/generate-video", response_model=ArtifactPublic)
def generate_annotated_video_route(
    *,
    run_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> ArtifactPublic:
    """Generate annotated video for a completed processing run."""
    try:
        run_uuid = uuid.UUID(run_id)
        
        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")
        
        # Validate ownership through project
        project = session.get(Project, run.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Validate run is completed
        if run.status != "completed":
            raise HTTPException(status_code=400, detail="Processing run must be completed to generate video")
        
        # Enqueue video generation task
        generate_annotated_video_task.delay(str(project.id), str(run.id))
        
        # Return a placeholder artifact (the actual artifact will be created by the task)
        placeholder_artifact = ArtifactPublic(
            id=uuid.uuid4(),
            kind="annotated_video",
            uri="",  # Will be updated by the task
            meta={"status": "generating"},
            created_at=datetime.utcnow()
        )
        
        return placeholder_artifact
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating annotated video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/processing/runs/{run_id}/artifacts", response_model=ArtifactsPublic)
def list_artifacts_route(
    *,
    run_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> ArtifactsPublic:
    """List artifacts for a processing run."""
    try:
        run_uuid = uuid.UUID(run_id)
        
        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")
        
        # Validate ownership through project
        project = session.get(Project, run.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        artifacts = list_artifacts(db=session, run_id=run_uuid)
        
        return ArtifactsPublic(
            data=[ArtifactPublic.model_validate(artifact) for artifact in artifacts],
            count=len(artifacts)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/artifacts/{artifact_id}/download")
def get_artifact_download_url(
    *,
    artifact_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
) -> dict:
    """Get presigned download URL for an artifact."""
    try:
        artifact_uuid = uuid.UUID(artifact_id)
        
        artifact = get_artifact(db=session, artifact_id=artifact_uuid)
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
        logger.error(f"Error getting artifact download URL: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/artifacts/{artifact_id}/content")
def get_artifact_content(
    *,
    artifact_id: str,
    session: Session = Depends(get_db),
    current_user: CurrentUser,
):
    """Get artifact content directly (proxied from S3 to avoid CORS issues)."""
    try:
        artifact_uuid = uuid.UUID(artifact_id)
        
        artifact = get_artifact(db=session, artifact_id=artifact_uuid)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Validate ownership through project
        project = session.get(Project, artifact.project_id)
        if not project or project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Generate presigned URL and fetch content
        bucket, key = parse_s3_uri(artifact.uri)
        presigned_url = generate_presigned_url(bucket, key)
        
        # Fetch content from S3
        import requests
        response = requests.get(presigned_url)
        response.raise_for_status()
        
        # Determine content type based on artifact kind
        content_type = "application/octet-stream"
        if artifact.kind == "jsonl_detections":
            content_type = "application/json"
        elif artifact.kind == "csv_detections":
            content_type = "text/csv"
        
        return Response(
            content=response.content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={key.split('/')[-1]}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting artifact content: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
