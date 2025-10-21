import json
import logging
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
from sqlalchemy.orm import Session

from src.common.config import settings
from src.common.database.db import engine
from src.common.database.models.media_asset_table import MediaAsset
from src.common.database.models.project_table import Project
from src.common.features.postprocess.event_publisher import LLMEventPublisher
from src.common.features.postprocess.llm_agent import (
    LLMAccidentAnalysisAgent,
    LLMAgentConfig,
)
from src.common.features.processing.crud import (
    bulk_insert_detections,
    create_artifact,
    update_run_progress,
    update_run_status,
)
from src.common.features.project import create_media_asset
from src.common.features.storage import (
    extract_first_frame,
    generate_presigned_url,
    parse_s3_uri,
    upload_file_to_s3,
)

logger = logging.getLogger(__name__)

# Initialize Celery app
from src.worker.celery_app.app import app as celery_app


def create_video_processor():
    """Lazy import helper to avoid heavy ML dependencies during API startup."""
    from src.common.features.process_video import VideoProcessor

    return VideoProcessor(
        model_path="yolov8s.pt",
        conf_threshold=0.2,
        iou_threshold=0.3,
        classes=[2, 3, 5, 7, 9],  # Vehicle classes
        trail_length=10,
        bbox_smoothing_method="kalman",
        bbox_smoothing_window=5,
        speed_smoothing_method="moving_average",
        speed_smoothing_window=5,
        tracking_point="bottom_center",
    )


def convert_to_json_serializable(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        return str(obj)


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
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError as e:
        logger.error(f"Invalid project_id format: {project_id}. Error: {e}")
        raise ValueError(f"Invalid project_id format: {project_id}")

    try:
        video_uuid = uuid.UUID(video_asset_id)
    except ValueError as e:
        logger.error(f"Invalid video_asset_id format: {video_asset_id}. Error: {e}")
        raise ValueError(f"Invalid video_asset_id format: {video_asset_id}")

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
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)

                temp_video_path = Path(temp_video.name)

            # Extract first frame
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_frame:
                temp_frame_path = Path(temp_frame.name)

            extract_first_frame(temp_video_path, temp_frame_path)

            # Upload frame to S3
            frame_key = f"projects/{project_id}/frames/{uuid.uuid4()}.png"

            with open(temp_frame_path, "rb") as frame_file:
                upload_result = upload_file_to_s3(
                    frame_file, settings.AWS_S3_BUCKET, frame_key
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


@celery_app.task(bind=True)
def process_video_task(self, project_id: str, run_id: str):
    """
    Process video with YOLO detection, ByteTrack tracking, and speed calculation.

    Stages:
    1. Download video from S3
    2. Load homography data
    3. Run YOLO + ByteTrack (with progress updates)
    4. Calculate speeds using homography
    5. Bulk insert detections to DB
    6. Upload JSONL artifact to S3
    7. Mark run as completed
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError as e:
        logger.error(f"Invalid project_id format: {project_id}. Error: {e}")
        raise ValueError(f"Invalid project_id format: {project_id}")

    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError as e:
        logger.error(f"Invalid run_id format: {run_id}. Error: {e}")
        raise ValueError(f"Invalid run_id format: {run_id}")

    try:
        with Session(engine) as session:
            # Get project and validate
            project = session.get(Project, project_uuid)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            if not project.video_id:
                raise ValueError("Project has no video")

            # Get video asset
            video_asset = session.get(MediaAsset, project.video_id)
            if not video_asset:
                raise ValueError("Video asset not found")

            # Validate homography is solved
            if not project.homography_session:
                raise ValueError("No homography session found")

            homography_session = project.homography_session
            if not homography_session or homography_session.status != "solved":
                raise ValueError("Homography must be solved before processing")

            # Update status to running
            update_run_status(db=session, run_id=run_uuid, status="running")

            # Stage 1: Download video from S3
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="downloading",
                percent=10,
                message="Downloading video from S3...",
            )

            bucket, key = parse_s3_uri(video_asset.uri)
            presigned_url = generate_presigned_url(bucket, key)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)

                temp_video_path = Path(temp_video.name)

            # Validate video duration (< 5 seconds)
            cap = cv2.VideoCapture(str(temp_video_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open downloaded video")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0
            cap.release()

            if duration_sec > 5:
                raise ValueError(
                    f"Video duration ({duration_sec:.1f}s) exceeds 5 second limit"
                )

            # Stage 2: Load homography data
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="loading_homography",
                percent=20,
                message="Loading homography data...",
            )

            # Get homography model
            homography_model = homography_session.model
            if not homography_model:
                raise ValueError("Homography model not found")

            # Initialize video processor with optimal settings
            processor = create_video_processor()

            # Create homography data (Python objects, not files)
            homography_data = processor.create_homography_data(
                homography_session, homography_model
            )

            # Stage 3: Extract video frames and process with Python objects
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="extracting_frames",
                percent=30,
                message="Extracting video frames...",
            )

            # Extract all frames from video
            cap = cv2.VideoCapture(str(temp_video_path))
            video_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                video_frames.append(frame)
            cap.release()

            # Process video frames with progress callback
            def progress_callback(frame_count, total_frames, message):
                percent = 30 + (frame_count / total_frames) * 40  # 30-70% range
                update_run_progress(
                    db=session,
                    run_id=run_uuid,
                    stage="detecting",
                    percent=int(percent),
                    message=message,
                )

            processing_result = processor.process_video_detections_from_objects(
                video_frames=video_frames,
                fps=fps,
                homography_data=homography_data,
                progress_callback=progress_callback,
            )

            # Stage 4: Convert detections to database format with speed calculation
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="calculating_speeds",
                percent=70,
                message="Calculating speeds and preparing data...",
            )

            detections_data = (
                processor.convert_detections_to_database_format_from_objects(
                    detections_list=processing_result["detections"],
                    project_uuid=project_uuid,
                    video_width=width,
                    video_height=height,
                    homography_data=homography_data,
                )
            )

            # Stage 5: Bulk insert detections to DB
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="saving_detections",
                percent=85,
                message="Saving detections to database...",
            )

            bulk_insert_detections(
                db=session, run_id=run_uuid, detections_list=detections_data
            )

            # Stage 6: Create JSONL artifact and upload to S3
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="uploading_artifacts",
                percent=90,
                message="Creating JSONL artifact and uploading to S3...",
            )

            # Create JSONL content from detections data
            jsonl_content = ""
            speed_updates_count = 0

            # Calculate real-world event time if video start time is available
            video_start_time = video_asset.video_start_time

            for detection_data in detections_data:
                # Calculate real-world event time
                event_time_real = None
                if video_start_time:
                    # Convert video time (seconds) to real-world datetime
                    video_time_seconds = detection_data["t_ms"] / 1000.0
                    event_time_real = video_start_time + timedelta(
                        seconds=video_time_seconds
                    )

                # Convert detection data to JSONL format
                jsonl_record = {
                    "video_id": "video.mp4",
                    "frame": detection_data["frame_idx"],
                    "time": detection_data["t_ms"] / 1000.0,
                    "event_time_real": event_time_real.isoformat()
                    if event_time_real
                    else None,
                    "track_id": detection_data["track_id"],
                    "det_idx": 0,  # Not used in annotation
                    "class_id": detection_data["extra"]["class_id"],
                    "class_name": detection_data["cls"],
                    "conf": detection_data["conf"],
                    "bbox_xyxy": [
                        detection_data["x"],
                        detection_data["y"],
                        detection_data["x"] + detection_data["w"],
                        detection_data["y"] + detection_data["h"],
                    ],
                    "center": detection_data["extra"]["center"],
                    "speed_mph": detection_data["extra"]["speed_mph"],
                    "world_coords": [detection_data["wx"], detection_data["wy"]]
                    if detection_data["wx"] is not None
                    else None,
                    "tracking_point": detection_data["extra"]["tracking_point"],
                    "raw_bbox": detection_data["extra"]["raw_bbox"],
                }
                # Convert all NumPy types to Python native types for JSON serialization
                jsonl_record = convert_to_json_serializable(jsonl_record)
                jsonl_content += json.dumps(jsonl_record) + "\n"
                if detection_data["extra"]["speed_mph"] is not None:
                    speed_updates_count += 1

            # Upload JSONL content to S3
            jsonl_key = f"projects/{project_id}/runs/{run_id}/detections.jsonl"
            import io

            jsonl_file = io.BytesIO(jsonl_content.encode("utf-8"))
            upload_result = upload_file_to_s3(
                jsonl_file, settings.AWS_S3_BUCKET, jsonl_key
            )

            # Create artifact record
            create_artifact(
                db=session,
                project_id=project_uuid,
                run_id=run_uuid,
                kind="jsonl_detections",
                uri=upload_result["uri"],
                meta={
                    "detection_count": len(detections_data),
                    "video_duration": duration_sec,
                    "fps": fps,
                    "frame_count": frame_count,
                },
            )

            # Stage 7: Mark run as completed
            update_run_status(db=session, run_id=run_uuid, status="completed")

            logger.info(
                f"Successfully processed video for project {project_id}, run {run_id}"
            )

            # Clean up temporary files
            temp_video_path.unlink(missing_ok=True)

            return {
                "success": True,
                "detection_count": len(detections_data),
                "duration_sec": duration_sec,
                "fps": fps,
            }

    except Exception as e:
        logger.error(
            f"Failed to process video for project {project_id}, run {run_id}: {e}"
        )

        # Update run status to failed
        try:
            with Session(engine) as session:
                update_run_status(
                    db=session, run_id=run_uuid, status="failed", error_message=str(e)
                )
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")

        # Clean up temporary files
        try:
            temp_video_path.unlink(missing_ok=True)
        except:
            pass

        # Re-raise the exception to mark task as failed
        raise


@celery_app.task(bind=True)
def generate_annotated_video_task(self, project_id: str, run_id: str):
    """
    Generate pre-rendered annotated video with bounding boxes and speed labels.

    Steps:
    1. Download original video from S3
    2. Load detections from database
    3. Load homography data
    4. Use VideoProcessor to render annotated video
    5. Upload annotated video to S3
    6. Create artifact record
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError as e:
        logger.error(f"Invalid project_id format: {project_id}. Error: {e}")
        raise ValueError(f"Invalid project_id format: {project_id}")

    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError as e:
        logger.error(f"Invalid run_id format: {run_id}. Error: {e}")
        raise ValueError(f"Invalid run_id format: {run_id}")

    try:
        with Session(engine) as session:
            # Get project and validate
            project = session.get(Project, project_uuid)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            if not project.video_id:
                raise ValueError("Project has no video")

            # Get video asset
            video_asset = session.get(MediaAsset, project.video_id)
            if not video_asset:
                raise ValueError("Video asset not found")

            # Get homography session
            if not project.homography_session:
                raise ValueError("No homography session found")

            homography_session = project.homography_session
            if not homography_session or homography_session.status != "solved":
                raise ValueError("Homography must be solved")

            # Update status to running
            update_run_status(db=session, run_id=run_uuid, status="running")

            # Step 1: Download original video from S3
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="downloading_video",
                percent=10,
                message="Downloading original video...",
            )

            bucket, key = parse_s3_uri(video_asset.uri)
            presigned_url = generate_presigned_url(bucket, key)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)

                temp_video_path = Path(temp_video.name)

            # Step 2: Load detections from database
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="loading_detections",
                percent=20,
                message="Loading detections from database...",
            )

            from src.common.features.processing.crud import get_detections_by_run

            detections = get_detections_by_run(
                db=session, run_id=run_uuid, skip=0, limit=10000
            )  # Get all detections

            if not detections:
                raise ValueError("No detections found for this run")

            # Step 3: Load homography data
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="loading_homography",
                percent=30,
                message="Loading homography data...",
            )

            homography_model = homography_session.model
            if not homography_model:
                raise ValueError("Homography model not found")

            # Initialize video processor with optimal settings
            processor = create_video_processor()

            # Create homography data
            homography_data = processor.create_homography_data(
                homography_session, homography_model
            )
            homography_path = processor.save_homography_file(homography_data)

            # Step 4: Convert detections to JSONL format for VideoAnnotator
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="preparing_data",
                percent=40,
                message="Preparing detection data...",
            )

            # Create temporary JSONL file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as jsonl_file:
                jsonl_path = Path(jsonl_file.name)

            # Get video dimensions for coordinate normalization
            cap = cv2.VideoCapture(str(temp_video_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open video for dimension extraction")

            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Convert database detections to JSONL format
            video_start_time = video_asset.video_start_time

            with open(jsonl_path, "w") as f:
                for detection in detections:
                    # Calculate real-world event time
                    event_time_real = None
                    if video_start_time:
                        # Convert video time (seconds) to real-world datetime
                        video_time_seconds = detection.t_ms / 1000.0
                        event_time_real = video_start_time + timedelta(
                            seconds=video_time_seconds
                        )

                    # Get bbox coordinates
                    bbox = [
                        detection.x,
                        detection.y,
                        detection.x + detection.w,
                        detection.y + detection.h,
                    ]

                    # Convert database format to JSONL format expected by VideoAnnotator
                    detection_record = {
                        "video_id": video_asset.uri.split("/")[-1]
                        if video_asset.uri
                        else "video.mp4",
                        "frame": detection.frame_idx,
                        "time": detection.t_ms / 1000.0,
                        "event_time_real": event_time_real.isoformat()
                        if event_time_real
                        else None,
                        "track_id": detection.track_id,
                        "det_idx": 0,  # Not used in annotation
                        "class_id": detection.extra.get("class_id", 0),
                        "class_name": detection.cls,
                        "conf": detection.conf or 0.0,
                        "bbox_xyxy": bbox,
                        "center": detection.extra.get("center", [0, 0]),
                        "speed_mph": detection.extra.get("speed_mph"),
                        "world_coords": [detection.wx, detection.wy]
                        if detection.wx is not None
                        else None,
                        "tracking_point": detection.extra.get(
                            "tracking_point", "bottom_center"
                        ),
                        "raw_bbox": detection.extra.get("raw_bbox", bbox),
                    }
                    # Convert all NumPy types to Python native types for JSON serialization
                    detection_record = convert_to_json_serializable(detection_record)
                    f.write(json.dumps(detection_record) + "\n")

            # Step 5: Use VideoProcessor to render annotated video
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="rendering_video",
                percent=60,
                message="Rendering annotated video...",
            )

            # Create output video path
            output_video_path = Path(tempfile.mktemp(suffix=".mp4"))

            # Create annotated video using VideoProcessor
            processor.create_annotated_video(
                original_video_path=temp_video_path,
                jsonl_path=jsonl_path,
                output_path=output_video_path,
                homography_file=str(homography_path),
                show_trails=True,
                show_labels=True,
                show_boxes=True,
            )

            # Step 6: Upload annotated video to S3
            update_run_progress(
                db=session,
                run_id=run_uuid,
                stage="uploading_video",
                percent=80,
                message="Uploading annotated video to S3...",
            )

            # Upload annotated video
            video_key = f"projects/{project_id}/runs/{run_id}/annotated_video.mp4"
            with open(output_video_path, "rb") as video_file:
                upload_result = upload_file_to_s3(
                    video_file, settings.AWS_S3_BUCKET, video_key
                )

            # Create artifact record
            create_artifact(
                db=session,
                project_id=project_uuid,
                run_id=run_uuid,
                kind="annotated_video",
                uri=upload_result["uri"],
                meta={
                    "detection_count": len(detections),
                    "video_size_bytes": output_video_path.stat().st_size,
                    "filename": "annotated_video.mp4",
                    "content_type": "video/mp4",
                },
            )

            # Mark task as completed
            update_run_status(db=session, run_id=run_uuid, status="completed")

            logger.info(
                f"Successfully generated annotated video for project {project_id}, run {run_id}"
            )

            # Clean up temporary files
            temp_video_path.unlink(missing_ok=True)
            homography_path.unlink(missing_ok=True)
            jsonl_path.unlink(missing_ok=True)
            output_video_path.unlink(missing_ok=True)

            return {
                "success": True,
                "video_uri": upload_result["uri"],
                "detection_count": len(detections),
            }

    except Exception as e:
        logger.error(
            f"Failed to generate annotated video for project {project_id}, run {run_id}: {e}"
        )

        # Update run status to failed
        try:
            with Session(engine) as session:
                update_run_status(
                    db=session, run_id=run_uuid, status="failed", error_message=str(e)
                )
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")

        # Clean up temporary files
        try:
            temp_video_path.unlink(missing_ok=True)
            homography_path.unlink(missing_ok=True)
            jsonl_path.unlink(missing_ok=True)
            output_video_path.unlink(missing_ok=True)
        except:
            pass

        # Re-raise the exception to mark task as failed
        raise


@celery_app.task(bind=True)
def analyze_accident_llm_task(
    self, analysis_id: str, project_id: str, run_id: str, detections_file_path: str
):
    """
    Analyze accident data using LLM agent with real-time event publishing.

    Args:
        analysis_id: Unique identifier for this analysis session
        project_id: Project UUID string
        run_id: Processing run UUID string
        detections_file_path: Path to the filtered JSONL detections file
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError as e:
        logger.error(f"Invalid project_id format: {project_id}. Error: {e}")
        raise ValueError(f"Invalid project_id format: {project_id}")

    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError as e:
        logger.error(f"Invalid run_id format: {run_id}. Error: {e}")
        raise ValueError(f"Invalid run_id format: {run_id}")

    try:
        with Session(engine) as session:
            # Get project and validate
            project = session.get(Project, project_uuid)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            # Get processing run
            from src.common.features.processing.crud import get_processing_run

            run = get_processing_run(db=session, run_id=run_uuid)
            if not run:
                raise ValueError(f"Processing run {run_id} not found")

            # Extract track IDs from the filtered JSONL file metadata
            # The file should have been filtered by track IDs already
            track_ids = []
            try:
                with open(detections_file_path) as f:
                    for line in f:
                        if line.strip():
                            import json

                            detection = json.loads(line.strip())
                            track_id = detection.get("track_id")
                            if track_id is not None and track_id not in track_ids:
                                track_ids.append(track_id)
            except Exception as e:
                logger.warning(f"Could not extract track IDs from file: {e}")
                # Fallback: use common track IDs
                track_ids = [7, 14]  # Default track IDs

            if not track_ids:
                raise ValueError("No track IDs found in detections file")

            logger.info(f"Analyzing tracks: {track_ids}")

            # Get existing analysis record from database
            from src.common.features.llm_analysis.crud import get_analysis, update_analysis_status, update_analysis_result, update_analysis_error
            
            # Get existing analysis record (created by API route)
            analysis_record = get_analysis(session, analysis_id)
            if not analysis_record:
                raise ValueError(f"Analysis record {analysis_id} not found - it should have been created by the API route")
            
            logger.info(f"Found existing analysis record {analysis_record.id} for session {analysis_id}")

            # Update status to analyzing
            update_analysis_status(session, analysis_id, "analyzing")

            # Initialize event publisher
            event_publisher = LLMEventPublisher()
            logger.info(f"Initialized event publisher for analysis {analysis_id}")

            try:

                # Create LLM agent configuration
                config = LLMAgentConfig(
                    track_ids=track_ids,
                    frame_range=None,  # Use all frames
                    iou_threshold=0.01,
                    distance_threshold_m=5.0,
                    persistence_frames=3,
                    padding_frames=10,
                    detections_file=detections_file_path,
                    aws_region=settings.AWS_REGION,
                    bedrock_models=[
                        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "us.anthropic.claude-sonnet-4-20250514-v1:0",
                        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                    ],
                    max_iterations=20,
                    temperature=0.0,
                    max_tokens=4096,
                )

                # Initialize LLM agent with event publishing
                agent = LLMAccidentAnalysisAgent(
                    config=config,
                    event_publisher=event_publisher,
                    analysis_id=analysis_id,
                )

                # Send test event to verify publishing works
                event_publisher.publish_thinking_start(analysis_id, "Starting LLM analysis...")
                logger.info(f"Published test event for analysis {analysis_id}")

                # Run analysis
                result = agent.analyze()

                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error occurred")
                    event_publisher.publish_error(analysis_id, error_msg)
                    raise RuntimeError(f"LLM analysis failed: {error_msg}")

                logger.info(
                    f"Successfully completed LLM analysis for project {project_id}"
                )

                # Store analysis result in database with complete data for frontend display
                try:
                    # Extract timeline and tool call data from the result for frontend display
                    enhanced_result = result.copy()
                    
                    # Extract timeline from tool results if available
                    timeline_data = None
                    weather_data = None
                    
                    # Look for timeline data in tool results
                    if "execution_log" in result:
                        for log_entry in result["execution_log"]:
                            if isinstance(log_entry, dict):
                                tool_name = log_entry.get("tool")
                                tool_result = log_entry.get("result", {})
                                
                                if isinstance(tool_result, dict):
                                    # Extract timeline from build_timeline tool
                                    if tool_name == "build_timeline" and tool_result.get("success"):
                                        timeline_data = tool_result.get("timeline", [])
                                    # Extract weather data from get_weather_data tool
                                    elif tool_name == "get_weather_data" and tool_result.get("success"):
                                        weather_data = tool_result.get("weather_data")
                    
                    # Add structured data for frontend
                    enhanced_result["frontend_data"] = {
                        "timeline": timeline_data,
                        "weather_data": weather_data,
                        "tool_calls": result.get("execution_log", []),
                        "analysis_text": result.get("report", ""),
                        "collision_detected": result.get("collision_detected", False),
                        "track_ids": track_ids
                    }
                    
                    # Extract collision frame and timestamp for report generation
                    collision_frame = None
                    collision_timestamp = None
                    
                    if timeline_data:
                        for event in timeline_data:
                            if isinstance(event, dict) and event.get("type") == "collision":
                                collision_frame = event.get("frame")
                                collision_timestamp = event.get("timestamp")
                                break
                    
                    # Add collision data for report generation
                    if collision_frame is not None and collision_timestamp is not None:
                        enhanced_result["collision_data"] = {
                            "frame": collision_frame,
                            "timestamp": collision_timestamp
                        }
                    
                    update_analysis_result(session, analysis_id, enhanced_result)
                    logger.info(f"Stored enhanced analysis result in database for {analysis_id}")
                except Exception as e:
                    logger.error(f"Could not store analysis result in database: {e}")
                    # Don't fail the entire analysis if database storage fails

                # Automatically trigger PDF report generation
                try:
                    from src.common.features.report.crud import create_report
                    
                    # Create report record
                    report = create_report(
                        db=session,
                        project_id=project_uuid,
                        run_id=run_uuid,
                        llm_analysis_id=analysis_record.id,
                        meta={
                            "auto_generated": True,
                            "analysis_completed_at": datetime.utcnow().isoformat(),
                            "track_ids": track_ids,
                        }
                    )
                    
                    # Trigger PDF generation task
                    generate_pdf_report_task.delay(
                        report_id=str(report.id),
                        project_id=project_id,
                        run_id=run_id,
                        llm_analysis_id=str(analysis_record.id),
                    )
                    
                    logger.info(f"Automatically triggered PDF generation for report {report.id}")
                    
                except Exception as e:
                    logger.error(f"Failed to trigger automatic PDF generation: {e}")
                    # Don't fail the entire analysis if PDF generation fails

                return {"success": True, "analysis_id": analysis_id, "result": result}

            finally:
                # Close event publisher
                event_publisher.close()

    except Exception as e:
        logger.error(
            f"Failed to analyze accident with LLM for project {project_id}, run {run_id}: {e}"
        )

        # Publish error event if possible
        try:
            event_publisher = LLMEventPublisher()
            event_publisher.publish_error(analysis_id, str(e))
            event_publisher.close()
        except:
            pass  # Ignore errors in error publishing

        # Update analysis status to failed
        try:
            with Session(engine) as session:
                from src.common.features.llm_analysis.crud import update_analysis_error
                update_analysis_error(session, analysis_id, str(e))
        except Exception as db_error:
            logger.error(f"Could not update analysis error status: {db_error}")

        # Re-raise the exception to mark task as failed
        raise


@celery_app.task(bind=True)
def generate_pdf_report_task(
    self, report_id: str, project_id: str, run_id: str, llm_analysis_id: str
):
    """
    Generate PDF report from LLM analysis results.
    
    Args:
        report_id: Report UUID string
        project_id: Project UUID string
        run_id: Processing run UUID string
        llm_analysis_id: LLM analysis UUID string
    """
    try:
        report_uuid = uuid.UUID(report_id)
        project_uuid = uuid.UUID(project_id)
        run_uuid = uuid.UUID(run_id)
        llm_analysis_uuid = uuid.UUID(llm_analysis_id)
    except ValueError as e:
        logger.error(f"Invalid UUID format: {e}")
        raise ValueError(f"Invalid UUID format: {e}")

    try:
        with Session(engine) as session:
            # Get report record
            from src.common.features.report.crud import get_report, update_report_status, update_report_pdf_uri
            
            report = get_report(db=session, report_id=report_uuid)
            if not report:
                raise ValueError(f"Report {report_id} not found")

            # Update status to generating
            update_report_status(db=session, report_id=report_uuid, status="generating")
            logger.info(f"Started PDF generation for report {report_id}")

            # Get project and validate
            project = session.get(Project, project_uuid)
            if not project:
                raise ValueError(f"Project {project_id} not found")

            if not project.video_id:
                raise ValueError("Project has no video")

            # Get video asset
            video_asset = session.get(MediaAsset, project.video_id)
            if not video_asset:
                raise ValueError("Video asset not found")

            # Get processing run
            from src.common.features.processing.crud import get_processing_run
            run = get_processing_run(db=session, run_id=run_uuid)
            if not run:
                raise ValueError(f"Processing run {run_id} not found")

            # Find filtered JSONL artifact for this run
            from src.common.database.models.artifact_table import Artifact
            artifacts = (
                session.query(Artifact)
                .filter(Artifact.run_id == run_uuid, Artifact.kind == "jsonl_detections")
                .all()
            )

            filtered_artifact = None
            for artifact in artifacts:
                if artifact.meta and "filtered_track_ids" in artifact.meta:
                    filtered_artifact = artifact
                    break

            if not filtered_artifact:
                raise ValueError("No filtered detections found")

            # Download video from S3
            bucket, key = parse_s3_uri(video_asset.uri)
            presigned_url = generate_presigned_url(bucket, key)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()

                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)

                temp_video_path = Path(temp_video.name)

            # Download filtered JSONL artifact
            bucket, key = parse_s3_uri(filtered_artifact.uri)
            presigned_url = generate_presigned_url(bucket, key)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_jsonl:
                response = requests.get(presigned_url)
                response.raise_for_status()
                temp_jsonl.write(response.text)
                temp_jsonl_path = Path(temp_jsonl.name)

            # Get LLM analysis result from database
            from src.common.features.llm_analysis.crud import get_analysis_by_id
            
            analysis_record = get_analysis_by_id(db=session, id=llm_analysis_uuid)
            if not analysis_record:
                raise ValueError(f"LLM analysis {llm_analysis_id} not found")
            
            if not analysis_record.result_data:
                raise ValueError(f"LLM analysis {llm_analysis_id} has no result data")
            
            analysis_result = analysis_record.result_data
            logger.info(f"Retrieved analysis result from database for {llm_analysis_id}")

            # Generate collision screenshots
            from src.common.features.report.screenshot_generator import generate_collision_screenshots
            
            screenshot_result = generate_collision_screenshots(
                video_path=str(temp_video_path),
                detections_jsonl_path=str(temp_jsonl_path),
                analysis_result=analysis_result,
                output_dir=None  # Use temporary directory
            )

            if not screenshot_result["success"]:
                logger.warning(f"Screenshot generation failed: {screenshot_result['error']}")

            # Generate PDF report
            from src.common.features.report.pdf_generator import generate_pdf_report
            
            pdf_result = generate_pdf_report(
                analysis_text=analysis_result.get("analysis", "") or analysis_result.get("report", "No analysis available"),
                project_title=project.title,
                project_description=project.description,
                video_screenshot_path=screenshot_result.get("video_screenshot_path"),
                map_overlay_path=screenshot_result.get("map_overlay_path"),
                metadata={
                    "analysis_id": llm_analysis_id,
                    "track_ids": analysis_result.get("track_ids", []),
                    "collision_frame": screenshot_result.get("collision_frame", 0),
                    "collision_timestamp": screenshot_result.get("collision_timestamp", 0.0),
                    "collision_point": screenshot_result.get("collision_point", (None, None)),
                    "generated_at": datetime.utcnow().isoformat(),
                    "project_id": str(project_id),
                    "run_id": str(run_id)
                }
            )

            if not pdf_result["success"]:
                raise RuntimeError(f"PDF generation failed: {pdf_result['error']}")

            # Upload PDF to S3
            pdf_key = f"projects/{project_id}/reports/{report_id}/report.pdf"
            with open(pdf_result["output_path"], "rb") as pdf_file:
                upload_result = upload_file_to_s3(
                    pdf_file, settings.AWS_S3_BUCKET, pdf_key
                )

            # Update report with PDF URI and metadata
            meta_updates = {
                "pdf_size_bytes": Path(pdf_result["output_path"]).stat().st_size,
                "screenshot_info": {
                    "video_screenshot_path": screenshot_result.get("video_screenshot_path"),
                    "map_overlay_path": screenshot_result.get("map_overlay_path"),
                    "collision_frame": screenshot_result.get("collision_frame"),
                    "collision_timestamp": screenshot_result.get("collision_timestamp"),
                    "collision_point": screenshot_result.get("collision_point")
                },
                "analysis_metadata": {
                    "track_ids": analysis_result.get("track_ids", []),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }

            update_report_pdf_uri(
                db=session,
                report_id=report_uuid,
                pdf_uri=upload_result["uri"],
                meta_updates=meta_updates
            )

            logger.info(f"Successfully generated PDF report for project {project_id}")

            # Clean up temporary files
            temp_video_path.unlink(missing_ok=True)
            temp_jsonl_path.unlink(missing_ok=True)
            Path(pdf_result["output_path"]).unlink(missing_ok=True)
            
            # Clean up screenshot files
            if screenshot_result.get("video_screenshot_path"):
                Path(screenshot_result["video_screenshot_path"]).unlink(missing_ok=True)
            if screenshot_result.get("map_overlay_path"):
                Path(screenshot_result["map_overlay_path"]).unlink(missing_ok=True)

            return {
                "success": True,
                "report_id": report_id,
                "pdf_uri": upload_result["uri"]
            }

    except Exception as e:
        logger.error(f"Failed to generate PDF report for project {project_id}: {e}")

        # Update report status to failed
        try:
            with Session(engine) as session:
                from src.common.features.report.crud import update_report_status
                update_report_status(
                    db=session,
                    report_id=report_uuid,
                    status="failed",
                    error_message=str(e)
                )
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")

        # Clean up temporary files
        try:
            temp_video_path.unlink(missing_ok=True)
            temp_jsonl_path.unlink(missing_ok=True)
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
