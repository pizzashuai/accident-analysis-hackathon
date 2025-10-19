import logging
import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import cv2
import numpy as np

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
from src.common.database.models.project_table import Project
from src.common.database.models.homography_session_table import HomographySession
from src.common.features.processing.crud import (
    update_run_progress,
    update_run_status,
    bulk_insert_detections,
    create_artifact,
)
from src.common.features.process_video import VideoProcessor, VideoProcessingResult
from src.common.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
from src.worker.celery_app.app import app as celery_app


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
            update_run_progress(db=session, run_id=run_uuid, stage="downloading", percent=10, message="Downloading video from S3...")
            
            bucket, key = parse_s3_uri(video_asset.uri)
            presigned_url = generate_presigned_url(bucket, key)
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)
                
                temp_video_path = Path(temp_video.name)
            
            # Validate video duration (< 10 seconds)
            cap = cv2.VideoCapture(str(temp_video_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open downloaded video")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0
            cap.release()
            
            if duration_sec > 10:
                raise ValueError(f"Video duration ({duration_sec:.1f}s) exceeds 10 second limit")
            
            # Stage 2: Load homography data
            update_run_progress(db=session, run_id=run_uuid, stage="loading_homography", percent=20, message="Loading homography data...")
            
            # Get homography model
            homography_model = homography_session.model
            if not homography_model:
                raise ValueError("Homography model not found")
            
            # Initialize video processor with optimal settings
            processor = VideoProcessor(
                model_path="yolov8s.pt",
                conf_threshold=0.2,
                iou_threshold=0.3,
                classes=[2, 3, 5, 7, 9],  # Vehicle classes
                trail_length=10,
                # Optimal smoothing settings
                bbox_smoothing_method="kalman",
                bbox_smoothing_window=5,
                speed_smoothing_method="moving_average",
                speed_smoothing_window=5,
                tracking_point="bottom_center",
            )
            
            # Create homography data (Python objects, not files)
            homography_data = processor.create_homography_data(homography_session, homography_model)
            
            # Stage 3: Extract video frames and process with Python objects
            update_run_progress(db=session, run_id=run_uuid, stage="extracting_frames", percent=30, message="Extracting video frames...")
            
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
                update_run_progress(db=session, run_id=run_uuid, stage="detecting", percent=int(percent), message=message)
            
            processing_result = processor.process_video_detections_from_objects(
                video_frames=video_frames,
                fps=fps,
                homography_data=homography_data,
                progress_callback=progress_callback,
            )
            
            # Stage 4: Convert detections to database format with speed calculation
            update_run_progress(db=session, run_id=run_uuid, stage="calculating_speeds", percent=70, message="Calculating speeds and preparing data...")
            
            detections_data = processor.convert_detections_to_database_format_from_objects(
                detections_list=processing_result["detections"],
                project_uuid=project_uuid,
                video_width=width,
                video_height=height,
                homography_data=homography_data,
            )
            
            # Stage 5: Bulk insert detections to DB
            update_run_progress(db=session, run_id=run_uuid, stage="saving_detections", percent=85, message="Saving detections to database...")
            
            bulk_insert_detections(db=session, run_id=run_uuid, detections_list=detections_data)
            
            # Stage 6: Create JSONL artifact and upload to S3
            update_run_progress(db=session, run_id=run_uuid, stage="uploading_artifacts", percent=90, message="Creating JSONL artifact and uploading to S3...")
            
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
                    event_time_real = video_start_time + timedelta(seconds=video_time_seconds)
                
                # Convert detection data to JSONL format
                jsonl_record = {
                    "video_id": "video.mp4",
                    "frame": detection_data["frame_idx"],
                    "time": detection_data["t_ms"] / 1000.0,
                    "event_time_real": event_time_real.isoformat() if event_time_real else None,
                    "track_id": detection_data["track_id"],
                    "det_idx": 0,  # Not used in annotation
                    "class_id": detection_data["extra"]["class_id"],
                    "class_name": detection_data["cls"],
                    "conf": detection_data["conf"],
                    "bbox_xyxy": [detection_data["x"], detection_data["y"], 
                                detection_data["x"] + detection_data["w"], 
                                detection_data["y"] + detection_data["h"]],
                    "center": detection_data["extra"]["center"],
                    "speed_mph": detection_data["extra"]["speed_mph"],
                    "world_coords": [detection_data["wx"], detection_data["wy"]] if detection_data["wx"] is not None else None,
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
            jsonl_file = io.BytesIO(jsonl_content.encode('utf-8'))
            upload_result = upload_file_to_s3(
                jsonl_file,
                settings.AWS_S3_BUCKET,
                jsonl_key
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
                }
            )
            
            # Stage 7: Mark run as completed
            update_run_status(db=session, run_id=run_uuid, status="completed")
            
            logger.info(f"Successfully processed video for project {project_id}, run {run_id}")
            
            # Clean up temporary files
            temp_video_path.unlink(missing_ok=True)
            
            return {
                "success": True,
                "detection_count": len(detections_data),
                "duration_sec": duration_sec,
                "fps": fps,
            }
            
    except Exception as e:
        logger.error(f"Failed to process video for project {project_id}, run {run_id}: {e}")
        
        # Update run status to failed
        try:
            with Session(engine) as session:
                update_run_status(db=session, run_id=run_uuid, status="failed", error_message=str(e))
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
            update_run_progress(db=session, run_id=run_uuid, stage="downloading_video", percent=10, message="Downloading original video...")
            
            bucket, key = parse_s3_uri(video_asset.uri)
            presigned_url = generate_presigned_url(bucket, key)
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                response = requests.get(presigned_url, stream=True)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    temp_video.write(chunk)
                
                temp_video_path = Path(temp_video.name)
            
            # Step 2: Load detections from database
            update_run_progress(db=session, run_id=run_uuid, stage="loading_detections", percent=20, message="Loading detections from database...")
            
            from src.common.features.processing.crud import get_detections_by_run
            
            detections = get_detections_by_run(db=session, run_id=run_uuid, skip=0, limit=10000)  # Get all detections
            
            if not detections:
                raise ValueError("No detections found for this run")
            
            # Step 3: Load homography data
            update_run_progress(db=session, run_id=run_uuid, stage="loading_homography", percent=30, message="Loading homography data...")
            
            homography_model = homography_session.model
            if not homography_model:
                raise ValueError("Homography model not found")
            
            # Initialize video processor with optimal settings
            processor = VideoProcessor(
                model_path="yolov8s.pt",
                conf_threshold=0.2,
                iou_threshold=0.3,
                classes=[2, 3, 5, 7, 9],  # Vehicle classes
                trail_length=10,
                # Optimal smoothing settings
                bbox_smoothing_method="kalman",
                bbox_smoothing_window=5,
                speed_smoothing_method="moving_average",
                speed_smoothing_window=5,
                tracking_point="bottom_center",
            )
            
            # Create homography data
            homography_data = processor.create_homography_data(homography_session, homography_model)
            homography_path = processor.save_homography_file(homography_data)
            
            # Step 4: Convert detections to JSONL format for VideoAnnotator
            update_run_progress(db=session, run_id=run_uuid, stage="preparing_data", percent=40, message="Preparing detection data...")
            
            # Create temporary JSONL file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as jsonl_file:
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
            
            with open(jsonl_path, 'w') as f:
                for detection in detections:
                    # Calculate real-world event time
                    event_time_real = None
                    if video_start_time:
                        # Convert video time (seconds) to real-world datetime
                        video_time_seconds = detection.t_ms / 1000.0
                        event_time_real = video_start_time + timedelta(seconds=video_time_seconds)
                    
                    # Get bbox coordinates
                    bbox = [detection.x, detection.y, detection.x + detection.w, detection.y + detection.h]
                    
                    # Convert database format to JSONL format expected by VideoAnnotator
                    detection_record = {
                        "video_id": video_asset.uri.split('/')[-1] if video_asset.uri else "video.mp4",
                        "frame": detection.frame_idx,
                        "time": detection.t_ms / 1000.0,
                        "event_time_real": event_time_real.isoformat() if event_time_real else None,
                        "track_id": detection.track_id,
                        "det_idx": 0,  # Not used in annotation
                        "class_id": detection.extra.get("class_id", 0),
                        "class_name": detection.cls,
                        "conf": detection.conf or 0.0,
                        "bbox_xyxy": bbox,
                        "center": detection.extra.get("center", [0, 0]),
                        "speed_mph": detection.extra.get("speed_mph"),
                        "world_coords": [detection.wx, detection.wy] if detection.wx is not None else None,
                        "tracking_point": detection.extra.get("tracking_point", "bottom_center"),
                        "raw_bbox": detection.extra.get("raw_bbox", bbox),
                    }
                    # Convert all NumPy types to Python native types for JSON serialization
                    detection_record = convert_to_json_serializable(detection_record)
                    f.write(json.dumps(detection_record) + "\n")
            
            # Step 5: Use VideoProcessor to render annotated video
            update_run_progress(db=session, run_id=run_uuid, stage="rendering_video", percent=60, message="Rendering annotated video...")
            
            # Create output video path
            output_video_path = Path(tempfile.mktemp(suffix='.mp4'))
            
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
            update_run_progress(db=session, run_id=run_uuid, stage="uploading_video", percent=80, message="Uploading annotated video to S3...")
            
            # Upload annotated video
            video_key = f"projects/{project_id}/runs/{run_id}/annotated_video.mp4"
            with open(output_video_path, 'rb') as video_file:
                upload_result = upload_file_to_s3(
                    video_file,
                    settings.AWS_S3_BUCKET,
                    video_key
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
                }
            )
            
            # Mark task as completed
            update_run_status(db=session, run_id=run_uuid, status="completed")
            
            logger.info(f"Successfully generated annotated video for project {project_id}, run {run_id}")
            
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
        logger.error(f"Failed to generate annotated video for project {project_id}, run {run_id}: {e}")
        
        # Update run status to failed
        try:
            with Session(engine) as session:
                update_run_status(db=session, run_id=run_uuid, status="failed", error_message=str(e))
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


# Example task - replace with your actual tasks
def example_task():
    """Example task that uses the database."""
    with Session(engine) as session:
        # Your task logic here
        pass
