import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import cv2

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
from src.common.features.process_video import process_video_with_supervision, VideoAnnotator
from src.common.features.process_video.src.estimate_distance import DistanceEstimator
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
            
            # Create homography file for DistanceEstimator (expects "pairs" format)
            pairs_data = []
            logger.info(f"Found {len(homography_session.pairs)} homography pairs")
            
            for idx, pair in enumerate(homography_session.pairs):
                logger.info(f"Pair {idx}: image({pair.image_x_norm:.4f}, {pair.image_y_norm:.4f}) -> geo({pair.map_lat:.6f}, {pair.map_lng:.6f})")
                pairs_data.append({
                    "id": idx,
                    "a": {
                        "xNorm": pair.image_x_norm,
                        "yNorm": pair.image_y_norm
                    },
                    "b": {
                        "lat": pair.map_lat,
                        "lng": pair.map_lng
                    }
                })
            
            homography_data = {
                "pairs": pairs_data,
                "imagesMeta": homography_model.meta.get("imagesMeta", {}) if homography_model.meta else {},
                "mapMeta": homography_model.meta.get("mapMeta", {}) if homography_model.meta else {},
            }
            
            logger.info(f"Created homography data with {len(pairs_data)} pairs")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as homography_file:
                json.dump(homography_data, homography_file)
                homography_path = Path(homography_file.name)
            
            # Stage 3: Run YOLO + ByteTrack processing
            update_run_progress(db=session, run_id=run_uuid, stage="detecting", percent=30, message="Running YOLO detection and ByteTrack tracking...")
            
            # Create output directory
            output_dir = Path(tempfile.mkdtemp())
            
            # Process video (without annotation for now)
            process_video_with_supervision(
                video_path=temp_video_path,
                test_folder=output_dir,
                model_path="yolov8s.pt",  # Default model
                conf_threshold=0.2,
                iou_threshold=0.3,
                classes=[2, 3, 5, 7, 9],  # Vehicle classes
                trail_length=10,
                annotate_video=False,  # Skip annotation for now
                homography_file=str(homography_path),
            )
            
            # Stage 4: Load detections and calculate speeds
            update_run_progress(db=session, run_id=run_uuid, stage="calculating_speeds", percent=70, message="Calculating speeds and preparing data...")
            
            # Initialize distance estimator for speed calculation
            logger.info(f"Initializing DistanceEstimator with homography file: {homography_path}")
            distance_estimator = DistanceEstimator(str(homography_path))
            logger.info(f"DistanceEstimator initialized successfully with {len(distance_estimator.homography_data.pairs)} point pairs")
            
            # Read detections from JSONL
            jsonl_path = output_dir / "detections.jsonl"
            detections_data = []
            
            # Track vehicle positions for speed calculation
            vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
            vehicle_speeds = {}    # {track_id: current_speed_mph}
            
            with open(jsonl_path, 'r') as f:
                for line in f:
                    detection = json.loads(line.strip())
                    
                    # Convert to database format
                    bbox = detection["bbox_xyxy"]
                    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                    
                    # Calculate world coordinates using homography
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # Convert to normalized coordinates (0-1 range)
                    x_norm = center_x / width
                    y_norm = center_y / height
                    
                    # Transform to world coordinates using homography
                    try:
                        geo_point = distance_estimator.image_to_geo(x_norm, y_norm)
                        wx, wy = geo_point.lng, geo_point.lat
                    except Exception as e:
                        logger.warning(f"Failed to transform coordinates: {e}")
                        wx, wy = None, None
                    
                    # Calculate speed if we have tracking data
                    speed_mph = None
                    track_id = detection.get("track_id")
                    if track_id is not None:
                        # Initialize tracking history for this vehicle
                        if track_id not in vehicle_positions:
                            vehicle_positions[track_id] = []
                        
                        # Add current position
                        timestamp = detection["time"]
                        vehicle_positions[track_id].append((detection["frame"], x_norm, y_norm, timestamp))
                        
                        # Keep only last 30 frames of history
                        if len(vehicle_positions[track_id]) > 30:
                            vehicle_positions[track_id] = vehicle_positions[track_id][-30:]
                        
                        # Calculate speed if we have enough history (use 5 frames for smoothing)
                        history = vehicle_positions[track_id]
                        if len(history) >= 5:
                            old_frame, old_x, old_y, old_time = history[-5]
                            new_frame, new_x, new_y, new_time = history[-1]
                            
                            # Calculate distance using homography
                            try:
                                distance_meters = distance_estimator.estimate_distance(
                                    (old_x, old_y), (new_x, new_y)
                                )
                                
                                # Calculate time difference
                                time_diff = new_time - old_time
                                
                                if time_diff > 0:
                                    # Calculate speed in meters per second
                                    speed_mps = distance_meters / time_diff
                                    
                                    # Convert to miles per hour (1 m/s = 2.23694 mph)
                                    speed_mph = speed_mps * 2.23694
                                    
                                    # Store current speed for this vehicle
                                    vehicle_speeds[track_id] = speed_mph
                                    
                                    # Debug logging for first few speed calculations
                                    if len(detections_data) < 10:
                                        logger.info(f"Speed calculated for track {track_id}: {speed_mph:.2f} mph (distance: {distance_meters:.2f}m, time: {time_diff:.3f}s)")
                                    
                            except Exception as e:
                                logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
                                speed_mph = vehicle_speeds.get(track_id)  # Use previous speed if available
                    
                    detection_data = {
                        "project_id": project_uuid,
                        "frame_idx": detection["frame"],
                        "t_ms": int(detection["time"] * 1000),
                        "track_id": track_id,
                        "cls": detection["class_name"],
                        "conf": detection["conf"],
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "wx": wx,
                        "wy": wy,
                        "extra": {
                            "speed_mph": speed_mph,
                            "class_id": detection["class_id"],
                            "center": detection["center"],
                        }
                    }
                    detections_data.append(detection_data)
            
            # Debug logging for detections data
            logger.info(f"Processed {len(detections_data)} detections from JSONL")
            speed_count = sum(1 for d in detections_data if d["extra"].get("speed_mph") is not None)
            logger.info(f"Detections with speed data: {speed_count}/{len(detections_data)}")
            
            # Stage 5: Bulk insert detections to DB
            update_run_progress(db=session, run_id=run_uuid, stage="saving_detections", percent=85, message="Saving detections to database...")
            
            bulk_insert_detections(db=session, run_id=run_uuid, detections_list=detections_data)
            
            # Stage 6: Update JSONL file with speed data and upload to S3
            update_run_progress(db=session, run_id=run_uuid, stage="uploading_artifacts", percent=90, message="Updating JSONL with speed data and uploading to S3...")
            
            # Update JSONL file with speed data
            updated_jsonl_path = output_dir / "detections_with_speed.jsonl"
            speed_updates_count = 0
            total_detections = 0
            
            with open(jsonl_path, 'r') as input_file, open(updated_jsonl_path, 'w') as output_file:
                for line in input_file:
                    detection = json.loads(line.strip())
                    total_detections += 1
                    
                    # Find matching detection in our processed data
                    matching_detection = None
                    for det_data in detections_data:
                        if (det_data["frame_idx"] == detection["frame"] and 
                            det_data["track_id"] == detection.get("track_id")):
                            matching_detection = det_data
                            break
                    
                    # Add speed data if found
                    if matching_detection and matching_detection["extra"].get("speed_mph") is not None:
                        detection["speed_mph"] = matching_detection["extra"]["speed_mph"]
                        speed_updates_count += 1
                        
                        # Debug logging for first few speed updates
                        if speed_updates_count <= 5:
                            logger.info(f"Updated detection frame {detection['frame']} track {detection.get('track_id')} with speed {detection['speed_mph']:.2f} mph")
                    
                    output_file.write(json.dumps(detection) + "\n")
            
            logger.info(f"JSONL update complete: {speed_updates_count}/{total_detections} detections updated with speed data")
            
            # Upload updated JSONL file
            jsonl_key = f"projects/{project_id}/runs/{run_id}/detections.jsonl"
            with open(updated_jsonl_path, 'rb') as jsonl_file:
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
            homography_path.unlink(missing_ok=True)
            
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
            homography_path.unlink(missing_ok=True)
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
    4. Use VideoAnnotator to render annotated video
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
            
            # Create homography file for DistanceEstimator (expects "pairs" format)
            pairs_data = []
            for idx, pair in enumerate(homography_session.pairs):
                pairs_data.append({
                    "id": idx,
                    "a": {
                        "xNorm": pair.image_x_norm,
                        "yNorm": pair.image_y_norm
                    },
                    "b": {
                        "lat": pair.map_lat,
                        "lng": pair.map_lng
                    }
                })
            
            homography_data = {
                "pairs": pairs_data,
                "imagesMeta": homography_model.meta.get("imagesMeta", {}) if homography_model.meta else {},
                "mapMeta": homography_model.meta.get("mapMeta", {}) if homography_model.meta else {},
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as homography_file:
                json.dump(homography_data, homography_file)
                homography_path = Path(homography_file.name)
            
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
            
            # Initialize distance estimator for speed calculation
            distance_estimator = DistanceEstimator(str(homography_path))
            
            # Track vehicle positions for speed calculation
            vehicle_positions = {}  # {track_id: [(frame, x_norm, y_norm, timestamp), ...]}
            vehicle_speeds = {}    # {track_id: current_speed_mph}
            
            # Convert database detections to JSONL format
            with open(jsonl_path, 'w') as f:
                for detection in detections:
                    # Calculate center point for tracking
                    center_x = detection.x + detection.w / 2
                    center_y = detection.y + detection.h / 2
                    
                    # Convert to normalized coordinates (0-1 range)
                    x_norm = center_x / video_width
                    y_norm = center_y / video_height
                    
                    # Calculate speed if we have tracking data
                    speed_mph = detection.extra.get("speed_mph")  # Use existing speed if available
                    track_id = detection.track_id
                    
                    if track_id is not None and speed_mph is None:
                        # Initialize tracking history for this vehicle
                        if track_id not in vehicle_positions:
                            vehicle_positions[track_id] = []
                        
                        # Add current position
                        timestamp = detection.t_ms / 1000.0
                        vehicle_positions[track_id].append((detection.frame_idx, x_norm, y_norm, timestamp))
                        
                        # Keep only last 30 frames of history
                        if len(vehicle_positions[track_id]) > 30:
                            vehicle_positions[track_id] = vehicle_positions[track_id][-30:]
                        
                        # Calculate speed if we have enough history (use 5 frames for smoothing)
                        history = vehicle_positions[track_id]
                        if len(history) >= 5:
                            old_frame, old_x, old_y, old_time = history[-5]
                            new_frame, new_x, new_y, new_time = history[-1]
                            
                            # Calculate distance using homography
                            try:
                                distance_meters = distance_estimator.estimate_distance(
                                    (old_x, old_y), (new_x, new_y)
                                )
                                
                                # Calculate time difference
                                time_diff = new_time - old_time
                                
                                if time_diff > 0:
                                    # Calculate speed in meters per second
                                    speed_mps = distance_meters / time_diff
                                    
                                    # Convert to miles per hour (1 m/s = 2.23694 mph)
                                    speed_mph = speed_mps * 2.23694
                                    
                                    # Store current speed for this vehicle
                                    vehicle_speeds[track_id] = speed_mph
                                    
                            except Exception as e:
                                logger.warning(f"Failed to calculate speed for track {track_id}: {e}")
                                speed_mph = vehicle_speeds.get(track_id)  # Use previous speed if available
                    
                    # Convert database format to JSONL format expected by VideoAnnotator
                    detection_record = {
                        "video_id": video_asset.uri.split('/')[-1] if video_asset.uri else "video.mp4",
                        "frame": detection.frame_idx,
                        "time": detection.t_ms / 1000.0,
                        "track_id": detection.track_id,
                        "det_idx": 0,  # Not used in annotation
                        "class_id": detection.extra.get("class_id", 0),
                        "class_name": detection.cls,
                        "conf": detection.conf or 0.0,
                        "bbox_xyxy": [
                            detection.x,
                            detection.y,
                            detection.x + detection.w,
                            detection.y + detection.h
                        ],
                        "center": detection.extra.get("center", [detection.x + detection.w/2, detection.y + detection.h/2]),
                        "speed_mph": speed_mph,
                        "world_coords": [detection.wx, detection.wy] if detection.wx is not None else None,
                    }
                    f.write(json.dumps(detection_record) + "\n")
            
            # Step 5: Use VideoAnnotator to render annotated video
            update_run_progress(db=session, run_id=run_uuid, stage="rendering_video", percent=60, message="Rendering annotated video...")
            
            # Create output video path
            output_video_path = Path(tempfile.mktemp(suffix='.mp4'))
            
            # Initialize VideoAnnotator with optimal settings for speed calculation
            annotator = VideoAnnotator(
                trail_length=10,
                homography_file=str(homography_path),  # Enable speed calculation
                bbox_smoothing="kalman",  # Use Kalman filter for best speed stability
                bbox_smoothing_window=5,
                speed_smoothing="moving_average",  # Use moving average for best speed smoothing
                smoothing_window=5,
                tracking_point="bottom_center",  # Use bottom center for more stable tracking
                debug_speed=False,  # Disable debug output for production
            )
            
            # Render annotated video
            annotator.annotate_video_from_jsonl(
                original_video_path=temp_video_path,
                jsonl_path=jsonl_path,
                output_path=output_video_path,
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
