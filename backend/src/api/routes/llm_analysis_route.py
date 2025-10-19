#!/usr/bin/env python3
"""
LLM Analysis API routes for streaming accident analysis.

This module provides endpoints for starting LLM analysis and streaming
real-time events via Server-Sent Events (SSE).
"""

import json
import logging
import tempfile
import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import CurrentUser, get_db
from src.common.config import settings
from src.common.database.models.artifact_table import Artifact
from src.common.database.models.project_table import Project
from src.common.features.storage import (
    generate_presigned_url,
    parse_s3_uri,
)
from src.worker.celery_app.tasks import analyze_accident_llm_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-analysis", tags=["llm-analysis"])


class StartAnalysisRequest(BaseModel):
    """Request model for starting LLM analysis."""

    run_id: str


class StartAnalysisResponse(BaseModel):
    """Response model for starting LLM analysis."""

    analysis_id: str
    status: str
    message: str


def get_redis_client():
    """Get Redis client for event streaming."""
    redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(redis_url, decode_responses=True)


@router.post("/projects/{project_id}/start", response_model=StartAnalysisResponse)
def start_llm_analysis(
    project_id: str,
    request: StartAnalysisRequest,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> StartAnalysisResponse:
    """
    Start LLM analysis for a project's processing run.

    This endpoint:
    1. Validates project ownership
    2. Finds the latest completed processing run
    3. Locates the filtered JSONL artifact
    4. Downloads the artifact to a temporary file
    5. Starts a Celery task for LLM analysis
    6. Returns analysis_id for SSE streaming
    """
    try:
        project_uuid = uuid.UUID(project_id)
        run_uuid = uuid.UUID(request.run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID format: {e}")

    try:
        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Get processing run
        from src.common.features.processing.crud import get_processing_run

        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")

        # Validate run belongs to project
        if run.project_id != project_uuid:
            raise HTTPException(
                status_code=400, detail="Processing run does not belong to this project"
            )

        # Validate run is completed
        if run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Processing run must be completed before LLM analysis",
            )

        # Find filtered JSONL artifact for this run
        # Look for artifacts with kind="jsonl_detections" and filtered_track_ids in meta
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
            raise HTTPException(
                status_code=400,
                detail="No filtered detections found. Please filter detections by track IDs first.",
            )

        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())

        # Download filtered JSONL artifact to temporary file
        bucket, key = parse_s3_uri(filtered_artifact.uri)
        presigned_url = generate_presigned_url(bucket, key)

        # Create temporary file for the detections
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        temp_file_path = Path(temp_file.name)
        temp_file.close()

        try:
            # Download file content
            import requests

            response = requests.get(presigned_url)
            response.raise_for_status()

            # Write to temporary file
            with open(temp_file_path, "w") as f:
                f.write(response.text)

            # Start Celery task
            analyze_accident_llm_task.delay(
                analysis_id=analysis_id,
                project_id=project_id,
                run_id=request.run_id,
                detections_file_path=str(temp_file_path),
            )

            logger.info(
                f"Started LLM analysis {analysis_id} for project {project_id}, run {request.run_id}"
            )

            return StartAnalysisResponse(
                analysis_id=analysis_id,
                status="started",
                message="LLM analysis started successfully",
            )

        except Exception as e:
            # Clean up temporary file on error
            if temp_file_path.exists():
                temp_file_path.unlink(missing_ok=True)
            raise e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting LLM analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.options("/projects/{project_id}/stream/{analysis_id}")
def options_llm_analysis_stream(
    project_id: str,
    analysis_id: str,
):
    """Handle preflight OPTIONS request for SSE endpoint."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:5173",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "Cache-Control, Authorization, Content-Type",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
        },
    )


@router.get("/projects/{project_id}/stream/{analysis_id}")
def stream_llm_analysis_events(
    project_id: str,
    analysis_id: str,
    token: str = Query(None, description="Authentication token for SSE connection"),
    session: Session = Depends(get_db),
):
    """
    Stream LLM analysis events via Server-Sent Events (SSE).

    This endpoint:
    1. Validates project ownership using token authentication
    2. Subscribes to Redis channel for the analysis_id
    3. Streams events in SSE format to the frontend
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    try:
        # Authenticate user using token from query parameter
        if not token:
            raise HTTPException(status_code=401, detail="Authentication token required")
        
        # Validate token and get user
        from src.api.deps import get_current_user
        from fastapi import Request
        from fastapi.security import OAuth2PasswordBearer
        
        # Create a mock request object for token validation
        class MockRequest:
            def __init__(self, token):
                self.headers = {"authorization": f"Bearer {token}"}
        
        # Validate token
        try:
            current_user = get_current_user(session, token)
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid authentication token")

        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Get Redis client
        redis_client = get_redis_client()

        # Create Redis pubsub for this analysis
        pubsub = redis_client.pubsub()
        channel = f"llm_analysis:{analysis_id}"
        pubsub.subscribe(channel)

        def event_generator():
            """Generate SSE events from Redis messages."""
            try:
                # Send initial connection event
                logger.info(f"Starting SSE stream for analysis {analysis_id}")
                yield "event: connected\n"
                yield f"data: {json.dumps({'message': 'Connected to analysis stream'})}\n\n"

                # Listen for events
                for message in pubsub.listen():
                    logger.debug(f"Received Redis message: {message}")
                    if message["type"] == "message":
                        try:
                            event_data = json.loads(message["data"])
                            event_type = event_data.get("type", "unknown")
                            data = event_data.get("data", {})

                            logger.info(f"Streaming event {event_type} with data: {data}")

                            # Format as SSE
                            yield f"event: {event_type}\n"
                            yield f"data: {json.dumps(data)}\n\n"

                            # Check if analysis is complete
                            if event_type in ["report_end", "error"]:
                                logger.info(f"Analysis complete, ending stream. Event: {event_type}")
                                # Small delay to ensure the event is flushed to client
                                import time
                                time.sleep(0.1)
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Invalid JSON in Redis message: {message['data']}, error: {e}"
                            )
                            continue
                    elif message["type"] == "subscribe":
                        logger.info(f"Subscribed to channel: {channel}")

            except Exception as e:
                logger.error(f"Error in event generator: {e}")
                yield "event: error\n"
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                logger.info(f"Closing SSE stream for analysis {analysis_id}")
                pubsub.close()
                redis_client.close()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "http://localhost:5173",  # React Router dev server default port
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Headers": "Cache-Control, Authorization, Content-Type",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming LLM analysis events: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
