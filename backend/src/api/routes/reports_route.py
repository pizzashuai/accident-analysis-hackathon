#!/usr/bin/env python3
"""
Report API routes for PDF report generation and retrieval.
"""

import logging
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.deps import CurrentUser, get_db
from src.common.database.models.project_table import Project
from src.common.database.models.report_table import Report
from src.common.features.report.crud import (
    create_report,
    get_report,
    list_reports_by_project,
    get_report_by_analysis_id,
)
from src.common.features.storage import generate_presigned_url, parse_s3_uri
from src.worker.celery_app.tasks import generate_pdf_report_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])


class GenerateReportRequest(BaseModel):
    """Request model for generating a PDF report."""
    
    analysis_id: str
    run_id: str


class GenerateReportResponse(BaseModel):
    """Response model for generating a PDF report."""
    
    report_id: str
    status: str
    message: str


class ReportResponse(BaseModel):
    """Response model for report data."""
    
    id: str
    project_id: str
    run_id: str
    analysis_id: str
    status: str
    pdf_uri: str | None
    meta: dict
    created_at: str
    completed_at: str | None


class ReportListResponse(BaseModel):
    """Response model for report list."""
    
    reports: List[ReportResponse]
    total: int


@router.post("/projects/{project_id}/generate", response_model=GenerateReportResponse)
def generate_report(
    project_id: str,
    request: GenerateReportRequest,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> GenerateReportResponse:
    """
    Generate a PDF report from LLM analysis results.
    
    This endpoint:
    1. Validates project ownership
    2. Validates the processing run and analysis
    3. Creates a report record
    4. Triggers a Celery task for PDF generation
    5. Returns the report ID
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

        # Validate processing run exists and belongs to project
        from src.common.features.processing.crud import get_processing_run
        
        run = get_processing_run(db=session, run_id=run_uuid)
        if not run:
            raise HTTPException(status_code=404, detail="Processing run not found")

        if run.project_id != project_uuid:
            raise HTTPException(
                status_code=400, detail="Processing run does not belong to this project"
            )

        # Validate run is completed
        if run.status != "completed":
            raise HTTPException(
                status_code=400,
                detail="Processing run must be completed before generating reports",
            )

        # Check if report already exists for this analysis
        existing_report = get_report_by_analysis_id(session, request.analysis_id)
        if existing_report:
            raise HTTPException(
                status_code=400,
                detail=f"Report already exists for analysis {request.analysis_id}",
            )

        # Create report record
        report = create_report(
            db=session,
            project_id=project_uuid,
            run_id=run_uuid,
            analysis_id=request.analysis_id,
            meta={
                "requested_by": str(current_user.id),
                "project_title": project.title,
                "run_status": run.status,
            }
        )

        # Trigger Celery task for PDF generation
        generate_pdf_report_task.delay(
            report_id=str(report.id),
            project_id=project_id,
            run_id=request.run_id,
            analysis_id=request.analysis_id,
        )

        logger.info(
            f"Started PDF generation for report {report.id} in project {project_id}"
        )

        return GenerateReportResponse(
            report_id=str(report.id),
            status="pending",
            message="PDF report generation started successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/projects/{project_id}", response_model=ReportListResponse)
def list_reports(
    project_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
) -> ReportListResponse:
    """
    List all reports for a project.
    
    Returns reports with presigned URLs for PDF downloads.
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    try:
        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Get reports
        reports = list_reports_by_project(
            db=session, project_id=project_uuid, skip=skip, limit=limit
        )

        # Convert to response format with presigned URLs
        report_responses = []
        for report in reports:
            pdf_url = None
            if report.pdf_uri:
                try:
                    bucket, key = parse_s3_uri(report.pdf_uri)
                    pdf_url = generate_presigned_url(bucket, key)
                except Exception as e:
                    logger.warning(f"Could not generate presigned URL for report {report.id}: {e}")

            report_responses.append(
                ReportResponse(
                    id=str(report.id),
                    project_id=str(report.project_id),
                    run_id=str(report.run_id) if report.run_id else "",
                    analysis_id=report.analysis_id,
                    status=report.status,
                    pdf_uri=pdf_url,
                    meta=report.meta,
                    created_at=report.created_at.isoformat(),
                    completed_at=report.completed_at.isoformat() if report.completed_at else None,
                )
            )

        return ReportListResponse(reports=report_responses, total=len(report_responses))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/projects/{project_id}/{report_id}", response_model=ReportResponse)
def get_report_details(
    project_id: str,
    report_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
) -> ReportResponse:
    """
    Get specific report details.
    
    Returns report with presigned PDF URL.
    """
    try:
        project_uuid = uuid.UUID(project_id)
        report_uuid = uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    try:
        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Get report
        report = get_report(db=session, report_id=report_uuid)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        if report.project_id != project_uuid:
            raise HTTPException(
                status_code=400, detail="Report does not belong to this project"
            )

        # Generate presigned URL for PDF
        pdf_url = None
        if report.pdf_uri:
            try:
                bucket, key = parse_s3_uri(report.pdf_uri)
                pdf_url = generate_presigned_url(bucket, key)
            except Exception as e:
                logger.warning(f"Could not generate presigned URL for report {report.id}: {e}")

        return ReportResponse(
            id=str(report.id),
            project_id=str(report.project_id),
            run_id=str(report.run_id) if report.run_id else "",
            analysis_id=report.analysis_id,
            status=report.status,
            pdf_uri=pdf_url,
            meta=report.meta,
            created_at=report.created_at.isoformat(),
            completed_at=report.completed_at.isoformat() if report.completed_at else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report details: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/projects/{project_id}/{report_id}/download")
def download_report(
    project_id: str,
    report_id: str,
    current_user: CurrentUser,
    session: Session = Depends(get_db),
):
    """
    Download PDF report directly.
    
    Redirects to presigned S3 URL for direct download.
    """
    try:
        project_uuid = uuid.UUID(project_id)
        report_uuid = uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    try:
        # Get project and validate ownership
        project = session.get(Project, project_uuid)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized")

        # Get report
        report = get_report(db=session, report_id=report_uuid)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        if report.project_id != project_uuid:
            raise HTTPException(
                status_code=400, detail="Report does not belong to this project"
            )

        if not report.pdf_uri:
            raise HTTPException(status_code=404, detail="PDF not available")

        # Generate presigned URL and redirect
        bucket, key = parse_s3_uri(report.pdf_uri)
        presigned_url = generate_presigned_url(bucket, key)

        return RedirectResponse(url=presigned_url, status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
