"""
CRUD operations for Report model.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.common.database.models.report_table import Report


def create_report(
    db: Session,
    project_id: uuid.UUID,
    run_id: uuid.UUID,
    analysis_id: str,
    meta: dict = None
) -> Report:
    """
    Create a new report record.
    
    Args:
        db: Database session
        project_id: Project UUID
        run_id: Processing run UUID
        analysis_id: LLM analysis session ID
        meta: Additional metadata
        
    Returns:
        Created Report instance
    """
    if meta is None:
        meta = {}
    
    report = Report(
        project_id=project_id,
        run_id=run_id,
        analysis_id=analysis_id,
        status="pending",
        meta=meta
    )
    
    db.add(report)
    db.commit()
    db.refresh(report)
    
    return report


def get_report(db: Session, report_id: uuid.UUID) -> Optional[Report]:
    """
    Get a report by ID.
    
    Args:
        db: Database session
        report_id: Report UUID
        
    Returns:
        Report instance or None
    """
    return db.get(Report, report_id)


def list_reports_by_project(
    db: Session, 
    project_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100
) -> List[Report]:
    """
    List all reports for a project.
    
    Args:
        db: Database session
        project_id: Project UUID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of Report instances
    """
    return (
        db.query(Report)
        .filter(Report.project_id == project_id)
        .order_by(desc(Report.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_report_status(
    db: Session,
    report_id: uuid.UUID,
    status: str,
    error_message: Optional[str] = None
) -> Optional[Report]:
    """
    Update report status.
    
    Args:
        db: Database session
        report_id: Report UUID
        status: New status (pending/generating/completed/failed)
        error_message: Error message if status is failed
        
    Returns:
        Updated Report instance or None
    """
    report = db.get(Report, report_id)
    if not report:
        return None
    
    report.status = status
    
    if status == "completed":
        report.completed_at = datetime.utcnow()
    elif status == "failed" and error_message:
        report.meta = {**report.meta, "error_message": error_message}
    
    db.commit()
    db.refresh(report)
    
    return report


def update_report_pdf_uri(
    db: Session,
    report_id: uuid.UUID,
    pdf_uri: str,
    meta_updates: Optional[dict] = None
) -> Optional[Report]:
    """
    Update report with PDF URI and additional metadata.
    
    Args:
        db: Database session
        report_id: Report UUID
        pdf_uri: S3 URI for the generated PDF
        meta_updates: Additional metadata to merge
        
    Returns:
        Updated Report instance or None
    """
    report = db.get(Report, report_id)
    if not report:
        return None
    
    report.pdf_uri = pdf_uri
    report.status = "completed"
    report.completed_at = datetime.utcnow()
    
    if meta_updates:
        report.meta = {**report.meta, **meta_updates}
    
    db.commit()
    db.refresh(report)
    
    return report


def get_report_by_analysis_id(
    db: Session,
    analysis_id: str
) -> Optional[Report]:
    """
    Get report by analysis ID.
    
    Args:
        db: Database session
        analysis_id: LLM analysis session ID
        
    Returns:
        Report instance or None
    """
    return (
        db.query(Report)
        .filter(Report.analysis_id == analysis_id)
        .first()
    )
