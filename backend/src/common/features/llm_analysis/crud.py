"""
CRUD operations for LLMAnalysis model.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.common.database.models.llm_analysis_table import LLMAnalysis


def create_analysis(
    db: Session,
    project_id: uuid.UUID,
    run_id: Optional[uuid.UUID],
    analysis_id: str,
    track_ids: Optional[List[int]] = None
) -> LLMAnalysis:
    """
    Create a new LLM analysis record.
    
    Args:
        db: Database session
        project_id: Project UUID
        run_id: Processing run UUID (optional)
        analysis_id: Session ID for SSE streaming
        track_ids: List of track IDs to analyze
        
    Returns:
        Created LLMAnalysis instance
    """
    analysis = LLMAnalysis(
        project_id=project_id,
        run_id=run_id,
        analysis_id=analysis_id,
        status="pending",
        track_ids=track_ids or []
    )
    
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    return analysis


def get_analysis(db: Session, analysis_id: str) -> Optional[LLMAnalysis]:
    """
    Get an analysis by session ID.
    
    Args:
        db: Database session
        analysis_id: Session ID
        
    Returns:
        LLMAnalysis instance or None
    """
    return (
        db.query(LLMAnalysis)
        .filter(LLMAnalysis.analysis_id == analysis_id)
        .first()
    )


def get_analysis_by_id(db: Session, id: uuid.UUID) -> Optional[LLMAnalysis]:
    """
    Get an analysis by UUID.
    
    Args:
        db: Database session
        id: Analysis UUID
        
    Returns:
        LLMAnalysis instance or None
    """
    return db.get(LLMAnalysis, id)


def list_analyses_by_project(
    db: Session, 
    project_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100
) -> List[LLMAnalysis]:
    """
    List all analyses for a project.
    
    Args:
        db: Database session
        project_id: Project UUID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of LLMAnalysis instances
    """
    return (
        db.query(LLMAnalysis)
        .filter(LLMAnalysis.project_id == project_id)
        .order_by(desc(LLMAnalysis.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def list_analyses_by_run(
    db: Session, 
    run_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100
) -> List[LLMAnalysis]:
    """
    List all analyses for a processing run.
    
    Args:
        db: Database session
        run_id: Processing run UUID
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of LLMAnalysis instances
    """
    return (
        db.query(LLMAnalysis)
        .filter(LLMAnalysis.run_id == run_id)
        .order_by(desc(LLMAnalysis.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_analysis_status(
    db: Session,
    analysis_id: str,
    status: str
) -> Optional[LLMAnalysis]:
    """
    Update analysis status.
    
    Args:
        db: Database session
        analysis_id: Session ID
        status: New status (pending/analyzing/completed/failed)
        
    Returns:
        Updated LLMAnalysis instance or None
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        return None
    
    analysis.status = status
    
    if status == "completed":
        analysis.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(analysis)
    
    return analysis


def update_analysis_result(
    db: Session,
    analysis_id: str,
    result_data: Dict[str, Any]
) -> Optional[LLMAnalysis]:
    """
    Update analysis with complete result data.
    
    Args:
        db: Database session
        analysis_id: Session ID
        result_data: Complete analysis result from LLM agent
        
    Returns:
        Updated LLMAnalysis instance or None
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        return None
    
    analysis.result_data = result_data
    analysis.status = "completed"
    analysis.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(analysis)
    
    return analysis


def update_analysis_error(
    db: Session,
    analysis_id: str,
    error_message: str
) -> Optional[LLMAnalysis]:
    """
    Update analysis with error information.
    
    Args:
        db: Database session
        analysis_id: Session ID
        error_message: Error message
        
    Returns:
        Updated LLMAnalysis instance or None
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        return None
    
    analysis.status = "failed"
    analysis.error_message = error_message
    analysis.completed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(analysis)
    
    return analysis


def delete_analysis(db: Session, analysis_id: str) -> bool:
    """
    Delete an analysis (for reset functionality).
    
    Args:
        db: Database session
        analysis_id: Session ID
        
    Returns:
        True if deleted, False if not found
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        return False
    
    db.delete(analysis)
    db.commit()
    
    return True


def get_analyses_by_status(
    db: Session,
    status: str,
    skip: int = 0,
    limit: int = 100
) -> List[LLMAnalysis]:
    """
    Get analyses by status.
    
    Args:
        db: Database session
        status: Analysis status
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List of LLMAnalysis instances
    """
    return (
        db.query(LLMAnalysis)
        .filter(LLMAnalysis.status == status)
        .order_by(desc(LLMAnalysis.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
