import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.common.database.models.homography_session_table import HomographySession
from src.common.database.models.homography_pair_table import HomographyPair
from src.common.database.models.homography_model_table import HomographyModel
from src.common.database.models.project_table import Project
from src.common.features.homography.schemas import (
    HomographyPairCreate,
    HomographySessionCreate,
    HomographySessionUpdate,
)
from src.common.features.homography.solver import solve_homography_from_pairs


def create_session(db: Session, project_id: uuid.UUID) -> HomographySession:
    """Create a new homography session for a project"""
    session = HomographySession(
        project_id=project_id,
        status="draft"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: uuid.UUID) -> Optional[HomographySession]:
    """Get a homography session by ID"""
    return db.get(HomographySession, session_id)


def get_or_create_session_for_project(db: Session, project_id: uuid.UUID) -> HomographySession:
    """Get existing session for project or create new one"""
    # Check if project exists
    project = db.get(Project, project_id)
    if not project:
        raise ValueError(f"Project {project_id} not found")
    
    # Look for existing session
    stmt = select(HomographySession).where(HomographySession.project_id == project_id)
    session = db.execute(stmt).scalar_one_or_none()
    
    if session:
        return session
    
    # Create new session
    return create_session(db, project_id)


def add_pair(db: Session, session_id: uuid.UUID, pair_data: HomographyPairCreate) -> HomographyPair:
    """Add a point pair to a homography session"""
    # Verify session exists
    session = get_session(db, session_id)
    if not session:
        raise ValueError(f"Homography session {session_id} not found")
    
    pair = HomographyPair(
        session_id=session_id,
        image_x_norm=pair_data.image_x_norm,
        image_y_norm=pair_data.image_y_norm,
        map_lat=pair_data.map_lat,
        map_lng=pair_data.map_lng,
        order_idx=pair_data.order_idx
    )
    
    db.add(pair)
    db.commit()
    db.refresh(pair)
    
    # Update session status to draft if it was solved
    if session.status == "solved":
        session.status = "draft"
        session.solved_at = None
        db.commit()
    
    return pair


def update_pairs(db: Session, session_id: uuid.UUID, pairs_list: List[HomographyPairCreate]) -> List[HomographyPair]:
    """Replace all pairs in a session (bulk update)"""
    # Verify session exists
    session = get_session(db, session_id)
    if not session:
        raise ValueError(f"Homography session {session_id} not found")
    
    # Delete existing pairs
    db.query(HomographyPair).filter(HomographyPair.session_id == session_id).delete()
    
    # Add new pairs
    new_pairs = []
    for i, pair_data in enumerate(pairs_list):
        pair = HomographyPair(
            session_id=session_id,
            image_x_norm=pair_data.image_x_norm,
            image_y_norm=pair_data.image_y_norm,
            map_lat=pair_data.map_lat,
            map_lng=pair_data.map_lng,
            order_idx=pair_data.order_idx if pair_data.order_idx != 0 else i
        )
        new_pairs.append(pair)
        db.add(pair)
    
    db.commit()
    
    # Refresh all pairs
    for pair in new_pairs:
        db.refresh(pair)
    
    # Update session status to draft if it was solved
    if session.status == "solved":
        session.status = "draft"
        session.solved_at = None
        db.commit()
    
    return new_pairs


def delete_pair(db: Session, pair_id: uuid.UUID) -> bool:
    """Delete a specific point pair"""
    pair = db.get(HomographyPair, pair_id)
    if not pair:
        return False
    
    session_id = pair.session_id
    db.delete(pair)
    db.commit()
    
    # Update session status to draft if it was solved
    session = get_session(db, session_id)
    if session and session.status == "solved":
        session.status = "draft"
        session.solved_at = None
        db.commit()
    
    return True


def solve_homography(db: Session, session_id: uuid.UUID) -> HomographyModel:
    """Solve homography for a session and store the result"""
    # Get session with pairs
    session = db.get(HomographySession, session_id)
    if not session:
        raise ValueError(f"Homography session {session_id} not found")
    
    # Get pairs
    pairs = db.query(HomographyPair).filter(
        HomographyPair.session_id == session_id
    ).order_by(HomographyPair.order_idx).all()
    
    if len(pairs) < 4:
        raise ValueError("At least 4 point pairs are required for homography calculation")
    
    # Solve homography
    result = solve_homography_from_pairs(pairs)
    
    # Delete existing model if any
    existing_model = db.query(HomographyModel).filter(
        HomographyModel.session_id == session_id
    ).first()
    if existing_model:
        db.delete(existing_model)
    
    # Create new model
    model = HomographyModel(
        session_id=session_id,
        matrix_data=result.matrix,
        reprojection_error=result.reprojection_error,
        meta={
            "inlier_count": result.inlier_count,
            "status": result.status,
            "total_pairs": len(pairs)
        }
    )
    
    db.add(model)
    
    # Update session status
    session.status = "solved"
    session.solved_at = datetime.utcnow()
    
    db.commit()
    db.refresh(model)
    
    return model


def get_session_with_relations(db: Session, session_id: uuid.UUID) -> Optional[HomographySession]:
    """Get session with all related data (pairs and model)"""
    session = db.get(HomographySession, session_id)
    if not session:
        return None
    
    # Load pairs
    pairs = db.query(HomographyPair).filter(
        HomographyPair.session_id == session_id
    ).order_by(HomographyPair.order_idx).all()
    session.pairs = pairs
    
    # Load model
    model = db.query(HomographyModel).filter(
        HomographyModel.session_id == session_id
    ).first()
    session.model = model
    
    return session


def export_homography_data(db: Session, session_id: uuid.UUID) -> dict:
    """Export homography data in process-video compatible format"""
    session = get_session_with_relations(db, session_id)
    if not session:
        raise ValueError(f"Homography session {session_id} not found")
    
    # Convert pairs to process-video format
    pairs_data = []
    for i, pair in enumerate(session.pairs):
        pairs_data.append({
            "id": i + 1,
            "a": {
                "xNorm": pair.image_x_norm,
                "yNorm": pair.image_y_norm
            },
            "b": {
                "lat": pair.map_lat,
                "lng": pair.map_lng
            }
        })
    
    # Create metadata
    images_meta = {
        "width": 1920,  # Default values, could be extracted from actual image
        "height": 1080,
        "format": "image/jpeg"
    }
    
    map_meta = {
        "projection": "WGS84",
        "bounds": {
            "north": max(pair.map_lat for pair in session.pairs) if session.pairs else 0,
            "south": min(pair.map_lat for pair in session.pairs) if session.pairs else 0,
            "east": max(pair.map_lng for pair in session.pairs) if session.pairs else 0,
            "west": min(pair.map_lng for pair in session.pairs) if session.pairs else 0,
        }
    }
    
    return {
        "pairs": pairs_data,
        "imagesMeta": images_meta,
        "mapMeta": map_meta
    }
