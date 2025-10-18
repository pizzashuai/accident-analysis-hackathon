import uuid
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from src.common.database.models import MediaAsset, Project, ProjectLocation
from src.common.features.project.schemas import (
    ProjectCreate,
    ProjectLocationCreate,
    ProjectUpdate,
)


def create_project(session: Session, project_create: ProjectCreate, user_id: uuid.UUID) -> Project:
    """Create a new project."""
    db_project = Project(
        user_id=user_id,
        title=project_create.title,
        description=project_create.description,
    )
    session.add(db_project)
    session.commit()
    session.refresh(db_project)
    return db_project


def get_project(session: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> Project | None:
    """Get a project by ID, ensuring it belongs to the user."""
    return session.query(Project).filter(
        and_(Project.id == project_id, Project.user_id == user_id)
    ).first()


def list_projects(
    session: Session, user_id: uuid.UUID, skip: int = 0, limit: int = 100
) -> tuple[list[Project], int]:
    """List projects for a user with pagination."""
    query = session.query(Project).filter(Project.user_id == user_id)
    count = query.count()
    projects = query.offset(skip).limit(limit).all()
    return projects, count


def update_project(
    session: Session, project_id: uuid.UUID, user_id: uuid.UUID, project_update: ProjectUpdate
) -> Project | None:
    """Update a project."""
    db_project = get_project(session, project_id, user_id)
    if not db_project:
        return None
    
    update_data = project_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_project, field, value)
    
    session.commit()
    session.refresh(db_project)
    return db_project


def delete_project(session: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    """Delete a project."""
    db_project = get_project(session, project_id, user_id)
    if not db_project:
        return False
    
    session.delete(db_project)
    session.commit()
    return True


def create_media_asset(
    session: Session,
    project_id: uuid.UUID,
    kind: str,
    uri: str,
    bytes: int | None = None,
    meta: dict[str, Any] | None = None,
) -> MediaAsset:
    """Create a media asset."""
    db_asset = MediaAsset(
        project_id=project_id,
        kind=kind,
        uri=uri,
        bytes=bytes,
        meta=meta or {},
    )
    session.add(db_asset)
    session.commit()
    session.refresh(db_asset)
    return db_asset


def upsert_project_location(
    session: Session, project_id: uuid.UUID, location_data: ProjectLocationCreate
) -> ProjectLocation:
    """Create or update project location."""
    db_location = session.query(ProjectLocation).filter(
        ProjectLocation.project_id == project_id
    ).first()
    
    if db_location:
        # Update existing location
        update_data = location_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_location, field, value)
    else:
        # Create new location
        db_location = ProjectLocation(
            project_id=project_id,
            **location_data.model_dump()
        )
        session.add(db_location)
    
    session.commit()
    session.refresh(db_location)
    return db_location


def get_project_with_relations(session: Session, project_id: uuid.UUID, user_id: uuid.UUID) -> Project | None:
    """Get a project with all related data loaded."""
    return session.query(Project).filter(
        and_(Project.id == project_id, Project.user_id == user_id)
    ).first()
