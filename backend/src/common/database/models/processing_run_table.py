from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.common.database.models.user_table import Base

class ProcessingRun(Base):
    __tablename__ = "processing_run"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    homography_session_id = Column(UUID(as_uuid=True), ForeignKey("homography_session.id"))
    params = Column(JSONB, nullable=False, default={})
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    progress = Column(JSONB, nullable=False, default={})  # {stage: str, percent: int, message: str}
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    finished_at = Column(DateTime)
    error_message = Column(String)
    
    # Relationships
    project = relationship("Project", back_populates="processing_runs")
    detections = relationship("Detection", back_populates="run", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="run", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="run", cascade="all, delete-orphan")
    llm_analyses = relationship("LLMAnalysis", back_populates="run")
