from sqlalchemy import Column, DateTime, ForeignKey, String, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.common.database.models.user_table import Base


class LLMAnalysis(Base):
    __tablename__ = "llm_analysis"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("processing_run.id", ondelete="SET NULL"), nullable=True)
    analysis_id = Column(String, nullable=False, unique=True)  # Session ID for SSE streaming
    status = Column(String, nullable=False, default="pending")  # pending, analyzing, completed, failed
    result_data = Column(JSONB, nullable=True)  # Complete raw analysis result from LLM agent
    track_ids = Column(JSONB, nullable=True)  # Track IDs analyzed (array)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="llm_analyses")
    run = relationship("ProcessingRun", back_populates="llm_analyses")
    reports = relationship("Report", back_populates="llm_analysis")
    
    # Indexes
    __table_args__ = (
        Index("idx_llm_analysis_project_id", "project_id"),
        Index("idx_llm_analysis_run_id", "run_id"),
        Index("idx_llm_analysis_analysis_id", "analysis_id"),
        Index("idx_llm_analysis_status", "status"),
    )
