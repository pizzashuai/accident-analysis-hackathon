from sqlalchemy import Column, DateTime, ForeignKey, String, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.common.database.models.user_table import Base


class Report(Base):
    __tablename__ = "report"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("processing_run.id", ondelete="SET NULL"))
    llm_analysis_id = Column(UUID(as_uuid=True), ForeignKey("llm_analysis.id", ondelete="SET NULL"), nullable=True)
    status = Column(String, nullable=False, default="pending")
    pdf_uri = Column(String, nullable=True)  # S3 URI for generated PDF
    meta = Column(JSONB, nullable=False, default={})  # Analysis metadata, screenshots info
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="reports")
    run = relationship("ProcessingRun", back_populates="reports")
    llm_analysis = relationship("LLMAnalysis", back_populates="reports")
    
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'generating', 'completed', 'failed')",
            name="report_status_check"
        ),
    )
