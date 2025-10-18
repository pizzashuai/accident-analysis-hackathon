from sqlalchemy import Column, DateTime, ForeignKey, String, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from src.common.database.models.user_table import Base

class Artifact(Base):
    __tablename__ = "artifact"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("processing_run.id", ondelete="SET NULL"))
    kind = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    meta = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    run = relationship("ProcessingRun", back_populates="artifacts")
    
    __table_args__ = (
        CheckConstraint(
            "kind IN ('jsonl_detections', 'csv_detections', 'annotated_video', 'report', 'debug')",
            name="artifact_kind_check"
        ),
    )
