from sqlalchemy import Column, Integer, Float, ForeignKey, String, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from src.common.database.models.user_table import Base

class Detection(Base):
    __tablename__ = "detection"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("project.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey("processing_run.id", ondelete="SET NULL"))
    frame_idx = Column(Integer, nullable=False)
    t_ms = Column(Integer, nullable=False)
    track_id = Column(Integer)
    cls = Column(String, nullable=False)
    conf = Column(Float)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    w = Column(Float, nullable=False)
    h = Column(Float, nullable=False)
    wx = Column(Float)  # World x (homography-transformed)
    wy = Column(Float)  # World y (homography-transformed)
    extra = Column(JSONB, nullable=False, default={})  # speed_mph, geo coords, etc.
    
    # Relationships
    run = relationship("ProcessingRun", back_populates="detections")
    
    __table_args__ = (
        Index("detection_project_time_idx", "project_id", "t_ms"),
        Index("detection_project_track_idx", "project_id", "track_id"),
        Index("detection_run_idx", "run_id"),
        Index("detection_frame_idx", "frame_idx"),
    )
