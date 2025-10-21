import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database.models.user_table import Base


class Project(Base):
    __tablename__ = "project"
    model_config = {"from_attributes": True}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("media_asset.id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationships
    user = relationship("User", back_populates="projects")
    video = relationship("MediaAsset", foreign_keys=[video_id], back_populates="project_video")
    media_assets = relationship(
        "MediaAsset",
        foreign_keys="MediaAsset.project_id",
        back_populates="project",
        passive_deletes=True,
    )
    location = relationship("ProjectLocation", back_populates="project", uselist=False, passive_deletes=True)
    homography_session = relationship("HomographySession", back_populates="project", uselist=False)
    processing_runs = relationship("ProcessingRun", back_populates="project", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="project", cascade="all, delete-orphan")
    llm_analyses = relationship("LLMAnalysis", back_populates="project", cascade="all, delete-orphan")
