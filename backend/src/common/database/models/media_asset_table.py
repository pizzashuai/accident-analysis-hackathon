import uuid
from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database.models.user_table import Base


class MediaAsset(Base):
    __tablename__ = "media_asset"
    model_config = {"from_attributes": True}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("project.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # 'video', 'image', 'map_snapshot', 'json'
    uri: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    meta: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    is_processing: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Relationships
    project = relationship("Project", foreign_keys=[project_id], back_populates="media_assets")
    project_video = relationship("Project", foreign_keys="Project.video_id", back_populates="video")
