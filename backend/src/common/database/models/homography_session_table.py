import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database.models.user_table import Base


class HomographySession(Base):
    __tablename__ = "homography_session"
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
    screenshot_asset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("media_asset.id"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="draft",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    solved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    project = relationship("Project", back_populates="homography_session")
    screenshot_asset = relationship("MediaAsset", foreign_keys=[screenshot_asset_id])
    pairs = relationship(
        "HomographyPair",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="HomographyPair.order_idx",
    )
    model = relationship(
        "HomographyModel",
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
    )
