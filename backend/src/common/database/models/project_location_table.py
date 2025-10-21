import uuid

from sqlalchemy import Double, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database.models.user_table import Base


class ProjectLocation(Base):
    __tablename__ = "project_location"
    model_config = {"from_attributes": True}

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("project.id", ondelete="CASCADE"),
        primary_key=True,
    )
    addr_line: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )
    lat: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
    )
    lon: Mapped[float | None] = mapped_column(
        Double,
        nullable=True,
    )
    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="user",
    )

    # Relationships
    project = relationship("Project", back_populates="location", passive_deletes=True)
