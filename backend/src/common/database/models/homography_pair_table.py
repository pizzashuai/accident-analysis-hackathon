import uuid

from sqlalchemy import Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.common.database.models.user_table import Base


class HomographyPair(Base):
    __tablename__ = "homography_pair"
    model_config = {"from_attributes": True}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("homography_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    image_x_norm: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    image_y_norm: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    map_lat: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    map_lng: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    order_idx: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # Relationships
    session = relationship("HomographySession", back_populates="pairs")
