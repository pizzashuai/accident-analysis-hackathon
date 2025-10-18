# Import all SQLAlchemy models to ensure they are registered with the metadata
from src.common.database.models.user_table import Base, User
from src.common.database.models.project_table import Project
from src.common.database.models.media_asset_table import MediaAsset
from src.common.database.models.project_location_table import ProjectLocation
from src.common.database.models.homography_session_table import HomographySession
from src.common.database.models.homography_pair_table import HomographyPair
from src.common.database.models.homography_model_table import HomographyModel

# Export the Base metadata for alembic
__all__ = [
    "Base",
    "User",
    "Project",
    "MediaAsset",
    "ProjectLocation",
    "HomographySession",
    "HomographyPair",
    "HomographyModel",
]
