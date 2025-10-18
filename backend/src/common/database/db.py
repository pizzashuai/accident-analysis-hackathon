from collections.abc import Generator

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import settings

# Import moved to avoid circular import
from src.common.database.models.user_table import User

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# make sure all SQLAlchemy models are imported (backend_database.models) before initializing DB
# otherwise, SQLAlchemy might fail to initialize relationships properly


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def init_db(session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next lines
    # from .models import Base
    # Base.metadata.create_all(engine)

    user = session.execute(
        select(User).where(User.email == settings.FIRST_SUPERUSER)
    ).scalar_one_or_none()
    if not user:
        from src.common.features.user import UserCreate, create_user

        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = create_user(session=session, user_create=user_in)
