from fastapi import APIRouter
from pydantic import BaseModel

from src.api.deps import SessionDep
from src.common.database.models.user_table import User
from src.common.features.user import UserPublic
from src.common.security import get_password_hash

router = APIRouter(tags=["private"], prefix="/private")


class PrivateUserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    is_verified: bool = False


@router.post("/users/", response_model=UserPublic)
def create_user(user_in: PrivateUserCreate, session: SessionDep) -> UserPublic:
    """
    Create a new user.
    """

    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
    )

    session.add(user)
    session.commit()

    return UserPublic.model_validate(user)
