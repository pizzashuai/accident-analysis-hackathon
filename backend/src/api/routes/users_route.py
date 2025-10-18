import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.deps import (
    CurrentUser,
    get_current_active_superuser,
    get_db,
)
from src.common.features.user import (
    Message,
    UpdatePassword,
    UserCreate,
    UserPublic,
    UserRegister,
    UsersPublic,
    UserUpdate,
    UserUpdateMe,
)
from src.common.features.user.crud import (
    create_user,
    delete_user,
    delete_user_me,
    get_user_by_email,
    get_user_by_id,
    list_users,
    update_password_me,
    update_user,
    update_user_me,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "/",
    dependencies=[Depends(get_current_active_superuser)],
    response_model=UsersPublic,
)
def read_users(
    session: Session = Depends(get_db), skip: int = 0, limit: int = 100
) -> UsersPublic:
    """
    Retrieve users.
    """
    try:
        users, count = list_users(session=session, skip=skip, limit=limit)
        return UsersPublic(
            data=[UserPublic.model_validate(user) for user in users], count=count
        )
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/", dependencies=[Depends(get_current_active_superuser)], response_model=UserPublic
)
def create_user_route(
    *, session: Session = Depends(get_db), user_in: UserCreate
) -> UserPublic:
    """
    Create new user.
    """
    try:
        user = create_user(session=session, user_create=user_in)
        return UserPublic.model_validate(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/me", response_model=UserPublic)
def update_user_me_route(
    *,
    session: Session = Depends(get_db),
    user_in: UserUpdateMe,
    current_user: CurrentUser,
) -> UserPublic:
    """
    Update own user.
    """
    try:
        updated_user = update_user_me(
            session=session, current_user=current_user, user_update=user_in
        )
        return UserPublic.model_validate(updated_user)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/me/password", response_model=Message)
def update_password_me_route(
    *,
    session: Session = Depends(get_db),
    body: UpdatePassword,
    current_user: CurrentUser,
) -> Message:
    """
    Update own password.
    """
    try:
        update_password_me(
            session=session, current_user=current_user, password_update=body
        )
        return Message(message="Password updated successfully")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating password: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/me", response_model=UserPublic)
def read_user_me(current_user: CurrentUser) -> UserPublic:
    """
    Get current user.
    """
    return UserPublic.model_validate(current_user)


@router.delete("/me", response_model=Message)
def delete_user_me_route(
    current_user: CurrentUser, session: Session = Depends(get_db)
) -> Message:
    """
    Delete own user.
    """
    try:
        delete_user_me(session=session, current_user=current_user)
        return Message(message="User deleted successfully")
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/signup", response_model=UserPublic)
def register_user(
    user_in: UserRegister, session: Session = Depends(get_db)
) -> UserPublic:
    """
    Create new user without the need to be logged in.
    """
    user = get_user_by_email(session=session, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system",
        )

    try:
        user = create_user(
            session=session,
            user_create=UserCreate(
                email=user_in.email,
                password=user_in.password,
                full_name=user_in.full_name,
            ),
        )
        return UserPublic.model_validate(user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{user_id}", response_model=UserPublic)
def read_user_by_id(
    user_id: uuid.UUID, current_user: CurrentUser, session: Session = Depends(get_db)
) -> UserPublic:
    """
    Get a specific user by id.
    """
    try:
        user = get_user_by_id(session=session, user_id=user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if user.id == current_user.id:
            return UserPublic.model_validate(user)

        if not current_user.is_superuser:
            raise HTTPException(
                status_code=403,
                detail="The user doesn't have enough privileges",
            )

        return UserPublic.model_validate(user)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user by id: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch(
    "/{user_id}",
    dependencies=[Depends(get_current_active_superuser)],
    response_model=UserPublic,
)
def update_user_route(
    *,
    session: Session = Depends(get_db),
    user_id: uuid.UUID,
    user_in: UserUpdate,
) -> UserPublic:
    """
    Update a user.
    """
    try:
        updated_user = update_user(
            session=session, user_id=user_id, user_update=user_in
        )
        return UserPublic.model_validate(updated_user)
    except ValueError as e:
        if "does not exist" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        elif "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{user_id}", dependencies=[Depends(get_current_active_superuser)])
def delete_user_route(
    current_user: CurrentUser, user_id: uuid.UUID, session: Session = Depends(get_db)
) -> Message:
    """
    Delete a user.
    """
    try:
        if user_id == current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Super users are not allowed to delete themselves",
            )

        success = delete_user(session=session, user_id=user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        return Message(message="User deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
