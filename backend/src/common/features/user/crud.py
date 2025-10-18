import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.common.database.models.user_table import User
from src.common.features.user.schemas import (
    UpdatePassword,
    UserCreate,
    UserUpdate,
    UserUpdateMe,
)
from src.common.security import get_password_hash, verify_password


def create_user(*, session: Session, user_create: UserCreate) -> User:
    """
    Create a new user in the database and return the ORM User instance.
    """
    # Check if user already exists
    existing_user = get_user_by_email(session=session, email=user_create.email)
    if existing_user:
        raise ValueError("The user with this email already exists in the system.")

    # Create user with hashed password
    user_data = user_create.model_dump()
    user_data["id"] = uuid.uuid4()
    user_data["hashed_password"] = get_password_hash(user_data.pop("password"))
    db_obj = User(**user_data)
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(
    *, session: Session, user_id: uuid.UUID, user_update: UserUpdate
) -> User:
    """
    Update an existing user in the database and return the ORM User instance.
    """
    db_user = session.get(User, str(user_id))
    if not db_user:
        raise ValueError("The user with this id does not exist in the system")

    # Check email uniqueness if email is being updated
    if user_update.email and user_update.email != db_user.email:
        existing_user = get_user_by_email(session=session, email=user_update.email)
        if existing_user and existing_user.id != user_id:
            raise ValueError("User with this email already exists")

    # Apply updates
    update_data = user_update.model_dump(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    for field, value in update_data.items():
        setattr(db_user, field, value)

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get_user_by_id(*, session: Session, user_id: uuid.UUID) -> User | None:
    """
    Get a user by ID.
    """
    db_user = session.get(User, str(user_id))
    if not db_user:
        return None
    return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    """
    Get a user by email.
    """
    statement = select(User).where(User.email == email)
    db_user = session.execute(statement).scalar_one_or_none()
    if not db_user:
        return None
    return db_user


def list_users(
    *, session: Session, skip: int = 0, limit: int = 100
) -> tuple[list[User], int]:
    """
    List users with pagination and return ORM User instances with total count.
    """
    count = session.execute(select(func.count()).select_from(User)).scalar_one()

    users = (
        session.execute(
            select(User)
            .order_by(User.id)  # ensure deterministic pagination
            .offset(skip)
            .limit(limit)
        )
        .scalars()
        .all()
    )

    return list(users), count


def delete_user(*, session: Session, user_id: uuid.UUID) -> bool:
    """
    Delete a user by ID. Returns True if user was deleted, False if not found.
    """
    db_user = session.get(User, str(user_id))
    if not db_user:
        return False

    session.delete(db_user)
    session.commit()
    return True


def authenticate_user(*, session: Session, email: str, password: str) -> User | None:
    """
    Authenticate a user by email and password. Returns the ORM User instance if successful.
    """
    user = get_user_by_email(session=session, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def update_user_me(
    *, session: Session, current_user: User, user_update: UserUpdateMe
) -> User:
    """
    Update own user profile.
    """
    # Check email uniqueness if email is being updated
    if user_update.email and user_update.email != current_user.email:
        existing_user = get_user_by_email(session=session, email=user_update.email)
        if existing_user and existing_user.id != current_user.id:
            raise ValueError("User with this email already exists")

    # Get the database user
    db_user = session.get(User, str(current_user.id))
    if not db_user:
        raise ValueError("User not found")

    # Apply updates
    update_data = user_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_user, field, value)

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def update_password_me(
    *, session: Session, current_user: User, password_update: UpdatePassword
) -> None:
    """
    Update own password.
    """
    # Verify current password
    if not verify_password(
        password_update.current_password, current_user.hashed_password
    ):
        raise ValueError("Incorrect password")

    # Check if new password is different
    if password_update.current_password == password_update.new_password:
        raise ValueError("New password cannot be the same as the current one")

    # Get the database user and update password
    db_user = session.get(User, str(current_user.id))
    if not db_user:
        raise ValueError("User not found")
    db_user.hashed_password = get_password_hash(password_update.new_password)
    session.add(db_user)
    session.commit()


def delete_user_me(*, session: Session, current_user: User) -> None:
    """
    Delete own user.
    """
    if current_user.is_superuser:
        raise ValueError("Super users are not allowed to delete themselves")

    success = delete_user(session=session, user_id=current_user.id)
    if not success:
        raise ValueError("User not found")
