from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth import CurrentUser, require_permission
from app.core.passwords import hash_password
from app.db.models import Role, User
from app.db.session import get_db
from app.db.user_repository import UserRepository


router = APIRouter(prefix="/admin", tags=["RBAC administration"])


class AdminUserCreate(BaseModel):
    username: str = Field(
        ..., min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=12, max_length=256)
    roles: list[str] = Field(..., min_length=1, max_length=10)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        local, separator, domain = normalized.partition("@")
        if not separator or not local or "." not in domain:
            raise ValueError("Geçerli bir e-posta adresi girin.")
        return normalized

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, value: list[str]) -> list[str]:
        roles = sorted({role.strip().lower() for role in value if role.strip()})
        if not roles:
            raise ValueError("En az bir rol gereklidir.")
        return roles


class AdminUserUpdate(BaseModel):
    roles: list[str] | None = Field(default=None, min_length=1, max_length=10)
    is_active: bool | None = None

    @field_validator("roles")
    @classmethod
    def normalize_roles(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        roles = sorted({role.strip().lower() for role in value if role.strip()})
        if not roles:
            raise ValueError("En az bir rol gereklidir.")
        return roles

    @model_validator(mode="after")
    def require_change(self) -> "AdminUserUpdate":
        if self.roles is None and self.is_active is None:
            raise ValueError("En az bir değişiklik alanı gereklidir.")
        return self


class AdminUserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    roles: list[str]
    permissions: list[str]
    created_at: datetime


class AdminRoleResponse(BaseModel):
    id: int
    name: str
    description: str | None
    permissions: list[str]


def _user_response(user: User) -> AdminUserResponse:
    return AdminUserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        roles=UserRepository.role_names(user),
        permissions=UserRepository.permission_codes(user),
        created_at=user.created_at,
    )


def _role_response(role: Role) -> AdminRoleResponse:
    return AdminRoleResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        permissions=sorted(permission.code for permission in role.permissions),
    )


@router.get("/users", response_model=list[AdminUserResponse])
def list_users(
    _: CurrentUser = Depends(require_permission("users:manage")),
    db: Session = Depends(get_db),
) -> list[AdminUserResponse]:
    return [_user_response(user) for user in UserRepository(db).list_users()]


@router.post(
    "/users", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED
)
def create_user(
    body: AdminUserCreate,
    _: CurrentUser = Depends(require_permission("users:manage")),
    db: Session = Depends(get_db),
) -> AdminUserResponse:
    repository = UserRepository(db)
    try:
        user = repository.create_user(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
            role_names=body.roles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Kullanıcı adı veya e-posta zaten kullanımda.",
        ) from exc
    return _user_response(user)


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
def update_user_access(
    user_id: str,
    body: AdminUserUpdate,
    current_user: CurrentUser = Depends(require_permission("users:manage")),
    db: Session = Depends(get_db),
) -> AdminUserResponse:
    repository = UserRepository(db)
    user = repository.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    if user.id == current_user.id and body.is_active is False:
        raise HTTPException(status_code=400, detail="Kendi hesabınızı kapatamazsınız.")
    if (
        user.id == current_user.id
        and body.roles is not None
        and "admin" not in body.roles
    ):
        raise HTTPException(
            status_code=400, detail="Kendi admin rolünüzü kaldıramazsınız."
        )

    try:
        updated_user = repository.update_access(
            user, role_names=body.roles, is_active=body.is_active
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _user_response(updated_user)


@router.get("/roles", response_model=list[AdminRoleResponse])
def list_roles(
    _: CurrentUser = Depends(require_permission("roles:manage")),
    db: Session = Depends(get_db),
) -> list[AdminRoleResponse]:
    return [_role_response(role) for role in UserRepository(db).list_roles()]
