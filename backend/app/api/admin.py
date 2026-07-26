from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.auth import CurrentUser, require_permission
from app.core.passwords import hash_password
from app.db.models import Role, TeamLocation, User
from app.db.player_context_repository import PlayerContextRepository
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


class AdminTeamLocationUpsert(BaseModel):
    data_source: str = Field(
        default="api_football",
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    team_id: int = Field(..., gt=0)
    name: str = Field(..., min_length=1, max_length=100)
    latitude: float | None = Field(default=None, ge=-90, le=90, allow_inf_nan=False)
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        allow_inf_nan=False,
    )

    @field_validator("team_id", "latitude", "longitude", mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Boolean değer sayısal alanlarda kullanılamaz.")
        return value

    @field_validator("data_source", mode="before")
    @classmethod
    def normalize_data_source(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Değer boş olamaz.")
        return normalized

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Değer boş olamaz.")
        return normalized

    @model_validator(mode="after")
    def require_coordinate_pair(self) -> "AdminTeamLocationUpsert":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Enlem ve boylam birlikte sağlanmalıdır.")
        return self


class AdminTeamLocationBatch(BaseModel):
    locations: list[AdminTeamLocationUpsert] = Field(
        ...,
        min_length=1,
        max_length=500,
    )


class AdminTeamLocationResponse(BaseModel):
    data_source: str
    team_id: int
    name: str
    latitude: float | None
    longitude: float | None


class AdminTeamLocationUpsertResponse(BaseModel):
    processed: int = Field(..., ge=0)


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


def _team_location_response(location: TeamLocation) -> AdminTeamLocationResponse:
    return AdminTeamLocationResponse(
        data_source=location.data_source,
        team_id=location.team_id,
        name=location.name,
        latitude=location.latitude,
        longitude=location.longitude,
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


@router.get(
    "/team-locations",
    response_model=list[AdminTeamLocationResponse],
)
def list_team_locations(
    data_source: str | None = Query(
        default=None,
        min_length=1,
        max_length=50,
        pattern=r"^[A-Za-z0-9_.-]+$",
    ),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _: CurrentUser = Depends(require_permission("users:manage")),
    db: Session = Depends(get_db),
) -> list[AdminTeamLocationResponse]:
    """List travel-coordinate inputs managed by an administrator."""
    return [
        _team_location_response(location)
        for location in PlayerContextRepository(db).list_team_locations(
            data_source=data_source,
            limit=limit,
            offset=offset,
        )
    ]


@router.put(
    "/team-locations",
    response_model=AdminTeamLocationUpsertResponse,
)
def upsert_team_locations(
    body: AdminTeamLocationBatch,
    _: CurrentUser = Depends(require_permission("users:manage")),
    db: Session = Depends(get_db),
) -> AdminTeamLocationUpsertResponse:
    """Validate and atomically upsert team base coordinates used for travel load."""
    repository = PlayerContextRepository(db)
    processed = repository.upsert_team_locations(
        [location.model_dump() for location in body.locations]
    )
    return AdminTeamLocationUpsertResponse(processed=processed)
