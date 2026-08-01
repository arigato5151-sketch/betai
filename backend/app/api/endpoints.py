import asyncio
import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import logging

from app.db.session import SessionLocal, get_db
from app.db.repository import MatchPredictionRepository
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import PlayerContextRepository
from app.db.models import HistoricalFixture
from app.services.api_football import APIFootballClient
from app.services.data_quality import DataQualityService
from app.services.external_features import external_feature_service
from app.services.odds_history import odds_history_service
from app.services.sportmonks_players import sportmonks_player_service
from app.services.travel_context import travel_context_service
from app.prediction.stats_engine import StatsEngine
from app.prediction.ensemble import ProbabilityEnsembler
from app.prediction.value_calc import ValueCalc
from app.prediction.ml.model import ml_pipeline
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import (
    HistoricalFeatureContext,
    HistoricalFeatureService,
    PlayerRatingValue,
)
from app.prediction.player_impact import PlayerImpactCalculator
from app.prediction.input_catalog import AnalysisInputCatalog
from app.prediction.ml.explain import ExplainabilityService
from app.prediction.ml.active_learning import ActiveLearningSelector
from app.prediction.audit import PredictionAuditor
from app.prediction.backtest import BacktestEngine
from app.core.allowed_leagues import ALLOWED_LEAGUES
from app.tasks.health import enqueue_retraining
from app.core.config import settings
from app.core.passwords import hash_password
from app.core.rate_limit import login_rate_limiter
from app.db.user_repository import UserRepository
from app.core.api_mode import ApiMode, get_api_mode
from app.core.auth import (
    LoginRequest,
    CurrentUser,
    SessionResponse,
    authenticate_user,
    clear_auth_cookies,
    current_user_from_model,
    issue_token_pair,
    require_authenticated_user,
    require_permission,
    revoke_refresh_token,
    rotate_refresh_token,
    set_auth_cookies,
)
from app.api.admin import router as admin_router

logger = logging.getLogger("bet-ai-pro.api")
router = APIRouter()
football_api = APIFootballClient()
router.include_router(admin_router)


class PlatformStatusResponse(BaseModel):
    api_mode: ApiMode
    registration_enabled: bool


class RegistrationRequest(BaseModel):
    username: str = Field(
        ..., min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=12, max_length=256)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        local, separator, domain = normalized.partition("@")
        if not separator or not local or "." not in domain:
            raise ValueError("Geçerli bir e-posta adresi girin.")
        return normalized


@router.get("/status", response_model=PlatformStatusResponse)
def get_platform_status() -> PlatformStatusResponse:
    return PlatformStatusResponse(
        api_mode=get_api_mode(settings.API_FOOTBALL_KEY),
        registration_enabled=settings.ALLOW_SELF_REGISTRATION,
    )


@router.post(
    "/auth/register",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    body: RegistrationRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    if not settings.ALLOW_SELF_REGISTRATION:
        raise HTTPException(status_code=403, detail="Kullanıcı kaydı devre dışı.")
    try:
        user = UserRepository(db).create_user(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
            role_names=[settings.SELF_REGISTRATION_ROLE],
        )
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Kullanıcı adı veya e-posta zaten kullanımda.",
        ) from exc
    tokens = issue_token_pair(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    set_auth_cookies(response, tokens)
    return SessionResponse(
        user=current_user_from_model(user),
        expires_in=tokens.expires_in,
    )


@router.post("/auth/login", response_model=SessionResponse)
def login(
    body: LoginRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
):
    ip_address = request.client.host if request.client else "unknown"
    retry_after = login_rate_limiter.retry_after(body.username, ip_address)
    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail="Çok fazla başarısız giriş denemesi.",
            headers={"Retry-After": str(retry_after)},
        )
    user = authenticate_user(db, body.username, body.password)
    if not user:
        login_rate_limiter.record_failure(body.username, ip_address)
        raise HTTPException(status_code=401, detail="Kullanıcı adı veya parola hatalı.")
    login_rate_limiter.reset(body.username, ip_address)
    tokens = issue_token_pair(
        db,
        user,
        user_agent=request.headers.get("user-agent"),
        ip_address=ip_address,
    )
    set_auth_cookies(response, tokens)
    return SessionResponse(
        user=current_user_from_model(user),
        expires_in=tokens.expires_in,
    )


@router.post("/auth/refresh", response_model=SessionResponse)
def refresh_access_token(
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
    refresh_token: str | None = Cookie(
        default=None, alias=settings.REFRESH_TOKEN_COOKIE_NAME
    ),
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Refresh çerezi bulunamadı.")
    user, tokens = rotate_refresh_token(
        db,
        refresh_token,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    set_auth_cookies(response, tokens)
    return SessionResponse(
        user=current_user_from_model(user),
        expires_in=tokens.expires_in,
    )


@router.get("/auth/session", response_model=SessionResponse)
def get_session(user: CurrentUser = Depends(require_authenticated_user)):
    return SessionResponse(
        user=user,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/auth/sessions")
def list_auth_sessions(
    user: CurrentUser = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    sessions = UserRepository(db).list_active_sessions(user.id)
    return [
        {
            "id": session.id,
            "created_at": session.created_at,
            "expires_at": session.expires_at,
            "user_agent": session.user_agent,
            "ip_address": session.ip_address,
        }
        for session in sessions
    ]


@router.delete("/auth/sessions/{session_id}", status_code=204)
def revoke_auth_session(
    session_id: str,
    user: CurrentUser = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    revoked = UserRepository(db).revoke_session_by_id(
        user.id, session_id, datetime.now(timezone.utc)
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="Aktif oturum bulunamadı.")


@router.post("/auth/logout", status_code=204)
def logout(
    response: Response,
    db: Session = Depends(get_db),
    refresh_token: str | None = Cookie(
        default=None, alias=settings.REFRESH_TOKEN_COOKIE_NAME
    ),
):
    revoke_refresh_token(db, refresh_token)
    clear_auth_cookies(response)


class TeamStatsInput(BaseModel):
    form: float = Field(..., ge=0, le=100, description="Team form (0-100)")
    attack: float = Field(..., ge=0, le=100, description="Attack strength (0-100)")
    defense: float = Field(..., ge=0, le=100, description="Defense strength (0-100)")
    xg: float = Field(..., ge=0, le=5, description="Expected goals (0-5)")

    @field_validator("form", "attack", "defense")
    @classmethod
    def validate_stats(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("Must be numeric")
        return round(float(v), 2)

    @field_validator("xg")
    @classmethod
    def validate_xg(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("xG must be numeric")
        return round(float(v), 3)


class Odds1X2Input(BaseModel):
    """Validated decimal 1X2 snapshot using public API outcome aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    home_win: float = Field(..., alias="HOME_WIN", gt=1.0, le=1000.0)
    draw: float = Field(..., alias="DRAW", gt=1.0, le=1000.0)
    away_win: float = Field(..., alias="AWAY_WIN", gt=1.0, le=1000.0)

    def as_outcome_dict(self) -> Dict[str, float]:
        return {
            "HOME_WIN": self.home_win,
            "DRAW": self.draw,
            "AWAY_WIN": self.away_win,
        }


class AnalysisRequest(BaseModel):
    home_team: str = Field(
        ..., min_length=1, max_length=100, description="Home team name"
    )
    away_team: str = Field(
        ..., min_length=1, max_length=100, description="Away team name"
    )
    home_stats: TeamStatsInput
    away_stats: TeamStatsInput
    odd: float = Field(..., gt=1.0, le=1000.0, description="Betting odd (>1.0)")
    market_1x2: Optional[Dict[str, Any]] = None
    opening_odds_1x2: Optional[Odds1X2Input] = None
    current_odds_1x2: Optional[Odds1X2Input] = None
    opening_odds_at: Optional[datetime] = None
    current_odds_at: Optional[datetime] = None
    fixture_id: Optional[int] = Field(None, gt=0, description="API Football fixture ID")
    home_team_id: Optional[int] = Field(
        None, gt=0, description="API Football home team ID"
    )
    away_team_id: Optional[int] = Field(
        None, gt=0, description="API Football away team ID"
    )
    league_id: Optional[int] = Field(None, gt=0, description="API Football league ID")
    season: Optional[int] = Field(
        None, ge=2000, le=2100, description="API Football league season"
    )
    kickoff: Optional[datetime] = Field(
        None, description="Fixture kickoff used for point-in-time rest features"
    )
    away_travel_distance_km: Optional[float] = Field(
        None,
        ge=0,
        le=20000,
        allow_inf_nan=False,
        description=(
            "Optional away-team base-to-venue distance. Server-side team locations "
            "are used when omitted."
        ),
    )
    feature_overrides: Dict[str, float] = Field(default_factory=dict)

    @field_validator("home_team", "away_team")
    @classmethod
    def validate_team_names(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Team name cannot be empty")
        return v.strip()[:100]

    @field_validator("odd")
    @classmethod
    def validate_odd(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 1.0:
            raise ValueError("Odd must be numeric and > 1.0")
        return round(float(v), 3)

    @field_validator("feature_overrides", mode="before")
    @classmethod
    def validate_feature_overrides(cls, value: object) -> Dict[str, float]:
        return AnalysisInputCatalog.validate_overrides(value)

    @model_validator(mode="after")
    def validate_odds_snapshot_timeline(self) -> "AnalysisRequest":
        snapshots = (
            ("opening", self.opening_odds_1x2, self.opening_odds_at),
            ("current", self.current_odds_1x2, self.current_odds_at),
        )
        for label, odds, captured_at in snapshots:
            if (odds is None) != (captured_at is None):
                raise ValueError(
                    f"{label}_odds_1x2 and {label}_odds_at must be provided together"
                )

        supplied_timestamps = [
            captured_at for _, _, captured_at in snapshots if captured_at is not None
        ]
        if not supplied_timestamps:
            return self
        if self.kickoff is None:
            raise ValueError("kickoff is required when odds snapshots are provided")
        if self.kickoff.utcoffset() is None:
            raise ValueError("kickoff must include a timezone for odds validation")
        if any(captured_at.utcoffset() is None for captured_at in supplied_timestamps):
            raise ValueError("odds snapshot timestamps must include a timezone")

        kickoff_utc = self.kickoff.astimezone(timezone.utc)
        if any(
            captured_at.astimezone(timezone.utc) >= kickoff_utc
            for captured_at in supplied_timestamps
        ):
            raise ValueError("odds snapshots must be captured before kickoff")
        if (
            self.opening_odds_at is not None
            and self.current_odds_at is not None
            and self.opening_odds_at.astimezone(timezone.utc)
            > self.current_odds_at.astimezone(timezone.utc)
        ):
            raise ValueError("opening odds timestamp cannot follow current odds")
        return self


class ActualResultUpdate(BaseModel):
    actual_result: str = Field(
        ..., pattern="^(HOME_WIN|DRAW|AWAY_WIN)$", description="Match result"
    )
    actual_score_home: Optional[int] = Field(
        None, ge=0, le=50, description="Home team goals"
    )
    actual_score_away: Optional[int] = Field(
        None, ge=0, le=50, description="Away team goals"
    )

    @field_validator("actual_result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        if v not in {"HOME_WIN", "DRAW", "AWAY_WIN"}:
            raise ValueError("Result must be HOME_WIN, DRAW, or AWAY_WIN")
        return v


class BacktestRequest(BaseModel):
    initial_bankroll: float = Field(
        1000.0, gt=0, le=1000000, description="Starting bankroll"
    )
    strategy: str = Field(
        "kelly",
        pattern="^(kelly|flat|fractional_kelly)$",
        description="Betting strategy",
    )
    flat_stake_amount: float = Field(10.0, gt=0, le=100000, description="Flat bet size")
    kelly_fraction: float = Field(
        0.25, gt=0, le=1.0, description="Kelly fraction (0-1)"
    )
    min_edge_pct: float = Field(3.0, ge=0, le=100, description="Minimum edge% to bet")
    commission_pct: float = Field(
        0.0, ge=0, le=20, description="Commission deducted from winning profit"
    )
    max_stake_pct: float = Field(
        5.0, gt=0, le=100, description="Maximum bankroll percentage per bet"
    )
    max_daily_exposure_pct: float = Field(
        15.0, gt=0, le=100, description="Maximum daily bankroll exposure"
    )
    require_closing_odds: bool = Field(
        False, description="Skip records without a valid closing price"
    )
    exclude_post_kickoff: bool = Field(
        True, description="Exclude analyses generated at or after kickoff"
    )

    @field_validator("initial_bankroll")
    @classmethod
    def validate_bankroll(cls, v: float) -> float:
        if v < 10:
            raise ValueError("Bankroll must be at least 10")
        return round(float(v), 2)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        valid = {"kelly", "flat", "fractional_kelly"}
        if v not in valid:
            raise ValueError(f"Strategy must be one of {valid}")
        return v


def _build_payload_from_prefill(prefill: Dict[str, Any]) -> AnalysisRequest:
    fixture = prefill.get("fixture") or {}
    raw_market = prefill.get("market_1x2")
    market = raw_market if isinstance(raw_market, dict) else {}
    return AnalysisRequest(
        home_team=prefill["home_team"],
        away_team=prefill["away_team"],
        home_stats=TeamStatsInput(**prefill["home_stats"]),
        away_stats=TeamStatsInput(**prefill["away_stats"]),
        odd=prefill["odd"],
        market_1x2=market or None,
        opening_odds_1x2=prefill.get("opening_odds_1x2"),
        current_odds_1x2=prefill.get("current_odds_1x2"),
        opening_odds_at=prefill.get("opening_odds_at"),
        current_odds_at=prefill.get("current_odds_at"),
        fixture_id=fixture.get("fixture_id"),
        home_team_id=fixture.get("home_team_id"),
        away_team_id=fixture.get("away_team_id"),
        league_id=fixture.get("league_id"),
        season=fixture.get("season"),
        kickoff=fixture.get("kickoff"),
        away_travel_distance_km=fixture.get("away_travel_distance_km"),
    )


def _ml_safety_label(ml_result: dict) -> str:
    if not ml_result.get("ready"):
        return "INSUFFICIENT_DATA"
    if ml_result.get("prediction") == "HOME_WIN":
        return "HIGH_CONFIDENCE"
    return "RISKY_UNDERDOG"


def _ml_cluster_value(ml_result: dict) -> int:
    return 1 if ml_result.get("prediction") == "HOME_WIN" else 0


def _build_analysis_response(
    record_id: int,
    home_team: str,
    away_team: str,
    analysis: dict,
    value_data: dict,
    ml_result: dict,
    insights: List[str],
    labeled_samples_count: int,
) -> dict:
    prediction_labels = {
        "HOME_WIN": "Ev Sahibi Galibiyeti",
        "AWAY_WIN": "Deplasman Galibiyeti",
        "DRAW": "Beraberlik",
    }

    return {
        "id": record_id,
        "match": f"{home_team} vs {away_team}",
        "analysis": {
            **analysis,
            "prediction_label": prediction_labels.get(
                analysis["prediction"], analysis["prediction"]
            ),
        },
        "value_assessment": value_data,
        "ml_safety_trigger": _ml_safety_label(ml_result),
        "ml_confidence": (
            ml_result.get("probability", 0.0) if ml_result.get("ready") else 0.0
        ),
        "ml_ready": ml_result.get("ready", False),
        "ml_samples": labeled_samples_count,
        "ml_min_samples": settings.MIN_TRAINING_SAMPLES,
        "labeled_samples_count": labeled_samples_count,
        "remaining_to_threshold": max(
            0, settings.MIN_TRAINING_SAMPLES - labeled_samples_count
        ),
        "insights": insights,
    }


async def _fetch_ml_match_data(
    payload: AnalysisRequest, historical: HistoricalFeatureContext
) -> Tuple[Any, Any, Any, Any, Any]:
    """Fill missing point-in-time sources from API without holding a DB session."""
    home_team_id = payload.home_team_id
    away_team_id = payload.away_team_id
    if home_team_id is None or away_team_id is None:
        return historical.home_matches_df, historical.away_matches_df, None, None, None

    def local_history_is_usable(local_frame: Any) -> bool:
        if (
            local_frame is None
            or len(local_frame) < settings.RECENT_FORM_MATCH_COUNT
            or payload.kickoff is None
            or "match_date" not in local_frame.columns
        ):
            return False
        latest = local_frame["match_date"].max()
        latest_datetime = (
            latest.to_pydatetime() if hasattr(latest, "to_pydatetime") else latest
        )
        if latest_datetime.tzinfo is None:
            latest_datetime = latest_datetime.replace(tzinfo=timezone.utc)
        kickoff = payload.kickoff
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        age_days = (kickoff - latest_datetime).total_seconds() / 86400.0
        return 0 <= age_days <= settings.HISTORICAL_FORM_MAX_AGE_DAYS

    async def recent_matches(local_frame: Any, team_id: int) -> Any:
        if local_history_is_usable(local_frame):
            return local_frame
        api_frame = await football_api.get_team_last_matches_df(
            team_id, last=settings.RECENT_FORM_MATCH_COUNT
        )
        if api_frame is not None and not api_frame.empty:
            return api_frame
        return local_frame

    async def h2h() -> Any:
        if historical.h2h_rates:
            return historical.h2h_rates
        return await football_api.get_h2h(
            home_team_id,
            away_team_id,
            last=settings.RECENT_FORM_MATCH_COUNT,
        )

    async def availability() -> Any:
        if payload.fixture_id is None:
            return None
        return await football_api.get_fixture_availability(
            payload.fixture_id, home_team_id, away_team_id
        )

    async def lineups() -> Any:
        if payload.fixture_id is None:
            return None
        return await football_api.get_fixture_lineups(
            payload.fixture_id, home_team_id, away_team_id
        )

    (
        home_matches_df,
        away_matches_df,
        h2h_rates,
        availability_data,
        lineup_data,
    ) = await asyncio.gather(
        recent_matches(historical.home_matches_df, home_team_id),
        recent_matches(historical.away_matches_df, away_team_id),
        h2h(),
        availability(),
        lineups(),
    )
    return (
        home_matches_df,
        away_matches_df,
        h2h_rates,
        availability_data,
        lineup_data,
    )


def _availability_player_ids(
    availability: Dict[str, Any] | None,
    *,
    side: Literal["home", "away"],
    status: Literal["missing", "questionable"],
) -> list[int]:
    """Extract stable provider IDs while preserving legacy count-only payloads."""
    if not isinstance(availability, dict):
        return []
    rows = availability.get(f"{side}_unavailable_players")
    if not isinstance(rows, list):
        return []

    player_ids: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != status:
            continue
        player_id = row.get("player_id")
        if (
            isinstance(player_id, int)
            and not isinstance(player_id, bool)
            and player_id > 0
        ):
            player_ids.append(player_id)
    return list(dict.fromkeys(player_ids))


def _derive_reference_lineup(
    player_ratings: Mapping[int, PlayerRatingValue],
) -> list[int] | None:
    """Infer a typical XI from season exposure when no prior XI was ingested."""
    return PlayerImpactCalculator.derive_reference_lineup(player_ratings)


def _select_reference_lineup(
    previous_lineup: list[int] | None,
    player_ratings: Mapping[int, PlayerRatingValue],
) -> list[int] | None:
    """Use a prior XI only when every member is still in the current rating pool."""
    if isinstance(previous_lineup, list):
        normalized = list(
            dict.fromkeys(
                player_id
                for player_id in previous_lineup
                if (
                    isinstance(player_id, int)
                    and not isinstance(player_id, bool)
                    and player_id > 0
                )
            )
        )
        if len(normalized) == 11:
            covered = {
                player_id: player_ratings[player_id]
                for player_id in normalized
                if player_id in player_ratings
            }
            if (
                len(covered) == 11
                and PlayerImpactCalculator.derive_reference_lineup(covered) is not None
            ):
                return normalized
    return _derive_reference_lineup(player_ratings)


async def _fetch_player_rating_data(
    payload: AnalysisRequest,
    historical: HistoricalFeatureContext,
) -> tuple[
    dict[int, PlayerRatingValue],
    dict[int, PlayerRatingValue],
]:
    """Merge point-in-time history with live season ratings for upcoming games."""
    home_local = dict(historical.home_player_ratings or {})
    away_local = dict(historical.away_player_ratings or {})
    season = payload.season
    if payload.home_team_id is None or payload.away_team_id is None or season is None:
        return home_local, away_local

    # A current season aggregate is safe for an upcoming match, but would leak
    # future information into a historical replay.
    if payload.kickoff is not None:
        kickoff = payload.kickoff
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        if kickoff.astimezone(timezone.utc) < datetime.now(timezone.utc):
            return home_local, away_local

    async def merged_ratings(
        team_id: int,
        team_name: str,
        local: dict[int, PlayerRatingValue],
        reference_lineup: list[int] | None,
    ) -> dict[int, PlayerRatingValue]:
        if _select_reference_lineup(reference_lineup, local) is not None:
            return local
        live = await football_api.get_team_player_ratings(
            team_id,
            season,
            league_id=payload.league_id,
        )
        if _derive_reference_lineup(live) is None:
            alternative_raw = await sportmonks_player_service.get_team_player_ratings(
                canonical_team_id=team_id,
                canonical_team_name=team_name,
                as_of=payload.kickoff or datetime.now(timezone.utc),
            )
            alternative: dict[int, PlayerRatingValue] = {
                player_id: rating for player_id, rating in alternative_raw.items()
            }
            if _derive_reference_lineup(alternative) is None:
                # Never manufacture an XI by mixing incomplete provider rosters.
                return local
            # Sportmonks IDs use a dedicated numeric namespace; do not mix them
            # with API-Football IDs or stale local rows.
            return alternative

        # The season feed defines current roster membership. Local rolling ratings
        # remain the fresher signal, but only for players present in that roster.
        return {
            player_id: local.get(player_id, rating)
            for player_id, rating in live.items()
        }

    home_ratings, away_ratings = await asyncio.gather(
        merged_ratings(
            payload.home_team_id,
            payload.home_team,
            home_local,
            historical.home_previous_starting_xi,
        ),
        merged_ratings(
            payload.away_team_id,
            payload.away_team,
            away_local,
            historical.away_previous_starting_xi,
        ),
    )
    return home_ratings, away_ratings


def _get_historical_feature_context(
    payload: AnalysisRequest,
) -> HistoricalFeatureContext:
    if (
        payload.home_team_id is None
        or payload.away_team_id is None
        or payload.league_id is None
        or payload.kickoff is None
    ):
        return HistoricalFeatureContext()

    with SessionLocal() as db:
        service = HistoricalFeatureService(
            HistoricalFixtureRepository(db),
            player_context_repository=PlayerContextRepository(db),
        )
        return service.build_context(
            home_team_id=payload.home_team_id,
            away_team_id=payload.away_team_id,
            home_team_name=payload.home_team,
            away_team_name=payload.away_team,
            league_id=payload.league_id,
            before=payload.kickoff,
            recent_match_count=settings.RECENT_FORM_MATCH_COUNT,
            elo_k_factor=settings.ELO_K_FACTOR,
            elo_home_advantage_points=settings.ELO_HOME_ADVANTAGE_POINTS,
            elo_season_regression=settings.ELO_SEASON_REGRESSION,
        )


async def _apply_external_elo_fallback(
    payload: AnalysisRequest,
    historical: HistoricalFeatureContext,
) -> HistoricalFeatureContext:
    if payload.kickoff is None:
        return historical

    home_result = away_result = None
    requests = []
    request_sides: list[str] = []
    if (
        not historical.home_elo_available
        and payload.home_team_id is not None
        and payload.home_team.strip()
    ):
        request_sides.append("home")
        requests.append(
            external_feature_service.get_team_elo(
                canonical_team_id=payload.home_team_id,
                canonical_team_name=payload.home_team,
                as_of=payload.kickoff,
            )
        )
    if (
        not historical.away_elo_available
        and payload.away_team_id is not None
        and payload.away_team.strip()
    ):
        request_sides.append("away")
        requests.append(
            external_feature_service.get_team_elo(
                canonical_team_id=payload.away_team_id,
                canonical_team_name=payload.away_team,
                as_of=payload.kickoff,
            )
        )
    if requests:
        results = await asyncio.gather(*requests)
        for side, result in zip(request_sides, results, strict=True):
            if side == "home":
                home_result = result
            else:
                away_result = result

    provenance = dict(historical.feature_provenance)
    if home_result is not None:
        provenance["home_elo"] = home_result.provenance()
    if away_result is not None:
        provenance["away_elo"] = away_result.provenance()
    if home_result is None and away_result is None:
        return historical
    return replace(
        historical,
        home_elo=(
            home_result.value if home_result is not None else historical.home_elo
        ),
        away_elo=(
            away_result.value if away_result is not None else historical.away_elo
        ),
        home_elo_available=historical.home_elo_available or home_result is not None,
        away_elo_available=historical.away_elo_available or away_result is not None,
        feature_provenance=provenance,
    )


async def _apply_external_travel_fallback(
    payload: AnalysisRequest,
    historical: HistoricalFeatureContext,
) -> HistoricalFeatureContext:
    if (
        not settings.AUTO_TEAM_LOCATION_ENABLED
        or payload.away_travel_distance_km is not None
        or historical.travel_context_available
        or historical.away_travel_distance_km > 0
        or payload.home_team_id is None
        or payload.away_team_id is None
    ):
        return historical
    point = await travel_context_service.get_away_travel_distance(
        home_team_id=payload.home_team_id,
        away_team_id=payload.away_team_id,
        home_team_name=payload.home_team,
        away_team_name=payload.away_team,
        client=football_api,
    )
    if point is None:
        return historical
    return replace(
        historical,
        away_travel_distance_km=point.value,
        travel_context_available=True,
        travel_provenance=point.provenance(),
    )


async def _compute_analysis(payload: AnalysisRequest) -> dict:
    """Run analysis with external inputs and a short point-in-time history read."""
    home_stats = payload.home_stats.model_dump()
    away_stats = payload.away_stats.model_dump()

    ml_result: dict = {"ready": False}
    ml_explanations: List[str] = []

    historical = await _apply_external_elo_fallback(
        payload,
        _get_historical_feature_context(payload),
    )
    historical = await _apply_external_travel_fallback(payload, historical)
    (
        home_matches_df,
        away_matches_df,
        h2h_rates,
        availability,
        lineups,
    ) = await _fetch_ml_match_data(payload, historical)
    home_player_ratings, away_player_ratings = await _fetch_player_rating_data(
        payload,
        historical,
    )
    home_reference_lineup = _select_reference_lineup(
        historical.home_previous_starting_xi,
        home_player_ratings,
    )
    away_reference_lineup = _select_reference_lineup(
        historical.away_previous_starting_xi,
        away_player_ratings,
    )
    lineup_context = {
        **(lineups or {}),
        "home_previous_starting_xi": home_reference_lineup,
        "away_previous_starting_xi": away_reference_lineup,
    }
    home_player_impact = PlayerImpactCalculator.assess(
        home_player_ratings,
        home_reference_lineup,
        lineup_context.get("home_starting_xi"),
        _availability_player_ids(availability, side="home", status="missing"),
        _availability_player_ids(availability, side="home", status="questionable"),
    )
    away_player_impact = PlayerImpactCalculator.assess(
        away_player_ratings,
        away_reference_lineup,
        lineup_context.get("away_starting_xi"),
        _availability_player_ids(availability, side="away", status="missing"),
        _availability_player_ids(availability, side="away", status="questionable"),
    )
    stats_analysis = StatsEngine.analyze_match(
        home_stats,
        away_stats,
        league_id=payload.league_id,
        home_match_history=home_matches_df,
        away_match_history=away_matches_df,
        as_of=payload.kickoff,
        home_player_impact=home_player_impact,
        away_player_impact=away_player_impact,
    )
    opening_odds = (
        payload.opening_odds_1x2.as_outcome_dict()
        if payload.opening_odds_1x2 is not None
        else None
    )
    current_odds = (
        payload.current_odds_1x2.as_outcome_dict()
        if payload.current_odds_1x2 is not None
        else None
    )
    calculated_feature_vector = FeatureEngine.build_inference_features(
        home_stats=payload.home_stats.model_dump(),
        away_stats=payload.away_stats.model_dump(),
        home_matches_df=home_matches_df,
        away_matches_df=away_matches_df,
        h2h_rates=h2h_rates,
        h2h_matches=historical.h2h_matches,
        home_elo=historical.home_elo,
        away_elo=historical.away_elo,
        availability=availability,
        lineup_context=lineup_context,
        fixture_date=payload.kickoff,
        league_id=payload.league_id,
        home_team_id=payload.home_team_id,
        away_team_id=payload.away_team_id,
        opening_odds=opening_odds,
        current_odds=current_odds,
        home_schedule_df=historical.home_schedule_df,
        away_schedule_df=historical.away_schedule_df,
        away_travel_distance_km=(
            payload.away_travel_distance_km
            if payload.away_travel_distance_km is not None
            else historical.away_travel_distance_km
        ),
        home_player_impact=home_player_impact,
        away_player_impact=away_player_impact,
    )
    feature_vector = {
        name: float(
            payload.feature_overrides.get(name, calculated_feature_vector[name])
        )
        for name in FeatureEngine.FEATURE_NAMES
    }
    if any(not math.isfinite(value) for value in feature_vector.values()):
        raise ValueError("Feature vector contains non-finite values")
    if ml_pipeline.is_ready:
        ml_result = ml_pipeline.predict_match(feature_vector)
        if ml_result.get("ready"):
            ml_explanations = ExplainabilityService.generate_explanation(
                ml_pipeline.model,
                feature_vector,
                ml_pipeline.feature_names,
            )

    analysis = ProbabilityEnsembler.apply(
        stats_analysis,
        ml_result=ml_result,
        market=payload.market_1x2,
        league_id=payload.league_id,
    )
    value_data = ValueCalc.calculate_professional(
        analysis, payload.market_1x2, fallback_odd=payload.odd
    )
    if payload.market_1x2:
        value_data["data_methodology"] = {
            "stats": "Zaman ağırlıklı geçmiş + sezon profili fallback + form decay",
            "odds": f"1X2 devig (overround %{payload.market_1x2.get('overround_pct', 0)})",
            "model": analysis.get("model", "poisson_dixon_coles"),
        }
    insights = StatsEngine.build_insights(analysis, value_data)
    insights.extend(ml_explanations)
    history_home_count = len(home_matches_df) if home_matches_df is not None else 0
    history_away_count = len(away_matches_df) if away_matches_df is not None else 0
    required_history = settings.RECENT_FORM_MATCH_COUNT
    h2h_source = h2h_rates.get("source") if isinstance(h2h_rates, dict) else None
    home_lineup = lineups.get("home_starting_xi") if isinstance(lineups, dict) else None
    away_lineup = lineups.get("away_starting_xi") if isinstance(lineups, dict) else None
    quality_checks = {
        "fixture_identified": payload.fixture_id is not None,
        "league_identified": payload.league_id is not None,
        "kickoff_known": payload.kickoff is not None,
        "market_available": payload.market_1x2 is not None,
        "h2h_available": bool(
            historical.h2h_matches
            or (h2h_rates and h2h_source not in {"fallback", "demo_default"})
        ),
        "home_history_available": history_home_count > 0,
        "away_history_available": history_away_count > 0,
        "home_history_sufficient": history_home_count >= required_history,
        "away_history_sufficient": history_away_count >= required_history,
        "home_elo_available": historical.home_elo_available,
        "away_elo_available": historical.away_elo_available,
        "availability_available": bool(
            isinstance(availability, dict)
            and availability.get("availability_report_present")
        ),
        "lineups_available": bool(
            isinstance(home_lineup, list)
            and len(home_lineup) == 11
            and isinstance(away_lineup, list)
            and len(away_lineup) == 11
        ),
        "home_player_impact_available": home_player_impact.data_available,
        "away_player_impact_available": away_player_impact.data_available,
        "travel_context_available": (
            payload.away_travel_distance_km is not None
            or historical.travel_context_available
            or historical.away_travel_distance_km > 0
        ),
        "odds_movement_available": bool(opening_odds and current_odds),
    }
    feature_provenance = dict(historical.feature_provenance)
    active_travel_provenance: dict[str, object] | None = (
        {
            "source": "manual_override",
            "captured_at": None,
            "confidence": 1.0,
            "is_fallback": False,
        }
        if payload.away_travel_distance_km is not None
        else historical.travel_provenance
        or (
            {
                "source": "curated_team_locations",
                "captured_at": None,
                "confidence": 1.0,
                "is_fallback": False,
            }
            if historical.away_travel_distance_km > 0
            else None
        )
    )
    if (
        active_travel_provenance is not None
        and quality_checks["home_history_sufficient"]
        and quality_checks["away_history_sufficient"]
    ):
        travel_source = str(active_travel_provenance.get("source") or "team_locations")
        feature_provenance["fatigue_index"] = {
            **active_travel_provenance,
            "source": f"schedule_and_{travel_source}",
        }
    if opening_odds and current_odds:
        odds_provenance: dict[str, object] = {
            "source": "api_football_odds",
            "captured_at": (
                payload.current_odds_at.isoformat()
                if payload.current_odds_at is not None
                else None
            ),
            "confidence": settings.ODDS_SNAPSHOT_CONFIDENCE,
            "is_fallback": False,
        }
        feature_provenance.update(
            {
                feature_name: odds_provenance
                for feature_name in AnalysisInputCatalog.ODDS_MOVEMENT_INPUTS
            }
        )
    data_quality = {
        "score": round(
            100.0
            * sum(1 for passed in quality_checks.values() if passed)
            / len(quality_checks),
            2,
        ),
        "checks": quality_checks,
        "home_history_matches": history_home_count,
        "away_history_matches": history_away_count,
        "required_history_matches": required_history,
        "player_impact": {
            "home_strength_ratio": home_player_impact.team_strength_ratio,
            "away_strength_ratio": away_player_impact.team_strength_ratio,
            "home_critical_missing": home_player_impact.critical_missing_count,
            "away_critical_missing": away_player_impact.critical_missing_count,
            "neutral_fallback_used": not (
                home_player_impact.data_available and away_player_impact.data_available
            ),
        },
        "away_travel_distance_km": round(
            float(
                payload.away_travel_distance_km
                if payload.away_travel_distance_km is not None
                else historical.away_travel_distance_km
            ),
            2,
        ),
        "travel_provenance": active_travel_provenance,
        "odds_snapshot": {
            "movement_features_used": bool(opening_odds and current_odds),
            "opening_captured_at": (
                payload.opening_odds_at.isoformat()
                if payload.opening_odds_at is not None
                else None
            ),
            "current_captured_at": (
                payload.current_odds_at.isoformat()
                if payload.current_odds_at is not None
                else None
            ),
        },
        "manual_feature_overrides": sorted(payload.feature_overrides),
        "manual_feature_override_count": len(payload.feature_overrides),
        "feature_provenance": feature_provenance,
    }

    return {
        "analysis": analysis,
        "value_data": value_data,
        "ml_result": ml_result,
        "feature_vector": feature_vector,
        "calculated_feature_vector": calculated_feature_vector,
        "insights": insights,
        "data_quality": data_quality,
    }


def _persist_analysis(payload: AnalysisRequest, computed: dict):
    analysis = computed["analysis"]
    value_data = computed["value_data"]
    ml_result = computed["ml_result"]
    feature_vector = computed["feature_vector"]
    data_quality = computed["data_quality"]
    probs = analysis["all_probabilities"]
    best_pick = value_data.get("best_pick") or {}
    analyzed_at = datetime.now(timezone.utc)
    kickoff = payload.kickoff
    if kickoff is not None and kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    analysis_lead_minutes = (
        round((kickoff - analyzed_at).total_seconds() / 60.0, 2)
        if kickoff is not None
        else None
    )

    record_data = {
        "fixture_id": payload.fixture_id,
        "home_team": payload.home_team,
        "away_team": payload.away_team,
        "home_team_id": payload.home_team_id,
        "away_team_id": payload.away_team_id,
        "league_id": payload.league_id,
        "home_xg": payload.home_stats.xg,
        "away_xg": payload.away_stats.xg,
        "home_form": payload.home_stats.form,
        "away_form": payload.away_stats.form,
        "home_attack": payload.home_stats.attack,
        "home_defense": payload.home_stats.defense,
        "away_attack": payload.away_stats.attack,
        "away_defense": payload.away_stats.defense,
        "prediction": analysis["prediction"],
        "probability": analysis["probability"],
        "prob_home": probs["HOME_WIN"],
        "prob_away": probs["AWAY_WIN"],
        "prob_draw": probs["DRAW"],
        "odd": payload.odd,
        "edge": value_data["edge"],
        "is_value_bet": 1 if value_data["value_bet"] else 0,
        "kelly_stake": best_pick.get("kelly_stake_pct"),
        "ml_cluster": _ml_cluster_value(ml_result),
        "ml_confidence": (
            ml_result.get("probability", 0.0) if ml_result.get("ready") else 0.0
        ),
        "feature_snapshot": feature_vector,
        "feature_schema_version": FeatureEngine.SCHEMA_VERSION,
        "feature_snapshot_at": datetime.now(timezone.utc),
        "probability_components": analysis.get("ensemble"),
        "ensemble_version": (analysis.get("ensemble") or {}).get("version"),
        "model_name": ml_result.get("model_name") or analysis.get("model"),
        "model_artifact_version": ml_result.get("artifact_version"),
        "data_quality": data_quality,
        "kickoff": kickoff,
        "analyzed_at": analyzed_at,
        "analysis_lead_minutes": analysis_lead_minutes,
        "market_snapshot_at": payload.current_odds_at
        or (analyzed_at if payload.market_1x2 else None),
    }

    with SessionLocal() as db:
        repo = MatchPredictionRepository(db)
        record = repo.upsert_prediction(record_data)
        return record, repo.count_labeled()


async def _run_analysis(payload: AnalysisRequest) -> dict:
    computed = await _compute_analysis(payload)
    db_record, labeled_samples_count = _persist_analysis(payload, computed)

    response = _build_analysis_response(
        db_record.id,
        payload.home_team,
        payload.away_team,
        computed["analysis"],
        computed["value_data"],
        computed["ml_result"],
        computed["insights"],
        labeled_samples_count,
    )
    if computed["value_data"].get("data_methodology"):
        response["data_methodology"] = computed["value_data"]["data_methodology"]
    response["data_quality"] = computed["data_quality"]
    response["provenance"] = {
        "model_name": computed["ml_result"].get("model_name")
        or computed["analysis"].get("model"),
        "model_artifact_version": computed["ml_result"].get("artifact_version"),
        "feature_schema_version": FeatureEngine.SCHEMA_VERSION,
        "ensemble_version": (computed["analysis"].get("ensemble") or {}).get("version"),
        "analyzed_at": db_record.analyzed_at,
        "kickoff": db_record.kickoff,
        "analysis_lead_minutes": db_record.analysis_lead_minutes,
    }
    return response


@router.get("/leagues")
def list_allowed_leagues():
    return ALLOWED_LEAGUES


@router.get("/fixtures/upcoming")
async def list_upcoming_fixtures(
    days: int = Query(default=7, ge=1, le=14),
    limit: int = Query(default=100, ge=1, le=200),
):
    return await football_api.get_upcoming_fixtures(days=days, limit=limit)


@router.get("/fixtures/{fixture_id}/prefill")
async def fixture_prefill(fixture_id: int):
    payload = await football_api.get_fixture_prefill(fixture_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    return odds_history_service.enrich_prefill(payload)


@router.post("/analyze", dependencies=[Depends(require_permission("analysis:create"))])
async def analyze_manual(payload: AnalysisRequest):
    try:
        return await _run_analysis(payload)
    except SQLAlchemyError as exc:
        logger.exception("Veritabanı hatası (manuel analiz)")
        raise HTTPException(status_code=500, detail="Veritabanı hatası.") from exc


@router.post(
    "/analyze/preview",
    dependencies=[Depends(require_permission("analysis:create"))],
)
async def preview_analysis_inputs(payload: AnalysisRequest):
    """Return every point-in-time model input without persisting a prediction."""
    try:
        computed = await _compute_analysis(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "feature_schema_version": FeatureEngine.SCHEMA_VERSION,
        "features": AnalysisInputCatalog.build(
            computed["calculated_feature_vector"],
            payload.feature_overrides,
            computed["data_quality"],
        ),
        "derived": {
            "expected_goals": computed["analysis"].get("expected_goals"),
            "player_impact": computed["analysis"].get("player_impact"),
            "statistics_probabilities": (
                computed["analysis"]
                .get("ensemble", {})
                .get("components", {})
                .get("stats", computed["analysis"].get("all_probabilities"))
            ),
        },
        "data_quality": computed["data_quality"],
    }


@router.post(
    "/analyze/fixture/{fixture_id}",
    dependencies=[Depends(require_permission("analysis:create"))],
)
async def analyze_fixture(fixture_id: int):
    prefill = await football_api.get_fixture_prefill(fixture_id)
    if not prefill:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    prefill = odds_history_service.enrich_prefill(prefill)

    try:
        payload = _build_payload_from_prefill(prefill)
    except (KeyError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="Geçersiz maç verisi.") from exc

    try:
        result = await _run_analysis(payload)
    except SQLAlchemyError as exc:
        logger.exception("Veritabanı hatası (fixture_id=%s)", fixture_id)
        raise HTTPException(status_code=500, detail="Veritabanı hatası.") from exc

    result["prefill"] = prefill
    result["data_methodology"] = prefill.get("data_methodology")
    result["data_quality"] = prefill.get("data_quality")
    return result


@router.get("/history", dependencies=[Depends(require_permission("history:read"))])
def get_history(
    paginated: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=1, le=100),
    query: str = Query(default="", max_length=100),
    result: Literal["all", "pending", "HOME_WIN", "DRAW", "AWAY_WIN"] = Query(
        default="all"
    ),
    value: Literal["all", "value", "non_value"] = Query(default="all"),
    sort: Literal["newest", "oldest", "edge", "odd"] = Query(default="newest"),
    db: Session = Depends(get_db),
):
    repo = MatchPredictionRepository(db)
    if not paginated:
        return repo.get_recent(limit=page_size)

    items, total = repo.search_history(
        page=page,
        page_size=page_size,
        query=query,
        result=result,
        value=value,
        sort=sort,
    )
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


@router.get(
    "/ml/labeling-queue",
    dependencies=[Depends(require_permission("history:update_result"))],
)
def get_ml_labeling_queue(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    repo = MatchPredictionRepository(db)
    candidates = ActiveLearningSelector.rank(repo.get_unlabeled(), limit=limit)
    labeled_samples_count = repo.count_labeled()
    return {
        "strategy": "uncertainty_sampling",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "labeled_samples_count": labeled_samples_count,
        "remaining_to_threshold": max(
            0, settings.MIN_TRAINING_SAMPLES - labeled_samples_count
        ),
    }


@router.get(
    "/ml/status",
    dependencies=[Depends(require_permission("history:read"))],
)
def get_ml_status(db: Session = Depends(get_db)):
    from app.services.model_monitoring import ModelMonitoringService

    result = ml_pipeline.status()
    labeled_predictions = MatchPredictionRepository(db).count_labeled()
    historical_fixtures = db.query(func.count(HistoricalFixture.id)).scalar() or 0
    result["training_data"] = {
        "labeled_predictions": labeled_predictions,
        "historical_fixtures": historical_fixtures,
        "minimum_samples": settings.MIN_TRAINING_SAMPLES,
        "historical_minimum_team_matches": (
            settings.HISTORICAL_TRAINING_MIN_TEAM_MATCHES
        ),
    }
    result["monitoring"] = ModelMonitoringService(db).snapshot()
    return result


@router.post(
    "/ml/rollback",
    dependencies=[Depends(require_permission("users:manage"))],
)
def rollback_ml_model():
    if not ml_pipeline.rollback():
        raise HTTPException(
            status_code=409,
            detail="Doğrulanmış önceki model artifact'ı bulunamadı.",
        )
    return {"ok": True, "model": ml_pipeline.status()}


@router.patch(
    "/history/{record_id}/result",
    dependencies=[Depends(require_permission("history:update_result"))],
)
def update_actual_result(
    record_id: int,
    body: ActualResultUpdate,
    db: Session = Depends(get_db),
):
    repo = MatchPredictionRepository(db)
    record = repo.get_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı.")

    roi = PredictionAuditor.calculate_bet_roi(
        record.prediction, body.actual_result, record.odd
    )

    repo.update_result(
        record_id=record_id,
        actual_result=body.actual_result,
        actual_score_home=body.actual_score_home,
        actual_score_away=body.actual_score_away,
        roi=roi,
    )

    labeled_samples_count = repo.count_labeled()
    if ActiveLearningSelector.should_retrain(
        labeled_samples_count, ml_pipeline.is_ready
    ):
        worker = enqueue_retraining()
    else:
        worker = {
            "status": "threshold_not_reached",
            "broker_reachable": None,
            "worker_reachable": None,
            "workers": [],
            "task_queued": False,
            "task_id": None,
        }

    return {
        "ok": True,
        "id": record_id,
        "actual_result": body.actual_result,
        "labeled_samples_count": labeled_samples_count,
        "remaining_to_threshold": max(
            0, settings.MIN_TRAINING_SAMPLES - labeled_samples_count
        ),
        "worker": worker,
    }


@router.post("/backtest", dependencies=[Depends(require_permission("backtest:run"))])
def run_backtest(body: BacktestRequest, db: Session = Depends(get_db)):
    repo = MatchPredictionRepository(db)
    predictions = repo.get_all()

    return BacktestEngine.run_simulation(
        predictions=predictions,
        initial_bankroll=body.initial_bankroll,
        strategy=body.strategy,
        flat_stake_amount=body.flat_stake_amount,
        kelly_fraction=body.kelly_fraction,
        min_edge_pct=body.min_edge_pct,
        commission_pct=body.commission_pct,
        max_stake_pct=body.max_stake_pct,
        max_daily_exposure_pct=body.max_daily_exposure_pct,
        require_closing_odds=body.require_closing_odds,
        exclude_post_kickoff=body.exclude_post_kickoff,
    )


@router.get("/audit", dependencies=[Depends(require_permission("audit:read"))])
def run_audit(db: Session = Depends(get_db)):
    repo = MatchPredictionRepository(db)
    return PredictionAuditor.audit_predictions(repo.get_all())


@router.get(
    "/operations/data-quality",
    dependencies=[Depends(require_permission("audit:read"))],
)
async def get_data_quality(db: Session = Depends(get_db)):
    from app.services.api_provider_health import api_football_health

    snapshot = DataQualityService(db).snapshot()
    snapshot["providers"] = {
        "api_football": await api_football_health.snapshot(),
        "sportmonks": {
            "status": "configured" if settings.SPORTMONKS_ENABLED else "disabled",
            "enabled": settings.SPORTMONKS_ENABLED,
        },
    }
    return snapshot
