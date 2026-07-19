import asyncio
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
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import logging

from app.db.session import SessionLocal, get_db
from app.db.repository import MatchPredictionRepository
from app.db.historical_repository import HistoricalFixtureRepository
from app.services.api_football import APIFootballClient
from app.prediction.stats_engine import StatsEngine
from app.prediction.ensemble import ProbabilityEnsembler
from app.prediction.value_calc import ValueCalc
from app.prediction.ml.model import ml_pipeline
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import (
    HistoricalFeatureContext,
    HistoricalFeatureService,
)
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
    return AnalysisRequest(
        home_team=prefill["home_team"],
        away_team=prefill["away_team"],
        home_stats=TeamStatsInput(**prefill["home_stats"]),
        away_stats=TeamStatsInput(**prefill["away_stats"]),
        odd=prefill["odd"],
        market_1x2=prefill.get("market_1x2"),
        fixture_id=fixture.get("fixture_id"),
        home_team_id=fixture.get("home_team_id"),
        away_team_id=fixture.get("away_team_id"),
        league_id=fixture.get("league_id"),
        season=fixture.get("season"),
        kickoff=fixture.get("kickoff"),
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
) -> Tuple[Any, Any, Any]:
    """Fill missing point-in-time sources from API without holding a DB session."""
    home_team_id = payload.home_team_id
    away_team_id = payload.away_team_id
    if home_team_id is None or away_team_id is None:
        return historical.home_matches_df, historical.away_matches_df, None

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

    home_matches_df, away_matches_df, h2h_rates = await asyncio.gather(
        recent_matches(historical.home_matches_df, home_team_id),
        recent_matches(historical.away_matches_df, away_team_id),
        h2h(),
    )
    return home_matches_df, away_matches_df, h2h_rates


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
        service = HistoricalFeatureService(HistoricalFixtureRepository(db))
        return service.build_context(
            home_team_id=payload.home_team_id,
            away_team_id=payload.away_team_id,
            league_id=payload.league_id,
            before=payload.kickoff,
            recent_match_count=settings.RECENT_FORM_MATCH_COUNT,
            elo_k_factor=settings.ELO_K_FACTOR,
            elo_home_advantage_points=settings.ELO_HOME_ADVANTAGE_POINTS,
            elo_season_regression=settings.ELO_SEASON_REGRESSION,
        )


async def _compute_analysis(payload: AnalysisRequest) -> dict:
    """Run analysis with external inputs and a short point-in-time history read."""
    home_stats = payload.home_stats.model_dump()
    away_stats = payload.away_stats.model_dump()
    stats_analysis = StatsEngine.analyze_match(
        home_stats, away_stats, league_id=payload.league_id
    )

    ml_result: dict = {"ready": False}
    ml_explanations: List[str] = []

    historical = _get_historical_feature_context(payload)
    home_matches_df, away_matches_df, h2h_rates = await _fetch_ml_match_data(
        payload, historical
    )
    feature_vector = FeatureEngine.build_inference_features(
        home_stats=payload.home_stats.model_dump(),
        away_stats=payload.away_stats.model_dump(),
        home_matches_df=home_matches_df,
        away_matches_df=away_matches_df,
        h2h_rates=h2h_rates,
        h2h_matches=historical.h2h_matches,
        home_elo=historical.home_elo,
        away_elo=historical.away_elo,
        fixture_date=payload.kickoff,
    )
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
    )
    value_data = ValueCalc.calculate_professional(
        analysis, payload.market_1x2, fallback_odd=payload.odd
    )
    if payload.market_1x2:
        value_data["data_methodology"] = {
            "stats": "Ev/deplasman sezon ortalamaları + form decay",
            "odds": f"1X2 devig (overround %{payload.market_1x2.get('overround_pct', 0)})",
            "model": analysis.get("model", "poisson_dixon_coles"),
        }
    insights = StatsEngine.build_insights(analysis, value_data)
    insights.extend(ml_explanations)

    return {
        "analysis": analysis,
        "value_data": value_data,
        "ml_result": ml_result,
        "feature_vector": feature_vector,
        "insights": insights,
    }


def _persist_analysis(payload: AnalysisRequest, computed: dict):
    analysis = computed["analysis"]
    value_data = computed["value_data"]
    ml_result = computed["ml_result"]
    feature_vector = computed["feature_vector"]
    probs = analysis["all_probabilities"]
    best_pick = value_data.get("best_pick") or {}

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
    return response


@router.get("/leagues")
def list_allowed_leagues():
    return ALLOWED_LEAGUES


@router.get("/fixtures/upcoming")
async def list_upcoming_fixtures():
    return await football_api.get_upcoming_fixtures()


@router.get("/fixtures/{fixture_id}/prefill")
async def fixture_prefill(fixture_id: int):
    payload = await football_api.get_fixture_prefill(fixture_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    return payload


@router.post("/analyze", dependencies=[Depends(require_permission("analysis:create"))])
async def analyze_manual(payload: AnalysisRequest):
    try:
        return await _run_analysis(payload)
    except SQLAlchemyError as exc:
        logger.exception("Veritabanı hatası (manuel analiz)")
        raise HTTPException(status_code=500, detail="Veritabanı hatası.") from exc


@router.post(
    "/analyze/fixture/{fixture_id}",
    dependencies=[Depends(require_permission("analysis:create"))],
)
async def analyze_fixture(fixture_id: int):
    prefill = await football_api.get_fixture_prefill(fixture_id)
    if not prefill:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

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
def get_ml_status():
    return ml_pipeline.status()


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
    )


@router.get("/audit", dependencies=[Depends(require_permission("audit:read"))])
def run_audit(db: Session = Depends(get_db)):
    repo = MatchPredictionRepository(db)
    return PredictionAuditor.audit_predictions(repo.get_all())
