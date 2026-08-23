"""Endpoints package.

Domain logic lives in ``prediction_helpers.py``.
This module defines the FastAPI router, mounts route handlers, and re-exports
every public and private helper so that existing imports like
``from app.api.endpoints import _compute_analysis`` keep working.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.db.session import SessionLocal, get_db
from app.db.repository import MatchPredictionRepository
from app.db.models import HistoricalFixture
from app.services.fixture_aggregator import FixtureAggregator
from app.services.fixture_context import fixture_context_service
from app.services.fixture_readiness import FixtureReadinessService
from app.services.cache import cache
from app.services.data_quality import DataQualityService
from app.services.odds_history import odds_history_service
from app.prediction.ml.active_learning import ActiveLearningSelector
from app.prediction.audit import PredictionAuditor
from app.prediction.backtest import BacktestEngine
from app.core.allowed_leagues import ALLOWED_LEAGUES
from app.tasks.health import enqueue_retraining
from app.core.config import settings
from app.core.api_mode import get_api_mode
from app.core.auth import (
    CurrentUser,
    require_authenticated_user,
    require_permission,
)
from app.core.rate_limit import batch_prediction_rate_limiter
from app.api.admin import router as admin_router
from app.api.endpoints.schemas import (
    PlatformStatusResponse,
    DataQualityResponse,
    MLStatusResponse,
    TeamStatsInput as TeamStatsInput,
    AnalysisRequest,
    TieredPredictionRequest,
    ActualResultUpdate,
    BacktestRequest,
)
from app.api.endpoints.auth_endpoints import router as auth_router

from app.api.endpoints.prediction_helpers import (  # noqa: F401
    _market_settlement_price,
    _build_payload_from_prefill,
    _probability_favorite,
    _assess_ml_safety,
    _ml_cluster_value,
    _build_analysis_response,
    _fetch_ml_match_data,
    _availability_player_ids,
    _derive_reference_lineup,
    _select_reference_lineup,
    _fetch_player_rating_data,
    _get_historical_feature_context,
    _apply_external_elo_fallback,
    _apply_external_travel_fallback,
    _fetch_match_weather,
    _INTERACTIVE_EXCLUDED_CHECKS,
    _ensure_live_market,
    _compute_analysis,
    _persist_analysis,
    _is_training_eligible,
    _enrich_with_remote_market,
    _run_analysis,
    get_tiered_predictor,
    _tiered_gate_evidence,
    football_api,
    open_meteo_client,
    external_feature_service,
    travel_context_service,
    sportmonks_player_service,
    StatsEngine,
    ProbabilityEnsembler,
    ValueCalc,
    PlayerImpactCalculator,
    PredictionEligibilityPolicy,
    PredictionDecisionPolicy,
    AnalysisInputCatalog,
    ExplainabilityService,
    AnalysisQualityScorer,
    ml_pipeline,
    FeatureEngine,
    HistoricalFeatureContext,
    HistoricalFeatureService,
    PlayerRatingValue,
    Predictor,
    TieredArtifactIntegrityError,
)

logger = logging.getLogger("bet-ai-pro.api")
router = APIRouter()
fixture_aggregator = FixtureAggregator(api_football=football_api)
router.include_router(admin_router)
router.include_router(auth_router)

def _annotate_fixture_readiness(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        return FixtureReadinessService(db).annotate(fixtures)


@router.get("/status", response_model=PlatformStatusResponse)
def get_platform_status() -> PlatformStatusResponse:
    return PlatformStatusResponse(
        api_mode=get_api_mode(settings.API_FOOTBALL_KEY),
        registration_enabled=settings.ALLOW_SELF_REGISTRATION,
    )


@router.get("/leagues", dependencies=[Depends(require_authenticated_user)])
def list_allowed_leagues():
    return ALLOWED_LEAGUES


@router.get(
    "/fixtures/upcoming",
    dependencies=[Depends(require_authenticated_user)],
)
async def list_upcoming_fixtures(
    days: int = Query(default=7, ge=1, le=14),
    limit: int = Query(default=100, ge=1, le=200),
):
    fixtures = await fixture_aggregator.get_upcoming_fixtures(days=days, limit=limit)
    return await run_in_threadpool(_annotate_fixture_readiness, fixtures)


@router.post(
    "/fixtures/history/sync-missing",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_permission("analysis:create"))],
)
async def sync_missing_fixture_history() -> dict[str, object]:
    """Queue a bounded history backfill after the fixture panel is loaded."""
    cooldown_key = "missing-history-sync:v1:7:100"
    if await cache.get("fixtures", cooldown_key):
        return {"queued": False, "reason": "cooldown"}

    # Refreshes and concurrent users share this cooldown. The worker also uses
    # a distributed lock as a second idempotency boundary.
    await cache.set("fixtures", cooldown_key, True, 900)
    try:
        from app.tasks.jobs import sync_missing_fixture_history_task

        task = sync_missing_fixture_history_task.delay()
    except Exception as exc:
        await cache.delete("fixtures", cooldown_key)
        logger.exception("Fixture history synchronization could not be queued")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Geçmiş maç senkronizasyonu başlatılamadı.",
        ) from exc
    return {"queued": True, "task_id": task.id}


@router.get(
    "/fixtures/{fixture_id}/prefill",
    dependencies=[Depends(require_authenticated_user)],
)
async def fixture_prefill(fixture_id: int):
    payload = await fixture_context_service.get_or_create(
        fixture_id,
        loader=fixture_aggregator.get_fixture_prefill,
        enricher=odds_history_service.enrich_prefill,
    )
    if not payload:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")
    return payload


@router.post("/analyze", dependencies=[Depends(require_permission("analysis:create"))])
async def analyze_manual(payload: AnalysisRequest):
    try:
        return await _run_analysis(
            payload,
            analysis_origin="scenario" if payload.feature_overrides else "manual",
            interactive=True,
        )
    except SQLAlchemyError as exc:
        logger.exception("Veritabanı hatası (manuel analiz)")
        raise HTTPException(status_code=500, detail="Veritabanı hatası.") from exc


@router.post(
    "/predict/tiered",
    dependencies=[Depends(require_permission("audit:read"))],
)
def predict_with_tiered_model(
    payload: TieredPredictionRequest,
    predictor: "Predictor" = Depends(get_tiered_predictor),
) -> dict[str, object]:
    """Run the signed tier bundle for research; never emit an actionable bet."""
    try:
        prediction = predictor.predict(
            {**payload.features, "league_id": payload.league_id}
        )
    except TieredArtifactIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="İmzalı tier modeli kullanıma hazır değil.",
        ) from exc
    except (TypeError, ValueError) as exc:
        logger.warning("Tiered prediction rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="İşleme hatası.",
        ) from exc

    used_tier = "Tier 1" if prediction.tier == "tier1" else "Tier 2"
    tier2_passed = not prediction.research_only
    decision = PredictionDecisionPolicy.evaluate(
        {
            "all_probabilities": {
                "AWAY_WIN": prediction.probabilities[0],
                "DRAW": prediction.probabilities[1],
                "HOME_WIN": prediction.probabilities[2],
            }
        },
        data_quality_score=70.0 if tier2_passed else None,
    )
    return {
        "decision_use": "research_only",
        "used_tier": used_tier,
        "confidence_scores": {
            "0": prediction.probabilities[0],
            "1": prediction.probabilities[1],
            "2": prediction.probabilities[2],
        },
        "confidence": max(prediction.probabilities),
        "decision_status": decision["status"],
        "decision_reasons": decision["reasons"],
        "confidence_tier": decision["confidence_tier"],
        "uncertainty": decision,
        "artifact_version": prediction.artifact_version,
        "research_only": prediction.research_only,
        "tier2_gate": _tiered_gate_evidence(predictor),
    }


@router.post(
    "/predictions/batch",
    dependencies=[Depends(require_permission("predictions:create"))],
)
async def batch_predict(
    fixture_ids: list[int] = Body(..., min_length=1, max_length=20),
    user: CurrentUser = Depends(require_authenticated_user),
) -> dict[str, object]:
    if len(set(fixture_ids)) != len(fixture_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Aynı maç bir toplu istekte birden fazla gönderilemez.",
        )
    user_key = user.id
    allowed, retry_after = await run_in_threadpool(
        batch_prediction_rate_limiter.consume, user_key
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Çok fazla toplu istek. Bir dakika içinde tekrar deneyin.",
            headers={"Retry-After": str(retry_after)},
        )
    semaphore = asyncio.Semaphore(2)

    async def predict_fixture(fid: int) -> dict[str, object]:
        try:
            async with semaphore:
                prefill = await fixture_context_service.get_or_create(
                    fid,
                    loader=fixture_aggregator.get_fixture_prefill,
                    enricher=odds_history_service.enrich_prefill,
                )
                if prefill is None:
                    return {
                        "fixture_id": fid,
                        "status": "not_found",
                        "prediction": None,
                    }
                payload = _build_payload_from_prefill(prefill)
                computed = await _compute_analysis(payload, interactive=False)

            analysis = computed["analysis"]
            probabilities = analysis["all_probabilities"]
            if not isinstance(probabilities, dict):
                raise ValueError("Analysis returned invalid probabilities")
            return {
                "fixture_id": fid,
                "status": "ok",
                "prediction": analysis.get("prediction"),
                "probabilities": probabilities,
                "model": analysis.get("model"),
                "ml_ready": bool(computed["ml_result"].get("ready")),
            }
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Batch prediction rejected fixture_id=%s: %s", fid, exc)
        except Exception:
            logger.exception("Batch prediction failed fixture_id=%s", fid)
        return {"fixture_id": fid, "status": "error", "prediction": None}

    results = await asyncio.gather(*(predict_fixture(fid) for fid in fixture_ids))
    return {"predictions": results, "count": len(results)}


@router.post(
    "/analyze/preview",
    dependencies=[Depends(require_permission("analysis:create"))],
)
async def preview_analysis_inputs(payload: AnalysisRequest):
    """Return every point-in-time model input without persisting a prediction."""
    try:
        computed = await _compute_analysis(payload, interactive=True)
    except (TypeError, ValueError) as exc:
        logger.warning("Preview analysis rejected: %s", exc)
        raise HTTPException(status_code=422, detail="İşleme hatası.") from exc

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
    prefill = await fixture_context_service.get_or_create(
        fixture_id,
        loader=fixture_aggregator.get_fixture_prefill,
        enricher=odds_history_service.enrich_prefill,
    )
    if not prefill:
        raise HTTPException(status_code=404, detail="Maç bulunamadı.")

    await _enrich_with_remote_market(prefill)

    try:
        payload = _build_payload_from_prefill(prefill)
    except (KeyError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="Geçersiz maç verisi.") from exc

    try:
        result = await _run_analysis(
            payload, analysis_origin="fixture_user", interactive=True
        )
    except SQLAlchemyError as exc:
        logger.exception("Veritabanı hatası (fixture_id=%s)", fixture_id)
        raise HTTPException(status_code=500, detail="Veritabanı hatası.") from exc

    result["prefill"] = prefill
    result["data_methodology"] = prefill.get("data_methodology")
    result["prefill_data_quality"] = prefill.get("data_quality")
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
    response_model=MLStatusResponse,
    dependencies=[Depends(require_permission("history:read"))],
)
def get_ml_status(db: Session = Depends(get_db)) -> dict[str, Any]:
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
    active_artifact_version = result.get("artifact_version")
    monitoring = ModelMonitoringService(db).snapshot(
        active_artifact_version if isinstance(active_artifact_version, str) else None
    )
    result["monitoring"] = monitoring
    verified_samples = int(monitoring["samples"])
    required_samples = int(monitoring["required_samples"])
    claims_enabled = (
        bool(active_artifact_version) and verified_samples >= required_samples
    )
    result["live_evaluation"] = {
        "status": (
            "available"
            if claims_enabled
            else "insufficient_data"
            if active_artifact_version
            else "model_unavailable"
        ),
        "verified_samples": verified_samples,
        "required_samples": required_samples,
        "claims_enabled": claims_enabled,
        "artifact_version": (
            active_artifact_version
            if isinstance(active_artifact_version, str)
            else None
        ),
    }
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
    kickoff = record.kickoff
    if kickoff is not None:
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        if kickoff > datetime.now(timezone.utc):
            raise HTTPException(
                status_code=409,
                detail="Maç başlamadan gerçek sonuç girilemez.",
            )

    roi = PredictionAuditor.calculate_bet_roi(
        record.prediction, body.actual_result, record.odd
    )

    updated = repo.update_result(
        record_id=record_id,
        actual_result=body.actual_result,
        actual_score_home=body.actual_score_home,
        actual_score_away=body.actual_score_away,
        roi=roi,
        verification_status="manual",
        result_source="manual_admin",
        verification_note="Manually entered result; excluded from model training",
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
        "actual_result": updated.actual_result if updated else body.actual_result,
        "result_verification_status": (
            updated.result_verification_status if updated else "manual"
        ),
        "labeled_samples_count": labeled_samples_count,
        "remaining_to_threshold": max(
            0, settings.MIN_TRAINING_SAMPLES - labeled_samples_count
        ),
        "worker": worker,
    }


@router.post("/backtest", dependencies=[Depends(require_permission("backtest:run"))])
def run_backtest(body: BacktestRequest, db: Session = Depends(get_db)):
    repo = MatchPredictionRepository(db)
    predictions = repo.get_all_auditable(limit=body.limit)

    result = BacktestEngine.run_simulation(
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

    minimum_bets = settings.AUDIT_MIN_CLOSING_SAMPLES
    minimum_roi_pct = settings.AUDIT_MIN_CLOSING_ROI_PCT
    gate_reasons: list[str] = []
    gate = False
    if not body.require_closing_odds:
        gate_reasons.append("closing_reference_not_enforced")
    if result["total_bets"] < minimum_bets:
        gate_reasons.append("insufficient_closing_odds_bets")
    if result["total_roi_pct"] < minimum_roi_pct:
        gate_reasons.append("closing_roi_below_threshold")
    if not gate_reasons:
        gate = True
    result["closing_gate"] = {
        "passed": gate,
        "reasons": gate_reasons,
        "closing_bets": result["total_bets"],
        "closing_roi_pct": result["total_roi_pct"],
        "minimum_bets": minimum_bets,
        "minimum_roi_pct": minimum_roi_pct,
        "evidence": "financial signals stay disabled until the closing-odds "
        "backtest is both large enough and positive.",
    }
    return result


@router.get("/audit", dependencies=[Depends(require_permission("audit:read"))])
def run_audit(
    limit: int = Query(default=5000, ge=1, le=50000),
    artifact_version: str | None = Query(default=None, min_length=1, max_length=100),
    db: Session = Depends(get_db),
):
    repo = MatchPredictionRepository(db)
    selected_artifact = artifact_version or ml_pipeline.artifact_version
    result = PredictionAuditor.audit_predictions(
        repo.get_all_auditable(
            limit=limit,
            model_artifact_version=selected_artifact,
        )
    )
    result["artifact_version"] = selected_artifact
    return result


@router.get(
    "/audit/leagues",
    dependencies=[Depends(require_permission("audit:read"))],
)
def run_league_audit(
    limit: int = Query(default=5000, ge=1, le=50000),
    artifact_version: str | None = Query(default=None, min_length=1, max_length=100),
    db: Session = Depends(get_db),
):
    repo = MatchPredictionRepository(db)
    selected_artifact = artifact_version or ml_pipeline.artifact_version
    result = PredictionAuditor.audit_by_league(
        repo.get_all_auditable(
            limit=limit,
            model_artifact_version=selected_artifact,
        )
    )
    result["artifact_version"] = selected_artifact
    return result


@router.get(
    "/operations/data-quality",
    response_model=DataQualityResponse,
    dependencies=[Depends(require_permission("audit:read"))],
)
async def get_data_quality(db: Session = Depends(get_db)) -> dict[str, Any]:
    from app.services.api_provider_health import api_football_health

    snapshot = await run_in_threadpool(DataQualityService(db).snapshot)
    snapshot["providers"] = {
        "api_football": await api_football_health.snapshot(),
        "sportmonks": {
            "status": "configured" if settings.SPORTMONKS_ENABLED else "disabled",
            "enabled": settings.SPORTMONKS_ENABLED,
        },
    }
    return snapshot
