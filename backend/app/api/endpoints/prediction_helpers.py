from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Tuple

from fastapi import HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.db.session import SessionLocal
from app.db.repository import MatchPredictionRepository
from app.db.historical_repository import HistoricalFixtureRepository
from app.db.player_context_repository import PlayerContextRepository
from app.services.api_football import APIFootballClient
from app.services.cloudflare_odds_collector import (
    CloudflareOddsCollectorError,
    cloudflare_odds_collector,
)
from app.services.data_quality import AnalysisQualityScorer
from app.services.external_features import external_feature_service
from app.services.financial_recommendations import financial_recommendation_service
from app.services.sportmonks_players import sportmonks_player_service
from app.services.travel_context import travel_context_service
from app.providers.open_meteo import (
    OpenMeteoClient,
    OpenMeteoError,
    WeatherObservation,
)
from app.prediction.stats_engine import StatsEngine
from app.prediction.ensemble import ProbabilityEnsembler
from app.prediction.value_calc import ValueCalc
from app.prediction.ml.model import ml_pipeline
from app.prediction.ml.model_router import (
    Predictor,
    TieredArtifactIntegrityError,
    get_active_tiered_predictor,
)
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import (
    HistoricalFeatureContext,
    HistoricalFeatureService,
    PlayerRatingValue,
)
from app.prediction.player_impact import PlayerImpactCalculator
from app.prediction.eligibility import (
    PredictionEligibilityPolicy,
    PredictionIneligibleError,
)
from app.prediction.decision import PredictionDecisionPolicy
from app.prediction.input_catalog import AnalysisInputCatalog
from app.prediction.ml.explain import ExplainabilityService
from app.core.config import settings
from app.api.endpoints.schemas import (
    TeamStatsInput,
    AnalysisRequest,
)

logger = logging.getLogger("bet-ai-pro.api")
football_api = APIFootballClient()
open_meteo_client = OpenMeteoClient()


def _market_settlement_price(
    market: object, analysis: Mapping[str, object]
) -> float | None:
    """Return the single honest settlement price for a recorded forecast.

    A bet may only be priced from a real market snapshot, never from a price
    typed into the form. When no market exists nothing is recorded as a price,
    so fabricated odds cannot leak into audits or backtests as realized profit.
    """
    if not isinstance(market, Mapping):
        return None
    raw_odds = market.get("raw_odds")
    if not isinstance(raw_odds, Mapping):
        return None

    def valid(value: object) -> float | None:
        if not isinstance(value, (int, float, str)) or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) and parsed > 1.0 else None

    prediction = analysis.get("prediction")
    selected = valid(raw_odds.get(prediction)) if prediction is not None else None
    if selected is not None:
        return round(selected, 4)
    available = [
        candidate
        for outcome in raw_odds
        if (candidate := valid(raw_odds[outcome])) is not None
    ]
    if not available:
        return None
    return round(min(available), 4)


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
        kelly_fraction=prefill.get("kelly_fraction", 0.25),
        market_1x2=market or None,
        opening_odds_1x2=prefill.get("opening_odds_1x2"),
        current_odds_1x2=prefill.get("current_odds_1x2"),
        opening_odds_at=prefill.get("opening_odds_at"),
        current_odds_at=prefill.get("current_odds_at"),
        fixture_id=fixture.get("fixture_id"),
        fixture_source=fixture.get("source"),
        provider_fixture_id=fixture.get("provider_fixture_id"),
        home_team_id=fixture.get("home_team_id"),
        away_team_id=fixture.get("away_team_id"),
        league_id=fixture.get("league_id"),
        season=fixture.get("season"),
        kickoff=fixture.get("kickoff"),
        away_travel_distance_km=fixture.get("away_travel_distance_km"),
    )


def _probability_favorite(probabilities: object) -> str | None:
    if not isinstance(probabilities, dict):
        return None
    outcomes = ("HOME_WIN", "DRAW", "AWAY_WIN")
    try:
        values = {outcome: float(probabilities[outcome]) for outcome in outcomes}
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        return None
    return max(outcomes, key=lambda outcome: values[outcome])


def _assess_ml_safety(
    ml_result: dict,
    analysis: dict,
    data_quality: dict,
) -> dict[str, object]:
    """Classify confidence using probability, source agreement and market context."""
    if not ml_result.get("ready"):
        return {"trigger": "INSUFFICIENT_DATA"}

    raw_probabilities = ml_result.get("all_probabilities")
    ml_prediction = _probability_favorite(raw_probabilities)
    if ml_prediction is None or not isinstance(raw_probabilities, dict):
        return {"trigger": "INSUFFICIENT_DATA"}

    probabilities = sorted(
        (
            float(raw_probabilities[outcome])
            for outcome in ("HOME_WIN", "DRAW", "AWAY_WIN")
        ),
        reverse=True,
    )
    confidence = probabilities[0]
    confidence_gap = confidence - probabilities[1]
    components = (analysis.get("ensemble") or {}).get("components") or {}
    stats_favorite = _probability_favorite(components.get("stats"))
    market_favorite = _probability_favorite(components.get("market"))
    ensemble_favorite = _probability_favorite(analysis.get("all_probabilities"))
    model_agreement = all(
        favorite == ml_prediction
        for favorite in (stats_favorite, ensemble_favorite)
        if favorite is not None
    )
    quality_score = float(data_quality.get("score", 100.0))
    market_disagreement = (
        market_favorite is not None and market_favorite != ml_prediction
    )
    weak_signal = (
        confidence < 45.0
        or confidence_gap < 8.0
        or quality_score < 50.0
        or not model_agreement
    )

    if market_disagreement:
        trigger = "RISKY_UPSET" if weak_signal else "UPSET_CANDIDATE"
    elif weak_signal:
        trigger = "LOW_CONFIDENCE"
    elif confidence >= 60.0 and confidence_gap >= 15.0:
        trigger = "HIGH_CONFIDENCE"
    else:
        trigger = "MEDIUM_CONFIDENCE"

    return {
        "trigger": trigger,
        "ml_prediction": ml_prediction,
        "ml_confidence": round(confidence, 2),
        "confidence_gap": round(confidence_gap, 2),
        "market_favorite": market_favorite,
        "stats_favorite": stats_favorite,
        "ensemble_favorite": ensemble_favorite,
        "model_agreement": model_agreement,
        "data_quality_score": round(quality_score, 2),
    }


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
    data_quality: dict,
) -> dict:
    prediction_labels = {
        "HOME_WIN": "Ev Sahibi Galibiyeti",
        "AWAY_WIN": "Deplasman Galibiyeti",
        "DRAW": "Beraberlik",
    }

    ml_assessment = _assess_ml_safety(ml_result, analysis, data_quality)
    model_training_samples = int(ml_result.get("training_samples", 0) or 0)
    reported_ml_samples = max(labeled_samples_count, model_training_samples)
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
        "ml_safety_trigger": ml_assessment["trigger"],
        "ml_safety_details": ml_assessment,
        "decision_status": (analysis.get("decision") or {}).get("status", "abstain"),
        "decision_reasons": (analysis.get("decision") or {}).get("reasons", []),
        "uncertainty": analysis.get("decision"),
        "ml_confidence": (
            ml_result.get("probability", 0.0) if ml_result.get("ready") else 0.0
        ),
        "ml_ready": ml_result.get("ready", False),
        "ml_samples": reported_ml_samples,
        "ml_min_samples": settings.MIN_TRAINING_SAMPLES,
        "ml_sample_source": (
            "active_model_training_set"
            if model_training_samples > 0
            else "verified_prediction_results"
        ),
        "labeled_samples_count": labeled_samples_count,
        "remaining_to_threshold": max(
            0, settings.MIN_TRAINING_SAMPLES - reported_ml_samples
        ),
        "insights": insights,
    }


async def _fetch_ml_match_data(
    payload: AnalysisRequest, historical: HistoricalFeatureContext
) -> Tuple[Any, Any, Any, Any, Any]:
    """Resolve point-in-time match inputs.

    Recent form and head-to-head are determined exclusively from local
    historical fixtures; no live provider data is used to decide the fixture
    outcome. Live data is only fetched for availability and lineups.
    """
    home_team_id = payload.home_team_id
    away_team_id = payload.away_team_id
    if home_team_id is None or away_team_id is None:
        return historical.home_matches_df, historical.away_matches_df, None, None, None

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

    availability_data, lineup_data = await asyncio.gather(
        availability(),
        lineups(),
    )
    return (
        historical.home_matches_df,
        historical.away_matches_df,
        historical.h2h_rates,
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


async def _fetch_match_weather(
    payload: AnalysisRequest,
    historical: HistoricalFeatureContext,
) -> WeatherObservation | None:
    if (
        payload.kickoff is None
        or historical.venue_latitude is None
        or historical.venue_longitude is None
    ):
        return None
    try:
        return await open_meteo_client.get_weather(
            latitude=historical.venue_latitude,
            longitude=historical.venue_longitude,
            at=payload.kickoff,
        )
    except (OpenMeteoError, ValueError) as exc:
        logger.warning("Open-Meteo weather lookup failed: %s", type(exc).__name__)
        return None


_INTERACTIVE_EXCLUDED_CHECKS = frozenset(
    {
        "fixture_identified",
        "fixture_source_identified",
        "provider_fixture_identified",
        "market_available",
        # Pre-match modules are naturally absent before kickoff; excluding them
        # from the interactive score keeps "Sınırlı veri" honest.
        "availability_available",
        "lineups_available",
        "home_player_impact_available",
        "away_player_impact_available",
        "odds_movement_available",
        "weather_available",
    }
)


async def _ensure_live_market(payload: AnalysisRequest) -> AnalysisRequest:
    """Fetch the fixture's 1X2 market on demand when the collector missed it."""
    if payload.market_1x2 is not None:
        return payload
    fixture_id = payload.fixture_id
    if not isinstance(fixture_id, int) or fixture_id <= 0:
        return payload
    market = await football_api.get_fixture_market(fixture_id)
    if not isinstance(market, dict) or not market.get("fair_probability"):
        return payload
    return payload.model_copy(update={"market_1x2": market})


def _predict_and_explain(feature_vector: dict[str, float]) -> tuple[dict, List[str]]:
    """Run CPU-bound model inference away from the ASGI event loop."""
    if not ml_pipeline.is_ready:
        return {"ready": False}, []
    ml_result = ml_pipeline.predict_match(feature_vector)
    if not ml_result.get("ready"):
        return ml_result, []
    return (
        ml_result,
        ExplainabilityService.generate_explanation(
            ml_pipeline.model,
            feature_vector,
            ml_pipeline.feature_names,
        ),
    )


async def _compute_analysis(
    payload: AnalysisRequest, *, interactive: bool = False
) -> dict:
    """Run analysis with external inputs and a short point-in-time history read."""
    payload = await _ensure_live_market(payload)
    home_stats = payload.home_stats.model_dump()
    away_stats = payload.away_stats.model_dump()

    ml_result: dict = {"ready": False}
    ml_explanations: List[str] = []

    historical_context = await run_in_threadpool(
        _get_historical_feature_context,
        payload,
    )
    historical = await _apply_external_elo_fallback(payload, historical_context)
    historical = await _apply_external_travel_fallback(payload, historical)
    weather_observation = await _fetch_match_weather(payload, historical)
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
    stats_analysis = await run_in_threadpool(
        StatsEngine.analyze_match,
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
    calculated_feature_vector = await run_in_threadpool(
        FeatureEngine.build_inference_features,
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
        weather=(
            weather_observation.features() if weather_observation is not None else None
        ),
    )
    feature_vector = {
        name: float(
            payload.feature_overrides.get(name, calculated_feature_vector[name])
        )
        for name in FeatureEngine.FEATURE_NAMES
    }
    if any(not math.isfinite(value) for value in feature_vector.values()):
        raise ValueError("Feature vector contains non-finite values")
    ml_result, ml_explanations = await run_in_threadpool(
        _predict_and_explain,
        feature_vector,
    )

    analysis = ProbabilityEnsembler.apply(
        stats_analysis,
        ml_result=ml_result,
        market=payload.market_1x2,
        league_id=payload.league_id,
    )
    # Value must be estimated from a market-independent forecast. Blending the
    # market into the forecast and comparing it back to the same market is circular.
    value_analysis = ProbabilityEnsembler.apply(
        stats_analysis,
        ml_result=ml_result,
        market=None,
        league_id=payload.league_id,
    )
    value_data = ValueCalc.calculate_professional(
        value_analysis,
        payload.market_1x2,
        fallback_odd=payload.odd,
        kelly_fraction=payload.kelly_fraction,
    )
    value_data["probability_source"] = "market_independent_stats_ml_ensemble"
    value_data["evaluation_probabilities"] = value_analysis["all_probabilities"]
    market_is_present = payload.market_1x2 is not None
    analysis["decision"] = {
        "status": "research",
        "reasons": [],
        "confidence_tier": "medium",
    }
    financial_decision = financial_recommendation_service.evaluate()
    value_data["recommendation_evidence"] = financial_decision
    # Without a live 1X2 price there is nothing to hold the model to account
    # against, so the bet signal must stay research-only even if past audits
    # passed. A missing market is never a recommendation to act.
    market_unavailable = payload.market_1x2 is None
    if market_unavailable:
        value_data = ValueCalc.suppress_financial_recommendations(
            value_data, reason="market_unavailable"
        )
    elif financial_decision.get("eligible") is not True:
        reasons = financial_decision.get("reasons")
        reason = (
            str(reasons[0])
            if isinstance(reasons, list) and reasons
            else "financial_validation_failed"
        )
        value_data = ValueCalc.suppress_financial_recommendations(
            value_data, reason=reason
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
    home_lineup = lineups.get("home_starting_xi") if isinstance(lineups, dict) else None
    away_lineup = lineups.get("away_starting_xi") if isinstance(lineups, dict) else None
    quality_checks = {
        "fixture_identified": payload.fixture_id is not None,
        "fixture_source_identified": bool(payload.fixture_source),
        "provider_fixture_identified": bool(payload.provider_fixture_id),
        "league_identified": payload.league_id is not None,
        "kickoff_known": payload.kickoff is not None,
        "market_available": payload.market_1x2 is not None,
        "h2h_available": bool(historical.h2h_matches),
        "home_history_available": history_home_count > 0,
        "away_history_available": history_away_count > 0,
        "home_history_sufficient": history_home_count >= required_history,
        "away_history_sufficient": history_away_count >= required_history,
        "local_match_outcome_available": bool(
            historical.h2h_matches
            or (
                history_home_count >= required_history
                and history_away_count >= required_history
            )
        ),
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
        "weather_available": weather_observation is not None,
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
    if weather_observation is not None:
        weather_provenance: dict[str, object] = {
            "source": weather_observation.source,
            "captured_at": weather_observation.fetched_at.isoformat(),
            "confidence": 0.85,
            "is_fallback": weather_observation.source == "open_meteo_archive",
        }
        feature_provenance.update(
            {
                feature_name: weather_provenance
                for feature_name in AnalysisInputCatalog.WEATHER_INPUTS
            }
        )
    data_quality = {
        "score": AnalysisQualityScorer.score(quality_checks),
        "score_method": "weighted_input_coverage_v1",
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
        "weather": (
            {
                **weather_observation.features(),
                "source": weather_observation.source,
                "observed_at": weather_observation.observed_at.isoformat(),
                "fetched_at": weather_observation.fetched_at.isoformat(),
            }
            if weather_observation is not None
            else None
        ),
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
        "analysis_outputs": {
            "expected_goals": analysis.get("expected_goals"),
            "expected_score": analysis.get("expected_score"),
            "score_band": analysis.get("score_band"),
            "secondary_markets": analysis.get("secondary_markets", []),
            "match_profile": analysis.get("match_profile"),
        },
        "financial_recommendation": {
            "status": value_data.get("recommendation_status", "eligible"),
            "reason": value_data.get("recommendation_reason"),
            "evidence": value_data.get("recommendation_evidence"),
        },
    }
    data_quality["ml_assessment"] = _assess_ml_safety(ml_result, analysis, data_quality)
    if interactive:
        data_quality["interactive_score"] = AnalysisQualityScorer.score(
            quality_checks,
            excluded=_INTERACTIVE_EXCLUDED_CHECKS,
        )
    data_quality["prediction_eligibility"] = PredictionEligibilityPolicy.evaluate(
        data_quality, interactive=interactive
    ).as_dict()

    raw_dq_score = data_quality.get("score", 100.0)
    try:
        dq_score = (
            float(raw_dq_score)
            if isinstance(raw_dq_score, (int, float, str))
            and not isinstance(raw_dq_score, bool)
            else 100.0
        )
    except ValueError:
        dq_score = 100.0
    decision_recommendation = PredictionDecisionPolicy.evaluate(
        analysis,
        market_edge_pct=None if not market_is_present else value_data.get("edge"),
        market_implied_pct=(
            None if not market_is_present else value_data.get("implied_probability")
        ),
        require_market=True,
        market_confirmed=(
            market_is_present
            and (value_data.get("edge") or 0) >= settings.DECISION_MIN_MARKET_EDGE_PCT
        ),
        data_quality_score=dq_score,
    )
    analysis["decision"] = decision_recommendation
    # Persist the final policy decision, not the provisional stats-engine value.
    data_quality["decision_recommendation"] = decision_recommendation

    return {
        "analysis": analysis,
        "value_data": value_data,
        "ml_result": ml_result,
        "feature_vector": feature_vector,
        "calculated_feature_vector": calculated_feature_vector,
        "insights": insights,
        "data_quality": data_quality,
    }


def _persist_analysis(
    payload: AnalysisRequest,
    computed: dict,
    *,
    analysis_origin: str,
    training_eligible: bool,
):
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
    model_name = ml_result.get("model_name") or analysis.get("model")
    ensemble_version = (analysis.get("ensemble") or {}).get("version")
    provider_market_source = (
        payload.market_1x2.get("source")
        if isinstance(payload.market_1x2, dict)
        else None
    )
    market_source = (
        str(provider_market_source)
        if provider_market_source
        else (
            "api_football_odds"
            if payload.market_1x2 is not None
            and payload.fixture_source == "api_football"
            else "request_payload" if payload.market_1x2 is not None else "unavailable"
        )
    )
    provenance_manifest = {
        "schema_version": "prediction_provenance_v1",
        "fixture": {
            "fixture_id": payload.fixture_id,
            "fixture_source": payload.fixture_source or "composite_identity",
            "provider_fixture_id": payload.provider_fixture_id,
            "league_id": payload.league_id,
            "kickoff": kickoff.isoformat() if kickoff is not None else None,
        },
        "analysis": {
            "model_name": model_name,
            "model_artifact_version": ml_result.get("artifact_version")
            or "not_applicable",
            "ensemble_version": ensemble_version,
        },
        "market": {
            "available": payload.market_1x2 is not None,
            "source": market_source,
            "snapshot_at": (
                (payload.current_odds_at or analyzed_at).isoformat()
                if payload.market_1x2 is not None
                else None
            ),
            "opening_snapshot_at": (
                payload.opening_odds_at.isoformat()
                if payload.opening_odds_at is not None
                else None
            ),
        },
        "features": {
            "schema_version": FeatureEngine.SCHEMA_VERSION,
            "snapshot_at": analyzed_at.isoformat(),
            "sources": data_quality.get("feature_provenance", {}),
            "manual_overrides": sorted(payload.feature_overrides),
        },
        "decision": {
            "analysis_origin": analysis_origin,
            "data_eligibility_status": (
                data_quality.get("prediction_eligibility") or {}
            ).get("status"),
            "forecast_decision_status": (analysis.get("decision") or {}).get("status"),
            "forecast_decision_reasons": (analysis.get("decision") or {}).get(
                "reasons", []
            ),
            "training_eligible": training_eligible,
        },
    }

    record_data = {
        "fixture_id": payload.fixture_id,
        "fixture_source": payload.fixture_source,
        "provider_fixture_id": payload.provider_fixture_id,
        "home_team": payload.home_team,
        "away_team": payload.away_team,
        "home_team_id": payload.home_team_id,
        "away_team_id": payload.away_team_id,
        "league_id": payload.league_id,
        "analysis_origin": analysis_origin,
        "eligibility_status": (
            "eligible"
            if (data_quality.get("prediction_eligibility") or {}).get("status")
            == "eligible"
            and (analysis.get("decision") or {}).get("status") == "eligible"
            else "abstain"
        ),
        "training_eligible": training_eligible,
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
        "odd": _market_settlement_price(payload.market_1x2, analysis),
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
        "ensemble_version": ensemble_version,
        "model_name": model_name,
        "model_artifact_version": ml_result.get("artifact_version"),
        "data_quality": data_quality,
        "provenance_manifest": provenance_manifest,
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


def _is_training_eligible(
    computed: dict,
    payload: AnalysisRequest,
    analysis_origin: str,
) -> bool:
    """A prediction is trainable when its data is complete and the fixture can
    be deterministically identified so a verified result can later be attached.

    Provider identity (API fixture id) is not required: the composite key
    (league + both teams + kickoff) is equally deterministic. Scenario runs and
    manual feature overrides are never training sample material. Forecast
    uncertainty deliberately does not exclude a row: doing so would bias future
    calibration and evaluation toward easy, high-confidence matches.
    """
    if payload.feature_overrides or analysis_origin not in {
        "automatic",
        "fixture_user",
    }:
        return False
    eligibility = (computed.get("data_quality") or {}).get(
        "prediction_eligibility"
    ) or {}
    if eligibility.get("status") != "eligible":
        return False
    has_fixture_identity = bool(
        payload.provider_fixture_id
        or (
            bool(payload.home_team)
            and bool(payload.away_team)
            and bool(payload.league_id)
            and payload.kickoff is not None
        )
        or bool(payload.fixture_id and payload.fixture_source)
    )
    has_market_evidence = bool(
        payload.market_1x2
        and payload.current_odds_1x2
        and payload.current_odds_at
        and payload.kickoff
        and payload.current_odds_at < payload.kickoff
    )
    return has_fixture_identity and has_market_evidence


async def _enrich_with_remote_market(prefill: dict[str, Any]) -> None:
    """Register the fixture remotely and fill a missing point-in-time market."""
    if not cloudflare_odds_collector.enabled:
        return
    fixture = prefill.get("fixture")
    if not isinstance(fixture, Mapping):
        return
    fixture_id = fixture.get("fixture_id")
    league_id = fixture.get("league_id")
    kickoff = fixture.get("kickoff")
    if (
        isinstance(fixture_id, bool)
        or not isinstance(fixture_id, int)
        or fixture_id <= 0
        or isinstance(league_id, bool)
        or not isinstance(league_id, int)
        or league_id <= 0
        or not isinstance(kickoff, datetime)
    ):
        return

    existing_market = prefill.get("market_1x2")
    preferred_bookmaker = (
        existing_market.get("bookmaker")
        if isinstance(existing_market, Mapping)
        and isinstance(existing_market.get("bookmaker"), str)
        else None
    )
    try:
        result = await cloudflare_odds_collector.track_fixture(
            fixture_id=fixture_id,
            fixture_source=(
                str(fixture["fixture_source"])
                if fixture.get("fixture_source")
                else None
            ),
            provider_fixture_id=(
                str(fixture["provider_fixture_id"])
                if fixture.get("provider_fixture_id")
                else None
            ),
            league_id=league_id,
            home_team=str(prefill.get("home_team") or ""),
            away_team=str(prefill.get("away_team") or ""),
            kickoff=kickoff,
            preferred_bookmaker=preferred_bookmaker,
            capture_entry=not isinstance(existing_market, Mapping),
        )
    except CloudflareOddsCollectorError:
        logger.warning(
            "Remote odds collector could not register fixture_id=%s", fixture_id
        )
        return

    entry = cloudflare_odds_collector.valid_entry(
        result.get("entry") if isinstance(result, Mapping) else None
    )
    if entry is None or isinstance(existing_market, Mapping):
        return
    raw_odds = entry["raw_odds"]
    if not isinstance(raw_odds, Mapping):
        return
    market = ValueCalc.devig_1x2(
        float(raw_odds["HOME_WIN"]),
        float(raw_odds["DRAW"]),
        float(raw_odds["AWAY_WIN"]),
    )
    market["bookmaker"] = entry["bookmaker"]
    market["source"] = "cloudflare_api_football_odds"
    market["payload_sha256"] = entry.get("payload_sha256")
    prefill["market_1x2"] = market
    prefill["current_odds_1x2"] = dict(raw_odds)
    prefill["current_odds_at"] = entry["captured_at"]
    prefill["odd"] = float(raw_odds["HOME_WIN"])


async def _run_analysis(
    payload: AnalysisRequest,
    *,
    require_eligible: bool = False,
    analysis_origin: str = "manual",
    interactive: bool = False,
) -> dict:
    computed = await _compute_analysis(payload, interactive=interactive)
    eligibility = PredictionEligibilityPolicy.evaluate(
        computed["data_quality"], interactive=interactive
    )
    computed["data_quality"]["prediction_eligibility"] = eligibility.as_dict()
    strict_decision = PredictionEligibilityPolicy.evaluate(
        computed["data_quality"], interactive=False
    )
    if require_eligible and not strict_decision.eligible:
        raise PredictionIneligibleError(strict_decision)
    # Interactive eligibility controls presentation only. Training admission
    # always uses the strict production policy so relaxed UI rules cannot leak
    # incomplete samples into evaluation or retraining.
    training_eligible = strict_decision.eligible and _is_training_eligible(
        computed,
        payload,
        analysis_origin,
    )
    db_record, labeled_samples_count = await run_in_threadpool(
        _persist_analysis,
        payload,
        computed,
        analysis_origin=analysis_origin,
        training_eligible=training_eligible,
    )

    response = _build_analysis_response(
        db_record.id,
        payload.home_team,
        payload.away_team,
        computed["analysis"],
        computed["value_data"],
        computed["ml_result"],
        computed["insights"],
        labeled_samples_count,
        computed["data_quality"],
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
        "analysis_origin": db_record.analysis_origin,
        "eligibility_status": db_record.eligibility_status,
        "training_eligible": db_record.training_eligible,
    }
    response["feature_snapshot"] = computed["feature_vector"]
    return response


def get_tiered_predictor() -> Predictor:
    """Return the currently active signed tiered predictor (singleton cache)."""
    try:
        return get_active_tiered_predictor()
    except TieredArtifactIntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="İmzalı tier modeli kullanıma hazır değil.",
        ) from exc


def _tiered_gate_evidence(predictor: Predictor) -> dict[str, object] | None:
    """Drop the Tier 2 evidence gate from the router without leaking internals."""
    tier2_gate = getattr(predictor, "tier2_gate", None)
    if not isinstance(tier2_gate, dict):
        return None
    trusted_keys = (
        "passed",
        "reasons",
        "training_samples",
        "minimum_samples",
        "min_per_class",
        "minimum_per_class",
        "class_distribution",
    )
    return {key: tier2_gate.get(key) for key in trusted_keys if key in tier2_gate}
