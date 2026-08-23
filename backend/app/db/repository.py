from datetime import datetime, timedelta, timezone
from typing import List, Literal, Mapping, Optional

from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.db.models import MatchPrediction
from app.core.team_identity import normalize_team_name

NON_MUTABLE_UPSERT_COLUMNS = frozenset(
    {"id", "created_at", "fixture_id", "analysis_origin"}
)
FIXTURE_IDENTITY_COLUMNS = (
    "fixture_id",
    "fixture_source",
    "provider_fixture_id",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "league_id",
    "kickoff",
)


class MatchPredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, record_id: int) -> Optional[MatchPrediction]:
        return (
            self.db.query(MatchPrediction)
            .filter(MatchPrediction.id == record_id)
            .first()
        )

    def get_by_fixture_id(self, fixture_id: int) -> Optional[MatchPrediction]:
        return (
            self.db.query(MatchPrediction)
            .filter(MatchPrediction.fixture_id == fixture_id)
            .first()
        )

    def get_equivalent_fixture(
        self, data: Mapping[str, object]
    ) -> Optional[MatchPrediction]:
        """Resolve the same real fixture across provider-specific identifiers."""
        league_id = data.get("league_id")
        kickoff = data.get("kickoff")
        home_team = data.get("home_team")
        away_team = data.get("away_team")
        if (
            not isinstance(league_id, int)
            or not isinstance(kickoff, datetime)
            or not isinstance(home_team, str)
            or not isinstance(away_team, str)
        ):
            return None

        home_key = normalize_team_name(home_team)
        away_key = normalize_team_name(away_team)
        if not home_key or not away_key:
            return None
        candidates = (
            self.db.query(MatchPrediction)
            .filter(
                MatchPrediction.league_id == league_id,
                MatchPrediction.kickoff >= kickoff - timedelta(hours=2),
                MatchPrediction.kickoff <= kickoff + timedelta(hours=2),
            )
            .order_by(MatchPrediction.id.asc())
            .all()
        )
        return next(
            (
                candidate
                for candidate in candidates
                if normalize_team_name(candidate.home_team or "") == home_key
                and normalize_team_name(candidate.away_team or "") == away_key
            ),
            None,
        )

    @staticmethod
    def _reuse_fixture_identity(
        data: dict[str, object], existing: MatchPrediction
    ) -> dict[str, object]:
        merged = dict(data)
        for column in FIXTURE_IDENTITY_COLUMNS:
            value = getattr(existing, column)
            if value is not None:
                merged[column] = value
        manifest = merged.get("provenance_manifest")
        if isinstance(manifest, dict):
            manifest = dict(manifest)
            fixture = dict(manifest.get("fixture") or {})
            fixture.update(
                fixture_id=existing.fixture_id,
                fixture_source=existing.fixture_source or "composite_identity",
                provider_fixture_id=existing.provider_fixture_id,
                league_id=existing.league_id,
                kickoff=(
                    existing.kickoff.isoformat() if existing.kickoff else None
                ),
            )
            manifest["fixture"] = fixture
            merged["provenance_manifest"] = manifest
        return merged

    def _commit(self) -> None:
        try:
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise

    @classmethod
    def _upsert_update_columns(cls, data: Mapping[str, object]) -> dict[str, object]:
        """Map the incoming fields to PostgreSQL ``excluded`` references.

        Only columns actually present in ``data`` are included. Adding every
        table column here would make PostgreSQL resolve missing columns to their
        DEFAULT (NULL), silently wiping verified labels, ROI and closing-odds
        provenance on every partial re-analysis.
        """
        return {
            col.name: getattr(pg_insert(MatchPrediction).excluded, col.name)
            for col in MatchPrediction.__table__.columns
            if col.name in data and col.name not in NON_MUTABLE_UPSERT_COLUMNS
        }

    def upsert_prediction(self, data: dict) -> MatchPrediction:
        """
        Upsert a prediction.
        Uses high-performance PostgreSQL-native upsert in production.
        Falls back to standard transaction check-and-update on SQLite for local environments.
        """
        data = dict(data)
        fixture_id = data.get("fixture_id")
        existing = (
            self.get_by_fixture_id(fixture_id) if isinstance(fixture_id, int) else None
        )
        equivalent_provider_fixture = existing is None
        if existing is None:
            existing = self.get_equivalent_fixture(data)
        if existing is not None and existing.actual_result is not None:
            # Never rewrite a forecast after its result became known.
            return existing
        if existing is not None and equivalent_provider_fixture:
            data = self._reuse_fixture_identity(data, existing)

        if self.db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")
        dialect = self.db.bind.dialect.name

        if dialect == "sqlite":
            # Dialect-agnostic SQLite fallback
            if existing:
                for k, v in data.items():
                    setattr(existing, k, v)
                self._commit()
                return existing
            else:
                record = MatchPrediction(**data)
                self.db.add(record)
                self._commit()
                self.db.refresh(record)
                return record

        # PostgreSQL native execution
        stmt = pg_insert(MatchPrediction).values(**data)
        update_cols = self._upsert_update_columns(data)

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["fixture_id"], set_=update_cols
        )

        self.db.execute(upsert_stmt)
        self._commit()

        upserted_record = self.get_by_fixture_id(data["fixture_id"])
        if upserted_record is None:
            raise RuntimeError(
                "Upsert completed but the prediction could not be loaded"
            )
        return upserted_record

    def get_all(self) -> List[MatchPrediction]:
        return self.db.query(MatchPrediction).order_by(MatchPrediction.id.asc()).all()

    def get_recent(self, limit: int = 15) -> List[MatchPrediction]:
        return (
            self.db.query(MatchPrediction)
            .order_by(MatchPrediction.id.desc())
            .limit(limit)
            .all()
        )

    def search_history(
        self,
        *,
        page: int,
        page_size: int,
        query: str = "",
        result: Literal["all", "pending", "HOME_WIN", "DRAW", "AWAY_WIN"] = "all",
        value: Literal["all", "value", "non_value"] = "all",
        sort: Literal["newest", "oldest", "edge", "odd"] = "newest",
    ) -> tuple[List[MatchPrediction], int]:
        history_query = self.db.query(MatchPrediction)

        normalized_query = query.strip()
        if normalized_query:
            # Escape SQL wildcard characters so user input is treated literally.
            escaped_query = (
                normalized_query.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = f"%{escaped_query}%"
            history_query = history_query.filter(
                or_(
                    MatchPrediction.home_team.ilike(pattern, escape="\\"),
                    MatchPrediction.away_team.ilike(pattern, escape="\\"),
                )
            )

        if result == "pending":
            history_query = history_query.filter(
                MatchPrediction.actual_result.is_(None)
            )
        elif result != "all":
            history_query = history_query.filter(
                MatchPrediction.actual_result == result
            )

        if value == "value":
            history_query = history_query.filter(MatchPrediction.is_value_bet == 1)
        elif value == "non_value":
            history_query = history_query.filter(
                or_(
                    MatchPrediction.is_value_bet.is_(None),
                    MatchPrediction.is_value_bet != 1,
                )
            )

        total = history_query.count()
        if sort == "oldest":
            history_query = history_query.order_by(
                MatchPrediction.created_at.asc(), MatchPrediction.id.asc()
            )
        elif sort == "edge":
            history_query = history_query.order_by(
                MatchPrediction.edge.desc(), MatchPrediction.id.desc()
            )
        elif sort == "odd":
            history_query = history_query.order_by(
                MatchPrediction.odd.desc(), MatchPrediction.id.desc()
            )
        else:
            history_query = history_query.order_by(
                MatchPrediction.created_at.desc(), MatchPrediction.id.desc()
            )
        items = history_query.offset((page - 1) * page_size).limit(page_size).all()
        return items, total

    def get_all_labeled(self) -> List[MatchPrediction]:
        """Fetch all records that have a verified actual_result (used for ML training)."""
        return (
            self.db.query(MatchPrediction)
            .filter(
                MatchPrediction.actual_result.isnot(None),
                MatchPrediction.training_eligible.is_(True),
                MatchPrediction.result_verification_status == "verified",
            )
            .order_by(MatchPrediction.id.asc())
            .all()
        )

    def get_unlabeled(self, limit: int = 500) -> List[MatchPrediction]:
        """Fetch recent predictions that still need a verified match result."""
        return (
            self.db.query(MatchPrediction)
            .filter(
                MatchPrediction.actual_result.is_(None),
                MatchPrediction.training_eligible.is_(True),
            )
            .order_by(MatchPrediction.id.desc())
            .limit(limit)
            .all()
        )

    def count_labeled(self) -> int:
        return (
            self.db.query(MatchPrediction)
            .filter(
                MatchPrediction.actual_result.isnot(None),
                MatchPrediction.training_eligible.is_(True),
                MatchPrediction.result_verification_status == "verified",
            )
            .count()
        )

    def get_all_auditable(
        self,
        limit: int = 5000,
        *,
        model_artifact_version: str | None = None,
    ) -> List[MatchPrediction]:
        """Return only forecasts admitted by the production eligibility policy."""
        query = (
            self.db.query(MatchPrediction)
            .filter(
                MatchPrediction.training_eligible.is_(True),
                or_(
                    MatchPrediction.actual_result.is_(None),
                    MatchPrediction.result_verification_status == "verified",
                ),
            )
        )
        if model_artifact_version is not None:
            query = query.filter(
                MatchPrediction.model_artifact_version == model_artifact_version
            )
        return query.order_by(MatchPrediction.id.asc()).limit(limit).all()

    def update_result(
        self,
        record_id: int,
        actual_result: str,
        actual_score_home: Optional[int] = None,
        actual_score_away: Optional[int] = None,
        roi: Optional[float] = None,
        clv: Optional[float] = None,
        closing_odds: Optional[float] = None,
        closing_odds_snapshot_at: Optional[datetime] = None,
        closing_odds_snapshot_id: Optional[int] = None,
        verification_status: str = "manual",
        result_source: Optional[str] = None,
        result_provider_fixture_id: Optional[str] = None,
        verification_note: Optional[str] = None,
    ) -> Optional[MatchPrediction]:
        record = self.get_by_id(record_id)
        if not record:
            return None

        if actual_result not in {"HOME_WIN", "DRAW", "AWAY_WIN"}:
            raise ValueError("actual_result must be a valid 1X2 outcome")
        if verification_status not in {"verified", "manual", "conflict", "rejected"}:
            raise ValueError("invalid result verification status")
        if (actual_score_home is None) != (actual_score_away is None):
            raise ValueError("both score values must be supplied together")
        if actual_score_home is not None and actual_score_away is not None:
            if actual_score_home < 0 or actual_score_away < 0:
                raise ValueError("score values cannot be negative")
            score_result = (
                "HOME_WIN"
                if actual_score_home > actual_score_away
                else "AWAY_WIN" if actual_score_home < actual_score_away else "DRAW"
            )
            if score_result != actual_result:
                raise ValueError("actual_result conflicts with the supplied score")

        if (
            record.result_verification_status == "verified"
            and record.actual_result is not None
            and record.actual_result != actual_result
        ):
            record.result_verification_status = "conflict"
            record.training_eligible = False
            record.result_verification_note = (
                verification_note or "A later result conflicts with the verified label"
            )[:255]
            self._commit()
            self.db.refresh(record)
            return record

        record.actual_result = actual_result
        if actual_score_home is not None:
            record.actual_score_home = actual_score_home
        if actual_score_away is not None:
            record.actual_score_away = actual_score_away
        if roi is not None:
            record.roi = roi
        if clv is not None:
            record.clv = clv
        if closing_odds is not None:
            record.closing_odds = closing_odds
        if closing_odds_snapshot_at is not None:
            record.closing_odds_snapshot_at = closing_odds_snapshot_at
        if closing_odds_snapshot_id is not None:
            record.closing_odds_snapshot_id = closing_odds_snapshot_id
        record.result_verification_status = verification_status
        record.result_source = result_source[:50] if result_source else None
        record.result_provider_fixture_id = (
            result_provider_fixture_id[:100] if result_provider_fixture_id else None
        )
        record.result_verified_at = (
            datetime.now(timezone.utc) if verification_status == "verified" else None
        )
        record.result_verification_note = (
            verification_note[:255] if verification_note else None
        )

        self._commit()
        self.db.refresh(record)
        return record

    def mark_result_verification(
        self,
        record_id: int,
        *,
        status: Literal["pending", "conflict", "rejected"],
        note: str,
    ) -> Optional[MatchPrediction]:
        record = self.get_by_id(record_id)
        if record is None:
            return None
        record.result_verification_status = status
        record.result_verification_note = note[:255]
        if status in {"conflict", "rejected"}:
            record.training_eligible = False
        self._commit()
        self.db.refresh(record)
        return record
