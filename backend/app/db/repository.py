from typing import List, Literal, Optional

from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.db.models import MatchPrediction


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

    def _commit(self) -> None:
        try:
            self.db.commit()
        except SQLAlchemyError:
            self.db.rollback()
            raise

    def upsert_prediction(self, data: dict) -> MatchPrediction:
        """
        Upsert a prediction.
        Uses high-performance PostgreSQL-native upsert in production.
        Falls back to standard transaction check-and-update on SQLite for local environments.
        """
        if self.db.bind is None:
            raise RuntimeError("Database session is not bound to an engine")
        dialect = self.db.bind.dialect.name

        if dialect == "sqlite":
            # Dialect-agnostic SQLite fallback
            fixture_id = data.get("fixture_id")
            existing = (
                self.get_by_fixture_id(fixture_id) if fixture_id is not None else None
            )
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
        update_cols = {
            col.name: getattr(stmt.excluded, col.name)
            for col in MatchPrediction.__table__.columns
            if col.name not in ["id", "created_at"]
        }

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
            .filter(MatchPrediction.actual_result.isnot(None))
            .order_by(MatchPrediction.id.asc())
            .all()
        )

    def get_unlabeled(self, limit: int = 500) -> List[MatchPrediction]:
        """Fetch recent predictions that still need a verified match result."""
        return (
            self.db.query(MatchPrediction)
            .filter(MatchPrediction.actual_result.is_(None))
            .order_by(MatchPrediction.id.desc())
            .limit(limit)
            .all()
        )

    def count_labeled(self) -> int:
        return (
            self.db.query(MatchPrediction)
            .filter(MatchPrediction.actual_result.isnot(None))
            .count()
        )

    def update_result(
        self,
        record_id: int,
        actual_result: str,
        actual_score_home: Optional[int] = None,
        actual_score_away: Optional[int] = None,
        roi: Optional[float] = None,
        clv: Optional[float] = None,
        closing_odds: Optional[float] = None,
    ) -> Optional[MatchPrediction]:
        record = self.get_by_id(record_id)
        if not record:
            return None

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

        self._commit()
        self.db.refresh(record)
        return record
