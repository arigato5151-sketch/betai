from sqlalchemy import BigInteger

from app.db.models import MatchPrediction


def test_prediction_external_identifiers_are_64_bit() -> None:
    for column_name in ("fixture_id", "home_team_id", "away_team_id"):
        assert isinstance(
            MatchPrediction.__table__.columns[column_name].type,
            BigInteger,
        )
