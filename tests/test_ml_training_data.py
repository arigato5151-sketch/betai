from datetime import UTC, datetime, timedelta

from app.db.models import HistoricalFixture
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.training_data import HistoricalTrainingDataBuilder


def fixture(
    fixture_id: int,
    kickoff: datetime,
    home_team_id: int,
    away_team_id: int,
    home_goals: int,
    away_goals: int,
) -> HistoricalFixture:
    result = (
        "HOME_WIN"
        if home_goals > away_goals
        else "AWAY_WIN" if away_goals > home_goals else "DRAW"
    )
    return HistoricalFixture(
        fixture_id=fixture_id,
        league_id=203,
        season=2025,
        kickoff=kickoff,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_team=f"Team {home_team_id}",
        away_team=f"Team {away_team_id}",
        home_goals=home_goals,
        away_goals=away_goals,
        home_starting_xi=list(range(home_team_id * 100, home_team_id * 100 + 11)),
        away_starting_xi=list(range(away_team_id * 100, away_team_id * 100 + 11)),
        actual_result=result,
        status="FT",
    )


def test_historical_training_features_use_only_prior_matches() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=7), 1, 3, 9, 8),
        fixture(4, start + timedelta(days=14), 1, 3, 0, 5),
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    assert [row.fixture_id for row in rows] == [3, 4]
    first_snapshot = rows[0].feature_snapshot
    assert list(first_snapshot) == FeatureEngine.FEATURE_NAMES
    assert first_snapshot["home_gf_last5"] == 2.0
    assert first_snapshot["away_gf_last5"] == 1.0
    assert first_snapshot["home_scoring_streak"] == 1.0
    assert first_snapshot["away_scoring_streak"] == 1.0
    assert first_snapshot["odds_movement_home"] == 0.0
    assert first_snapshot["odds_movement_draw"] == 0.0
    assert first_snapshot["odds_movement_away"] == 0.0
    assert first_snapshot["league_203"] == 1.0
    assert (
        sum(first_snapshot[name] for name in FeatureEngine.LEAGUE_FEATURE_NAMES) == 1.0
    )
    assert rows[0].feature_snapshot_at == fixtures[2].kickoff
    # The 9-8 outcome is visible only to the following sample.
    assert rows[1].feature_snapshot["home_gf_last5"] == 5.5


def test_historical_training_requires_prior_history_for_both_teams() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 1, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=7), 1, 3, 1, 1),
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=2).build(fixtures)

    assert rows == []


def test_simultaneous_fixtures_do_not_leak_results_between_snapshots() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    shared_kickoff = start + timedelta(days=7)
    fixtures = [
        fixture(1, start, 1, 4, 1, 0),
        fixture(2, start, 3, 5, 2, 0),
        fixture(3, shared_kickoff, 1, 2, 9, 0),
        fixture(4, shared_kickoff, 1, 3, 0, 1),
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    assert [row.fixture_id for row in rows] == [4]
    assert rows[0].feature_snapshot["home_gf_last5"] == 1.0
    assert rows[0].training_source == "historical_fixture:api_football"
