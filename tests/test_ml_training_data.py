from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import settings
from app.db.models import (
    HistoricalFixture,
    HistoricalPlayerPerformance,
    TeamLocation,
)
from app.db.player_context_repository import haversine_distance_km
from app.prediction.ml.features import FeatureEngine
from app.prediction.ml.historical import HistoricalFeatureService
from app.prediction.ml.training_data import HistoricalTrainingDataBuilder
from app.prediction.player_impact import PlayerImpactCalculator


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


def performance(
    fixture_id: int,
    kickoff: datetime,
    team_id: int,
    player_id: int,
    rating: float,
) -> HistoricalPlayerPerformance:
    return HistoricalPlayerPerformance(
        fixture_id=fixture_id,
        league_id=203,
        kickoff=kickoff,
        team_id=team_id,
        player_id=player_id,
        started=True,
        minutes=90,
        rating=rating,
        position=None,
        goals=0,
        assists=0,
        source="api_football",
    )


def team_location(
    team_id: int,
    latitude: float,
    longitude: float,
) -> TeamLocation:
    return TeamLocation(
        data_source="api_football",
        team_id=team_id,
        name=f"Team {team_id}",
        latitude=latitude,
        longitude=longitude,
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
    assert first_snapshot["league_id"] == 203.0
    assert first_snapshot["home_team_id"] == 1.0
    assert first_snapshot["away_team_id"] == 3.0
    assert first_snapshot["home_team_strength_ratio"] == 1.0
    assert first_snapshot["away_team_strength_ratio"] == 1.0
    assert rows[0].home_team_id == 1
    assert rows[0].away_team_id == 3
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


def test_historical_training_uses_fixture_opening_and_closing_odds() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    target = fixtures[-1]
    target.opening_home_odd = 2.0
    target.opening_draw_odd = 3.0
    target.opening_away_odd = 4.0
    target.closing_home_odd = 1.8
    target.closing_draw_odd = 3.3
    target.closing_away_odd = 4.4

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    assert len(rows) == 1
    snapshot = rows[0].feature_snapshot
    assert snapshot["odds_movement_home"] == -10.0
    assert snapshot["odds_movement_draw"] == 10.0
    assert snapshot["odds_movement_away"] == 10.0


def test_historical_training_prefers_prior_observed_xg() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    fixtures[0].home_xg = 2.45
    fixtures[1].home_xg = 1.35

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    assert len(rows) == 1
    assert rows[0].feature_snapshot["home_xg"] == 2.45
    assert rows[0].feature_snapshot["away_xg"] == 1.35


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


def test_training_lineup_reference_uses_latest_cross_competition_fixture() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    cup_lineup = list(range(200, 211))
    fixtures = [
        fixture(1, start, 1, 2, 1, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=3), 1, 5, 2, 1),
        fixture(4, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    fixtures[2].league_id = 39
    fixtures[2].home_starting_xi = cup_lineup
    fixtures[3].home_starting_xi = cup_lineup

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    assert [row.fixture_id for row in rows] == [4]
    assert rows[0].feature_snapshot["home_lineup_continuity"] == 1.0


def test_training_lineup_reference_ignores_later_missing_lineup() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    confirmed_lineup = list(range(100, 111))
    fixtures = [
        fixture(1, start, 1, 2, 1, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=3), 1, 5, 2, 1),
        fixture(4, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    fixtures[0].home_starting_xi = confirmed_lineup
    fixtures[2].home_starting_xi = None
    fixtures[3].home_starting_xi = confirmed_lineup

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    assert [row.fixture_id for row in rows] == [4]
    assert rows[0].feature_snapshot["home_lineup_reference_available"] == 1.0
    assert rows[0].feature_snapshot["home_lineup_continuity"] == 1.0


def test_historical_training_uses_prior_player_ratings_and_excludes_same_kickoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "PLAYER_IMPACT_MIN_RATED_STARTERS", 7)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_LOOKBACK_MATCHES", 2)
    monkeypatch.setattr(settings, "PLAYER_IMPACT_RATING_DECAY", 0.5)
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=3), 1, 5, 1, 0),
        fixture(4, start + timedelta(days=4), 3, 6, 2, 0),
        fixture(5, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    fixtures[4].home_starting_xi = list(range(101, 112))
    performances = [
        *[
            performance(
                1,
                start,
                1,
                player_id,
                5.0 if player_id == 100 else 6.0,
            )
            for player_id in range(100, 111)
        ],
        performance(3, start + timedelta(days=3), 1, 100, 9.0),
        *[
            performance(2, start + timedelta(days=1), 3, player_id, 7.0)
            for player_id in range(300, 311)
        ],
        # The target fixture's rating must not enter its own pre-match snapshot.
        performance(5, start + timedelta(days=7), 1, 111, 10.0),
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(
        fixtures,
        player_performances=performances,
    )

    home_ratings = {player_id: 6.0 for player_id in range(100, 111)}
    home_ratings[100] = (9.0 + 5.0 * 0.5) / 1.5
    expected = PlayerImpactCalculator.assess(
        home_ratings,
        list(range(100, 111)),
        list(range(101, 112)),
        [100],
    )
    assert [row.fixture_id for row in rows] == [5]
    assert rows[0].feature_snapshot["home_team_strength_ratio"] == pytest.approx(
        expected.team_strength_ratio
    )
    assert rows[0].feature_snapshot["home_team_strength_ratio"] < 1.0
    assert rows[0].feature_snapshot["away_team_strength_ratio"] == 1.0


def test_historical_training_excludes_stale_player_ratings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "HISTORICAL_FORM_MAX_AGE_DAYS", 45)
    start = datetime(2025, 8, 1, tzinfo=UTC)
    target_kickoff = start + timedelta(days=60)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, target_kickoff, 1, 3, 1, 1),
    ]
    # If stale ratings leaked into the snapshot, this replacement would reduce
    # home strength. With the live-serving freshness policy it must stay neutral.
    fixtures[2].home_starting_xi = list(range(101, 112))
    performances = [
        *[performance(1, start, 1, player_id, 7.0) for player_id in range(100, 111)],
        *[
            performance(2, start + timedelta(days=1), 3, player_id, 7.0)
            for player_id in range(300, 311)
        ],
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(
        fixtures,
        player_performances=performances,
    )

    assert [row.fixture_id for row in rows] == [3]
    assert rows[0].feature_snapshot["home_team_strength_ratio"] == 1.0
    assert rows[0].feature_snapshot["away_team_strength_ratio"] == 1.0


def test_historical_training_derives_reference_xi_from_contribution_only_data() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    fixtures[0].home_starting_xi = None
    fixtures[2].home_starting_xi = list(range(101, 112))
    performances = [
        HistoricalPlayerPerformance(
            fixture_id=1,
            league_id=203,
            kickoff=start,
            team_id=1,
            player_id=player_id,
            started=True,
            minutes=90,
            rating=None,
            position=None,
            goals=2 if player_id == 100 else 1,
            assists=0,
            source="api_football",
        )
        for player_id in range(100, 111)
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(
        fixtures,
        player_performances=performances,
    )

    assert [row.fixture_id for row in rows] == [3]
    assert rows[0].feature_snapshot["home_team_strength_ratio"] < 1.0
    assert rows[0].feature_snapshot["away_team_strength_ratio"] == 1.0


def test_historical_training_combines_prior_schedule_with_away_travel() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    fixtures = [
        fixture(1, start, 1, 2, 2, 0),
        fixture(2, start + timedelta(days=1), 3, 4, 1, 0),
        fixture(3, start + timedelta(days=7), 1, 3, 1, 1),
    ]
    locations = [
        team_location(1, 41.0082, 28.9784),
        team_location(3, 51.5074, -0.1278),
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(
        fixtures,
        team_locations=locations,
    )

    distance = haversine_distance_km(
        locations[1].latitude,
        locations[1].longitude,
        locations[0].latitude,
        locations[0].longitude,
    )
    expected = FeatureEngine.compute_fatigue_index(
        HistoricalFeatureService._schedule_frame(
            [fixtures[0]],
            team_id=1,
            since=fixtures[2].kickoff - timedelta(days=14),
            before=fixtures[2].kickoff,
        ),
        HistoricalFeatureService._schedule_frame(
            [fixtures[1]],
            team_id=3,
            since=fixtures[2].kickoff - timedelta(days=14),
            before=fixtures[2].kickoff,
        ),
        fixtures[2].kickoff,
        away_travel_distance_km=distance,
    )
    assert rows[0].feature_snapshot["fatigue_index"] == expected
    assert rows[0].feature_snapshot["fatigue_index"] > 0.0


def test_simultaneous_fixture_is_excluded_from_schedule_fatigue() -> None:
    start = datetime(2025, 8, 1, tzinfo=UTC)
    shared_kickoff = start + timedelta(days=7)
    fixtures = [
        fixture(1, start, 1, 4, 1, 0),
        fixture(2, start, 3, 5, 2, 0),
        fixture(3, shared_kickoff, 1, 2, 9, 0),
        fixture(4, shared_kickoff, 1, 3, 0, 1),
    ]

    rows = HistoricalTrainingDataBuilder(minimum_team_history=1).build(fixtures)

    expected = FeatureEngine.compute_fatigue_index(
        HistoricalFeatureService._schedule_frame(
            [fixtures[0]],
            team_id=1,
            since=shared_kickoff - timedelta(days=14),
            before=shared_kickoff,
        ),
        HistoricalFeatureService._schedule_frame(
            [fixtures[1]],
            team_id=3,
            since=shared_kickoff - timedelta(days=14),
            before=shared_kickoff,
        ),
        shared_kickoff,
    )
    assert rows[0].feature_snapshot["fatigue_index"] == expected
