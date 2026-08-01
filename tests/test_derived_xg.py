from datetime import UTC, datetime, timedelta

from app.db.models import HistoricalFixture
from app.services.derived_xg import MODEL_VERSION, DerivedXGService


def fixture(
    fixture_id: int,
    *,
    observed: bool,
    kickoff: datetime,
) -> HistoricalFixture:
    home_shots = 6 + fixture_id % 18
    away_shots = 5 + (fixture_id * 3) % 17
    home_on_target = min(home_shots, 2 + fixture_id % 8)
    away_on_target = min(away_shots, 1 + (fixture_id * 2) % 7)
    return HistoricalFixture(
        fixture_id=fixture_id,
        league_id=39 if observed else 203,
        season=2025,
        kickoff=kickoff,
        home_team_id=fixture_id * 2,
        away_team_id=fixture_id * 2 + 1,
        home_team=f"Home {fixture_id}",
        away_team=f"Away {fixture_id}",
        home_goals=1,
        away_goals=0,
        actual_result="HOME_WIN",
        status="FT",
        home_shots=home_shots,
        away_shots=away_shots,
        home_shots_on_target=home_on_target,
        away_shots_on_target=away_on_target,
        home_corners=fixture_id % 10,
        away_corners=(fixture_id * 2) % 10,
        home_yellow_cards=fixture_id % 5,
        away_yellow_cards=(fixture_id + 1) % 5,
        home_red_cards=0,
        away_red_cards=0,
        home_xg=(
            round(0.06 * home_shots + 0.18 * home_on_target, 6) if observed else None
        ),
        away_xg=(
            round(0.06 * away_shots + 0.18 * away_on_target, 6) if observed else None
        ),
        xg_source="understat" if observed else None,
        xg_confidence=0.95 if observed else None,
    )


def test_derived_xg_passes_quality_gate_and_only_fills_missing_rows() -> None:
    start = datetime(2025, 7, 1, tzinfo=UTC)
    observed = [
        fixture(index, observed=True, kickoff=start + timedelta(days=index))
        for index in range(1, 121)
    ]
    target = fixture(1000, observed=False, kickoff=start + timedelta(days=200))

    result = DerivedXGService(
        min_training_matches=50,
        max_holdout_mae=0.5,
        min_baseline_improvement=0.05,
        confidence=0.65,
    ).build_updates([*observed, target])

    assert result.status == "ready"
    assert result.training_matches == 120
    assert result.holdout_mae is not None and result.holdout_mae < 0.5
    assert result.baseline_mae is not None
    assert result.holdout_mae < result.baseline_mae
    assert len(result.updates) == 1
    update = result.updates[0]
    assert update["fixture_id"] == 1000
    assert update["xg_source"] == "derived_shot_model"
    assert update["xg_provider_match_id"] == MODEL_VERSION
    assert update["xg_confidence"] == 0.65
    assert 0.0 <= float(update["home_xg"]) <= 6.0
    assert 0.0 <= float(update["away_xg"]) <= 6.0


def test_derived_xg_does_not_train_below_minimum_sample_count() -> None:
    row = fixture(
        1,
        observed=True,
        kickoff=datetime(2025, 7, 1, tzinfo=UTC),
    )

    result = DerivedXGService(min_training_matches=2).build_updates([row])

    assert result.status == "insufficient_training_data"
    assert result.updates == ()


def test_derived_xg_rejects_inconsistent_shot_statistics() -> None:
    row = fixture(
        1,
        observed=False,
        kickoff=datetime(2025, 7, 1, tzinfo=UTC),
    )
    row.home_shots = 3
    row.home_shots_on_target = 4

    assert DerivedXGService._match_features(row) is None
