from app.db.models import MatchPrediction
from app.core.config import settings
from app.prediction.ml.active_learning import ActiveLearningSelector


def prediction(
    record_id: int,
    probabilities: tuple[float | None, float | None, float | None],
    actual_result: str | None = None,
) -> MatchPrediction:
    return MatchPrediction(
        id=record_id,
        home_team=f"Home {record_id}",
        away_team=f"Away {record_id}",
        prob_home=probabilities[0],
        prob_draw=probabilities[1],
        prob_away=probabilities[2],
        actual_result=actual_result,
    )


def test_balanced_prediction_is_ranked_above_confident_prediction() -> None:
    balanced = prediction(1, (33.34, 33.33, 33.33))
    confident = prediction(2, (90.0, 5.0, 5.0))

    ranked = ActiveLearningSelector.rank([confident, balanced])

    assert [candidate["id"] for candidate in ranked] == [1, 2]
    assert ranked[0]["uncertainty_score"] > ranked[1]["uncertainty_score"]


def test_selector_excludes_labeled_and_incomplete_probability_rows() -> None:
    labeled = prediction(1, (33.34, 33.33, 33.33), actual_result="DRAW")
    incomplete = prediction(2, (50.0, None, 50.0))
    zero_sum = prediction(4, (0.0, 0.0, 0.0))
    eligible = prediction(3, (40.0, 30.0, 30.0))

    ranked = ActiveLearningSelector.rank([labeled, incomplete, zero_sum, eligible])

    assert [candidate["id"] for candidate in ranked] == [3]


def test_selector_applies_limit_and_deterministic_tie_break() -> None:
    rows = [prediction(record_id, (40.0, 30.0, 30.0)) for record_id in (3, 1, 2)]

    ranked = ActiveLearningSelector.rank(rows, limit=2)

    assert [candidate["id"] for candidate in ranked] == [1, 2]


def test_retraining_policy_respects_threshold_and_batch_size(monkeypatch) -> None:
    monkeypatch.setattr(settings, "MIN_TRAINING_SAMPLES", 200)
    monkeypatch.setattr(settings, "RETRAIN_EVERY_N_NEW", 25)

    assert ActiveLearningSelector.should_retrain(199, False) is False
    assert ActiveLearningSelector.should_retrain(200, False) is True
    assert ActiveLearningSelector.should_retrain(201, True) is False
    assert ActiveLearningSelector.should_retrain(225, True) is True
