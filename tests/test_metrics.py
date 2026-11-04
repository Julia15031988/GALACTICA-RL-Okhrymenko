import pytest

from galactica.evaluation.metrics import (
    EpisodeResult,
    calculate_metrics,
)


def test_calculate_metrics():
    results = [
        EpisodeResult(
            seed=1,
            steps=10,
            total_reward=1.0,
            goal_reached=True,
            collision=False,
            out_of_bounds=False,
            fuel_used=5.0,
        ),
        EpisodeResult(
            seed=2,
            steps=20,
            total_reward=-1.0,
            goal_reached=False,
            collision=True,
            out_of_bounds=False,
            fuel_used=15.0,
        ),
    ]

    metrics = calculate_metrics(results)

    assert metrics["episodes"] == 2
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["collision_rate"] == pytest.approx(0.5)
    assert metrics["average_steps"] == pytest.approx(15.0)
    assert metrics["average_reward"] == pytest.approx(0.0)
    assert metrics["average_fuel_used"] == pytest.approx(10.0)
