import pytest

from galactica.environment.space_env import SpaceEnv


def test_reward_for_progress_toward_goal():
    env = SpaceEnv()
    env.reset(seed=42)

    # Start: spaceship at (10, 50), goal at (90, 50).
    # Distance to goal = 80.
    previous_distance = 80.0

    # Moving closer should give a positive progress reward.
    current_distance = 79.0

    reward = env._calculate_progress_reward(
        previous_distance=previous_distance,
        current_distance=current_distance,
    )

    assert reward > 0.0


def test_reward_for_moving_away():
    env = SpaceEnv()
    env.reset(seed=42)

    previous_distance = 80.0
    current_distance = 81.0

    reward = env._calculate_progress_reward(
        previous_distance=previous_distance,
        current_distance=current_distance,
    )

    assert reward < 0.0


def test_reward_for_no_progress():
    env = SpaceEnv()
    env.reset(seed=42)

    reward = env._calculate_progress_reward(
        previous_distance=80.0,
        current_distance=80.0,
    )

    assert reward == pytest.approx(0.0)
