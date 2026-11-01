from galactica.environment.space_env import SpaceEnv
import pytest
import numpy as np


def test_reset_generates_asteroids():
    env = SpaceEnv()

    observation, info = env.reset(seed=42)

    assert observation.shape == (25,)
    assert len(env.asteroids) == 20
    assert isinstance(info, dict)

def test_reset_reproducibility():
    env = SpaceEnv()

    observation1, _ = env.reset(seed=42)
    asteroids1 = list(env.asteroids)

    observation2, _ = env.reset(seed=42)
    asteroids2 = list(env.asteroids)

    assert (observation1 == observation2).all()
    assert asteroids1 == asteroids2

def test_goal_reached():
    env = SpaceEnv()
    env.reset(seed=42)

    env.asteroids = []

    env.spaceship.x = env.goal_x
    env.spaceship.y = env.goal_y

    observation, reward, terminated, truncated, info = env.step(
        [0.0, -1.0]
    )

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["goal_reached"] is True

def test_episode_truncated_after_max_steps():
    env = SpaceEnv()
    env.reset(seed=42)

    env.asteroids = []
    env.max_steps = 3

    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(
            [0.0, -1.0]
        )

    assert terminated is False
    assert truncated is True
    assert env.current_step == 3

def test_spaceship_and_goal_must_not_overlap():
    env = SpaceEnv()
    env.reset(seed=42)

    env.goal_x = env.spaceship.x
    env.goal_y = env.spaceship.y

    with pytest.raises(
        ValueError,
        match="Spaceship and goal safety zones must not overlap",
    ):
        env._validate_start_position()


def test_spaceship_must_be_inside_field():
    env = SpaceEnv()
    env.reset(seed=42)

    # Переміщуємо корабель за допустиму межу
    env.spaceship.x = 99.0

    with pytest.raises(
        ValueError,
        match="Spaceship must be fully inside the field",
    ):
        env._validate_field_boundaries()


def test_goal_must_be_inside_field():
    env = SpaceEnv()
    env.reset(seed=42)

    # Переміщуємо ціль за допустиму межу
    env.goal_x = 98.0

    with pytest.raises(
        ValueError,
        match="Goal must be fully inside the field",
    ):
        env._validate_field_boundaries()

def test_spaceship_out_of_bounds_terminates_episode():
    env = SpaceEnv()
    env.reset(seed=42)

    # Remove asteroids to isolate boundary behavior.
    env.asteroids = []

    # Place spaceship near the right boundary.
    env.spaceship.x = 96.5
    env.spaceship.y = 50.0

    # Give spaceship velocity toward the right wall.
    env.spaceship.vx = 5.0
    env.spaceship.vy = 0.0

    # No steering and no thrust.
    action = np.array([0.0, -1.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    assert info["out_of_bounds"] is True
    assert info["collision"] is False
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert observation.shape == (25,)


def test_out_of_bounds_penalty_has_priority_over_goal():
    env = SpaceEnv()
    env.reset(seed=42)

    env.asteroids = []

    # Artificial state to test reward priority.
    env.goal_x = 95.0
    env.goal_y = 50.0

    # Spaceship body crosses the right boundary:
    # x + radius = 98 + 3 = 101.
    env.spaceship.x = 98.0
    env.spaceship.y = 50.0

    # No movement.
    env.spaceship.vx = 0.0
    env.spaceship.vy = 0.0

    action = np.array([0.0, -1.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    assert info["out_of_bounds"] is True
    assert info["goal_reached"] is True
    assert reward == -1.0
    assert terminated is True
    assert truncated is False

