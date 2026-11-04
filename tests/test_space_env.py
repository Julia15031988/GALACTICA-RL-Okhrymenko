from dataclasses import replace

import numpy as np

from galactica.environment.asteroid import AsteroidState
from galactica.environment.space_env import SpaceEnv


def test_space_env_detects_tunneling():
    env = SpaceEnv()
    env.reset(seed=42)

    # Place one asteroid directly on the spaceship's path.
    env.asteroids = [
        AsteroidState(
            x=20.0,
            y=50.0,
            radius=2.0,
            vx=0.0,
            vy=0.0,
        )
    ]

    # Give the spaceship enough speed to cross
    # the asteroid in a single simulation step.
    env.spaceship = replace(
        env.spaceship,
        x=10.0,
        y=50.0,
        vx=20.0,
        vy=0.0,
    )

    action = np.array([0.0, -1.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    assert info["collision"] is True
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
