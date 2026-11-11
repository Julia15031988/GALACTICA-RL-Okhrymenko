import math

import numpy as np
from stable_baselines3.common.env_checker import check_env

from galactica.environment.space_env_v2 import SpaceEnvV2


def main():
    print("GALACTICA-RL V2.1: CONTROL TESTS")
    print("=" * 55)

    env = SpaceEnvV2(asteroid_count=0)

    # TEST 1: Gymnasium compatibility
    print("\nTEST 1: Gymnasium compatibility")

    check_env(env, warn=True)

    print("PASSED")

    # TEST 2: 72 radar rays and 3 control actions
    print("\nTEST 2: Observation and action spaces")

    observation, _ = env.reset(seed=42)

    print("Observation shape:", observation.shape)
    print("Action shape:", env.action_space.shape)
    print("Radar readings:", len(observation[7:79]))

    # 7 spaceship parameters + 72 radar rays + 2 goal values
    assert observation.shape == (81,)

    # Steering, longitudinal thrust, lateral thrust
    assert env.action_space.shape == (3,)

    # Radar readings occupy indices 7 through 78
    assert len(observation[7:79]) == 72

    assert env.observation_space.contains(observation)

    print("PASSED")

    # TEST 3: Neutral controls
    print("\nTEST 3: Neutral controls")

    env.reset(seed=42)

    # Give the spaceship an initial velocity.
    env.spaceship.vx = 2.0

    fuel_before = env.spaceship.fuel

    # No steering, no longitudinal thrust, no lateral thrust.
    env.step(
        np.array(
            [0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    print("Position X:", env.spaceship.x)
    print("Velocity X:", env.spaceship.vx)
    print("Fuel:", env.spaceship.fuel)

    # Spaceship continues moving because of inertia.
    assert math.isclose(
        env.spaceship.x,
        12.0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        env.spaceship.vx,
        2.0,
        abs_tol=1e-9,
    )

    # Engines are off, so fuel is not consumed.
    assert math.isclose(
        env.spaceship.fuel,
        fuel_before,
        abs_tol=1e-9,
    )

    print("PASSED")

    # TEST 4: Forward thrust
    print("\nTEST 4: Forward thrust")

    env.reset(seed=42)

    env.step(
        np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )
    )

    print("Velocity X:", env.spaceship.vx)
    print("Fuel:", env.spaceship.fuel)

    assert math.isclose(
        env.spaceship.vx,
        1.0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        env.spaceship.fuel,
        99.0,
        abs_tol=1e-9,
    )

    print("PASSED")

    # TEST 5: Reverse thrust
    print("\nTEST 5: Reverse thrust")

    env.reset(seed=42)

    env.spaceship.vx = 3.0

    env.step(
        np.array(
            [0.0, -1.0, 0.0],
            dtype=np.float32,
        )
    )

    print("Velocity X:", env.spaceship.vx)
    print("Fuel:", env.spaceship.fuel)

    assert math.isclose(
        env.spaceship.vx,
        2.0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        env.spaceship.fuel,
        99.0,
        abs_tol=1e-9,
    )

    print("PASSED")

    # TEST 6: Lateral thrust
    print("\nTEST 6: Lateral thrust")

    env.reset(seed=42)

    env.step(
        np.array(
            [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    )

    print("Velocity Y:", env.spaceship.vy)
    print("Fuel:", env.spaceship.fuel)

    assert math.isclose(
        env.spaceship.vy,
        0.6,
        abs_tol=1e-9,
    )

    assert math.isclose(
        env.spaceship.fuel,
        99.0,
        abs_tol=1e-9,
    )

    print("PASSED")

    # TEST 7: Random actions
    print("\nTEST 7: Random actions")

    observation, _ = env.reset(seed=42)

    steps = 0

    for _ in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert observation.shape == (81,)
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)

        steps += 1

        if terminated or truncated:
            break

    print("Steps:", steps)
    print("Last info:", info)

    print("PASSED")

    env.close()

    print("\n" + "=" * 55)
    print("ALL V2.1 CONTROL TESTS PASSED!")


if __name__ == "__main__":
    main()
