import math

import numpy as np

from galactica.environment.asteroid import AsteroidState
from galactica.environment.space_env_v2 import SpaceEnvV2


def main():
    print("GALACTICA-RL V2: SAFETY TESTS")
    print("=" * 55)

    # TEST 1: High-speed asteroid collision
    env = SpaceEnvV2(asteroid_count=0)
    env.reset(seed=42)

    env.asteroids = [
        AsteroidState(
            x=50.0,
            y=50.0,
            radius=5.0,
            vx=0.0,
            vy=0.0,
        )
    ]

    # Ship starts far from asteroid, but crosses it in one step.
    env.spaceship.x = 30.0
    env.spaceship.y = 50.0
    env.spaceship.vx = 30.0
    env.spaceship.vy = 0.0
    env.spaceship.angle = 0.0

    # No engine thrust.
    action = np.array(
        [0.0, -1.0, -1.0, 0.0],
        dtype=np.float32,
    )

    _, reward, terminated, truncated, info = env.step(action)

    print("\nTEST 1: HIGH-SPEED COLLISION")
    print("Position:", env.spaceship.x, env.spaceship.y)
    print("Collision:", info["collision"])
    print("Reward:", reward)

    assert info["collision"] is True
    assert terminated is True
    assert reward == -1.0
    print("PASSED")

    # TEST 2: Reverse braking inside the environment
    env.reset(seed=42)

    env.spaceship.vx = 5.0
    env.spaceship.vy = 0.0
    env.spaceship.angle = 0.0

    reverse_action = np.array(
        [0.0, -1.0, 1.0, 0.0],
        dtype=np.float32,
    )

    print("\nTEST 2: REVERSE BRAKING")

    for step in range(5):
        _, reward, terminated, truncated, info = env.step(
            reverse_action
        )

        speed = math.hypot(
            env.spaceship.vx,
            env.spaceship.vy,
        )

        print(
            f"Step {step + 1}: "
            f"speed={speed:.2f}, "
            f"fuel={env.spaceship.fuel:.2f}"
        )

        assert not terminated
        assert not truncated

    assert math.isclose(env.spaceship.vx, 0.0, abs_tol=1e-9)
    print("PASSED")

    env.close()

    print("\nALL V2 SAFETY TESTS PASSED!")


if __name__ == "__main__":
    main()
