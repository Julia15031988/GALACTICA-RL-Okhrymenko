import math

import numpy as np

from galactica.environment.space_env_v2 import SpaceEnvV2


def main():
    print("GALACTICA-RL V2.2: DOCKING TESTS")
    print("=" * 55)

    env = SpaceEnvV2(asteroid_count=0)
    neutral = np.zeros(3, dtype=np.float32)

    # TEST 1: Arriving slowly succeeds
    print("\nTEST 1: Slow arrival")

    env.reset(seed=42)
    env.spaceship.x = 89.0
    env.spaceship.y = 50.0
    env.spaceship.vx = 0.5

    _, reward, terminated, truncated, info = env.step(neutral)

    assert info["goal_reached"] is True
    assert terminated is True
    assert truncated is False
    assert info["speed"] <= 1.0

    print("Speed:", info["speed"])
    print("Reward:", reward)
    print("PASSED")

    # TEST 2: Arriving too fast is not success
    print("\nTEST 2: High-speed arrival")

    env.reset(seed=42)
    env.spaceship.x = 80.0
    env.spaceship.y = 50.0
    env.spaceship.vx = 6.0

    _, _, terminated, truncated, info = env.step(neutral)

    assert info["distance_to_goal"] <= env.goal_radius
    assert info["speed"] > env.goal_speed_threshold
    assert info["goal_reached"] is False
    assert terminated is False
    assert truncated is False

    print("Speed:", info["speed"])
    print("Distance:", info["distance_to_goal"])
    print("PASSED")

    # TEST 3: Reverse thrust slows the ship
    print("\nTEST 3: Braking")

    env.reset(seed=42)
    env.spaceship.x = 70.0
    env.spaceship.y = 50.0
    env.spaceship.vx = 3.0

    initial_speed = math.hypot(
        env.spaceship.vx,
        env.spaceship.vy,
    )

    _, _, _, _, info = env.step(
        np.array([0.0, -1.0, 0.0], dtype=np.float32)
    )

    assert info["speed"] < initial_speed
    assert math.isclose(env.spaceship.fuel, 99.0)

    print("Initial speed:", initial_speed)
    print("Final speed:", info["speed"])
    print("Fuel:", env.spaceship.fuel)
    print("PASSED")

    env.close()

    print("\n" + "=" * 55)
    print("ALL V2.2 DOCKING TESTS PASSED!")


if __name__ == "__main__":
    main()
