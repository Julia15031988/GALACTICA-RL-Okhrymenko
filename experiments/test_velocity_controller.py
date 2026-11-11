import math

import numpy as np

from galactica.environment.space_env_v2 import SpaceEnvV2
from galactica.environment.velocity_controller import VelocityController


def run_controller(env, controller, vx, vy, steps):
    """Apply controller commands through real spaceship physics."""

    for _ in range(steps):
        action = controller.calculate_action(
            spaceship=env.spaceship,
            desired_vx=vx,
            desired_vy=vy,
        )

        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    return env.spaceship


def main():
    print("=" * 55)
    print("GALACTICA-RL V3: VELOCITY CONTROLLER TESTS")
    print("=" * 55)

    controller = VelocityController(
        kp=0.5,
        max_desired_speed=5.0,
    )

    env = SpaceEnvV2(asteroid_count=0)

    # TEST 1: Forward acceleration
    print("\nTEST 1: Forward velocity control")

    env.reset(seed=42)

    ship = run_controller(
        env, controller, vx=3.0, vy=0.0, steps=10
    )

    assert math.isclose(ship.vx, 3.0, abs_tol=0.05)
    assert abs(ship.vy) < 0.05

    print(f"Velocity: ({ship.vx:.3f}, {ship.vy:.3f})")
    print("PASSED")

    # TEST 2: Braking
    print("\nTEST 2: Braking to zero")

    env.reset(seed=42)
    env.spaceship.vx = 4.0

    ship = run_controller(
        env, controller, vx=0.0, vy=0.0, steps=15
    )

    assert abs(ship.vx) < 0.05
    assert abs(ship.vy) < 0.05

    print(f"Velocity: ({ship.vx:.3f}, {ship.vy:.3f})")
    print("PASSED")

    # TEST 3: Lateral movement
    print("\nTEST 3: Lateral velocity control")

    env.reset(seed=42)

    ship = run_controller(
        env, controller, vx=0.0, vy=2.0, steps=15
    )

    assert abs(ship.vx) < 0.05
    assert math.isclose(ship.vy, 2.0, abs_tol=0.05)

    print(f"Velocity: ({ship.vx:.3f}, {ship.vy:.3f})")
    print("PASSED")

    # TEST 4: Speed limitation
    print("\nTEST 4: Maximum desired speed")

    env.reset(seed=42)

    action = controller.calculate_action(
        spaceship=env.spaceship,
        desired_vx=100.0,
        desired_vy=0.0,
    )

    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    print("Action:", action)
    print("PASSED")

    env.close()

    print("\n" + "=" * 55)
    print("ALL VELOCITY CONTROLLER TESTS PASSED!")


if __name__ == "__main__":
    main()
