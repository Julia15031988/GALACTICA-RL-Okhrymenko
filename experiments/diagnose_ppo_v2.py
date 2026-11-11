from pathlib import Path
import math

import numpy as np
from stable_baselines3 import PPO

from galactica.environment.space_env_v2 import SpaceEnvV2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v2_empty_100k.zip"


def main():
    print("=" * 105)
    print("GALACTICA-RL V2.1: PPO FLIGHT DIAGNOSTICS")
    print("=" * 105)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Load trained PPO model
    model = PPO.load(str(MODEL_PATH), device="cpu")

    # Empty space, same scenario as training
    env = SpaceEnvV2(asteroid_count=0)

    observation, _ = env.reset(seed=42)

    print("\nMODEL:", MODEL_PATH.name)
    print("OBSERVATION:", observation.shape)
    print("ACTION SPACE:", env.action_space.shape)

    assert observation.shape == (81,)
    assert env.action_space.shape == (3,)

    print("\nSTART POSITION: (10, 50)")
    print("GOAL POSITION: (90, 50)")

    print("\n" + "-" * 105)

    print(
        f"{'STEP':<6}"
        f"{'X':>8}"
        f"{'Y':>8}"
        f"{'SPEED':>10}"
        f"{'STEER':>10}"
        f"{'THRUST':>10}"
        f"{'LATERAL':>10}"
        f"{'FUEL':>10}"
        f"{'DISTANCE':>12}"
        f"{'REWARD':>11}"
    )

    print("-" * 105)

    terminated = False
    truncated = False

    step_count = 0
    total_reward = 0.0

    max_speed = 0.0
    reverse_steps = 0
    lateral_steps = 0
    neutral_steps = 0

    info = {}

    while not (terminated or truncated):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        steering = float(action[0])
        longitudinal = float(action[1])
        lateral = float(action[2])

        if longitudinal < -0.05:
            reverse_steps += 1

        if abs(lateral) > 0.05:
            lateral_steps += 1

        if (
            abs(longitudinal) <= 0.05
            and abs(lateral) <= 0.05
        ):
            neutral_steps += 1

        observation, reward, terminated, truncated, info = (
            env.step(
                np.asarray(action, dtype=np.float32)
            )
        )

        ship = env.spaceship

        speed = math.hypot(ship.vx, ship.vy)

        distance = math.hypot(
            env.goal_x - ship.x,
            env.goal_y - ship.y,
        )

        max_speed = max(max_speed, speed)
        total_reward += float(reward)
        step_count += 1

        print(
            f"{step_count:<6}"
            f"{ship.x:>8.2f}"
            f"{ship.y:>8.2f}"
            f"{speed:>10.2f}"
            f"{steering:>10.2f}"
            f"{longitudinal:>10.2f}"
            f"{lateral:>10.2f}"
            f"{ship.fuel:>10.2f}"
            f"{distance:>12.2f}"
            f"{reward:>11.3f}"
        )

    final_speed = math.hypot(
        env.spaceship.vx,
        env.spaceship.vy,
    )

    print("-" * 105)

    print("\nFLIGHT SUMMARY")
    print("=" * 55)

    print("Goal reached:", info.get("goal_reached", False))
    print("Collision:", info.get("collision", False))
    print("Out of bounds:", info.get("out_of_bounds", False))

    print("Total steps:", step_count)
    print("Total reward:", round(total_reward, 3))

    print("Maximum speed:", round(max_speed, 3))
    print("Final speed:", round(final_speed, 3))

    print("Reverse thrust steps:", reverse_steps)
    print("Lateral thrust steps:", lateral_steps)
    print("Neutral thrust steps:", neutral_steps)

    print(
        "Fuel consumed:",
        round(100.0 - env.spaceship.fuel, 3),
    )

    print(
        "Final position:",
        (
            round(env.spaceship.x, 3),
            round(env.spaceship.y, 3),
        ),
    )

    print("\nDIAGNOSTIC NOTES")

    if info.get("goal_reached", False):
        print("SUCCESS: Spaceship reached the goal.")
    else:
        print("FAILURE: Spaceship did not reach the goal.")

    if reverse_steps > 0:
        print("Reverse thrust was used during the flight.")
    else:
        print("Reverse thrust was not used.")

    if final_speed > 2.0:
        print(
            "WARNING: Spaceship reached the end "
            "with significant remaining speed."
        )
    else:
        print(
            "Final speed is relatively low "
            "(diagnostic threshold: 2.0)."
        )

    env.close()

    print("\nDIAGNOSTICS COMPLETED")


if __name__ == "__main__":
    main()
