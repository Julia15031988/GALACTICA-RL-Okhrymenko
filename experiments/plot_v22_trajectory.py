from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from galactica.environment.space_env_v2 import SpaceEnvV2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v22_docking_100k.zip"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading GALACTICA-RL V2.2 model...")

    model = PPO.load(str(MODEL_PATH))

    env = SpaceEnvV2(asteroid_count=0)

    observation, _ = env.reset(seed=42)

    positions_x = [env.spaceship.x]
    positions_y = [env.spaceship.y]

    speeds = [
        np.hypot(env.spaceship.vx, env.spaceship.vy)
    ]

    distances = [
        np.hypot(
            env.goal_x - env.spaceship.x,
            env.goal_y - env.spaceship.y,
        )
    ]

    fuel_levels = [env.spaceship.fuel]

    terminated = False
    truncated = False
    step = 0
    info = {}

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        step += 1

        positions_x.append(env.spaceship.x)
        positions_y.append(env.spaceship.y)

        speeds.append(
            np.hypot(
                env.spaceship.vx,
                env.spaceship.vy,
            )
        )

        distances.append(
            np.hypot(
                env.goal_x - env.spaceship.x,
                env.goal_y - env.spaceship.y,
            )
        )

        fuel_levels.append(env.spaceship.fuel)

    steps = np.arange(len(positions_x))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    fig.suptitle(
        "GALACTICA-RL V2.2 | PPO Flight Diagnostics",
        fontsize=16,
        fontweight="bold",
    )

    # ---------------------------------
    # 1. Flight trajectory
    # ---------------------------------

    ax = axes[0, 0]

    ax.plot(
        positions_x,
        positions_y,
        linewidth=2,
        label="PPO trajectory",
    )

    ax.scatter(
        positions_x[0],
        positions_y[0],
        s=100,
        marker="^",
        label="Start",
        zorder=5,
    )

    ax.scatter(
        positions_x[-1],
        positions_y[-1],
        s=100,
        marker="x",
        label="Final position",
        zorder=6,
    )

    goal_circle = plt.Circle(
        (env.goal_x, env.goal_y),
        env.goal_radius,
        fill=False,
        linestyle="--",
        linewidth=2,
        label="Goal zone",
    )

    ax.add_patch(goal_circle)

    ax.scatter(
        env.goal_x,
        env.goal_y,
        marker="*",
        s=140,
        label="Goal center",
    )

    ax.set_xlim(0, env.field_width)
    ax.set_ylim(0, env.field_height)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Flight trajectory (X, Y)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---------------------------------
    # 2. Speed
    # ---------------------------------

    ax = axes[0, 1]

    ax.plot(
        steps,
        speeds,
        linewidth=2,
        label="Actual speed",
    )

    ax.axhline(
        y=env.goal_speed_threshold,
        linestyle="--",
        label="Docking speed limit",
    )

    ax.set_title("Speed during flight")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---------------------------------
    # 3. Distance to goal
    # ---------------------------------

    ax = axes[1, 0]

    ax.plot(
        steps,
        distances,
        linewidth=2,
        label="Distance to goal",
    )

    ax.axhline(
        y=env.goal_radius,
        linestyle="--",
        label="Goal radius",
    )

    ax.set_title("Distance to goal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---------------------------------
    # 4. Fuel
    # ---------------------------------

    ax = axes[1, 1]

    ax.plot(
        steps,
        fuel_levels,
        linewidth=2,
        label="Fuel remaining",
    )

    ax.set_title("Fuel consumption")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fuel")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_path = (
        RESULTS_DIR / "ppo_v22_flight_diagnostics.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    print("\nFLIGHT RESULTS")
    print("-" * 50)

    print(f"Steps: {step}")

    print(
        f"Final position: "
        f"({positions_x[-1]:.2f}, "
        f"{positions_y[-1]:.2f})"
    )

    print(f"Final speed: {speeds[-1]:.3f}")
    print(f"Minimum distance: {min(distances):.3f}")
    print(f"Final distance: {distances[-1]:.3f}")
    print(f"Fuel remaining: {fuel_levels[-1]:.2f}")

    print(f"Goal reached: {info.get('goal_reached')}")
    print(f"Collision: {info.get('collision')}")
    print(f"Out of bounds: {info.get('out_of_bounds')}")

    print(f"\nGraph saved: {output_path}")

    env.close()

    plt.show()


if __name__ == "__main__":
    main()
