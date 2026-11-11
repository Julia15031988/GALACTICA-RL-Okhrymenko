
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from galactica.environment.space_env_v2 import SpaceEnvV2


def main():
    model_path = Path("models/ppo_v2_empty_100k.zip")
    output_path = Path("results/ppo_v21_flight_diagnostics.png")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = PPO.load(str(model_path))
    env = SpaceEnvV2(asteroid_count=0)

    observation, _ = env.reset(seed=100)

    positions_x = [float(env.spaceship.x)]
    positions_y = [float(env.spaceship.y)]
    speeds = [float(np.hypot(env.spaceship.vx, env.spaceship.vy))]
    distances = [
        float(
            np.hypot(
                env.goal_x - env.spaceship.x,
                env.goal_y - env.spaceship.y,
            )
        )
    ]
    fuel = [float(env.spaceship.fuel)]

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        observation, reward, terminated, truncated, info = env.step(action)

        ship = env.spaceship

        positions_x.append(float(ship.x))
        positions_y.append(float(ship.y))
        speeds.append(float(np.hypot(ship.vx, ship.vy)))
        distances.append(
            float(np.hypot(env.goal_x - ship.x, env.goal_y - ship.y))
        )
        fuel.append(float(ship.fuel))

    steps = np.arange(len(speeds))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(positions_x, positions_y, linewidth=2, label="Spacecraft")
    ax.scatter(positions_x[0], positions_y[0], s=80, label="Start")
    ax.scatter(positions_x[-1], positions_y[-1], s=80, label="Final position")

    target = plt.Circle(
        (env.goal_x, env.goal_y),
        env.goal_radius,
        fill=False,
        linestyle="--",
        linewidth=2,
        color="green",
        label="Target zone",
    )
    ax.add_patch(target)

    ax.set_xlim(0, env.field_width)
    ax.set_ylim(0, env.field_height)
    ax.set_aspect("equal")
    ax.set_title("Spacecraft Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(alpha=0.25)

    # 2. Speed
    ax = axes[0, 1]
    ax.plot(steps, speeds, linewidth=2)
    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        label="Safe docking speed (V2.2 criterion)",
    )
    ax.set_title("Spacecraft Speed")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Speed")
    ax.legend()
    ax.grid(alpha=0.25)

    # 3. Distance to target
    ax = axes[1, 0]
    ax.plot(steps, distances, linewidth=2)
    ax.axhline(
        y=env.goal_radius,
        color="green",
        linestyle="--",
        label="Target radius",
    )
    ax.set_title("Distance to Target")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4. Fuel
    ax = axes[1, 1]
    ax.plot(steps, fuel, linewidth=2)
    ax.set_title("Remaining Fuel")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Fuel")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "GALACTICA-RL | PPO V2.1 Flight Diagnostics",
        fontsize=17,
        fontweight="bold",
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    print("\nGALACTICA-RL | PPO V2.1 RESULTS")
    print("-" * 45)
    print(f"Steps: {len(steps) - 1}")
    print(f"Final speed: {speeds[-1]:.3f}")
    print(f"Final distance: {distances[-1]:.3f}")
    print(f"Remaining fuel: {fuel[-1]:.3f}")
    print(f"Goal reached: {info.get('goal_reached')}")
    print(f"Graph saved: {output_path.resolve()}")

    env.close()
    plt.show()


if __name__ == "__main__":
    main()
