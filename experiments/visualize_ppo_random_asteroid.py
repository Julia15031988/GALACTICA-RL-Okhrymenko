import math

import matplotlib.pyplot as plt
import numpy as np

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.asteroid import AsteroidState
from galactica.environment.random_one_asteroid_env import (
    RandomOneAsteroidEnv,
)


MODEL_PATH = "models/ppo_random_asteroid_100k"

SCENARIOS = [
    (50.0, 50.0),
    (50.0, 60.0),
    (50.0, 40.0),
]


def run_scenario(agent, asteroid_x, asteroid_y):
    env = RandomOneAsteroidEnv()
    observation, _ = env.reset(seed=42)

    # Set the exact position for this evaluation.
    env.asteroids = [
        AsteroidState(
            x=asteroid_x,
            y=asteroid_y,
            radius=5.0,
            vx=0.0,
            vy=0.0,
        )
    ]

    # Refresh radar readings after moving the asteroid.
    observation = env._get_observation()

    asteroid = env.asteroids[0]

    positions = [(env.spaceship.x, env.spaceship.y)]
    speeds = [math.hypot(env.spaceship.vx, env.spaceship.vy)]
    clearances = []
    rewards = []

    def clearance():
        return (
            math.hypot(
                env.spaceship.x - asteroid.x,
                env.spaceship.y - asteroid.y,
            )
            - env.spaceship.radius
            - asteroid.radius
        )

    clearances.append(clearance())

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        # Our PPOAgent.predict() returns only the action.
        action = agent.predict(observation)

        observation, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32)
        )

        positions.append((env.spaceship.x, env.spaceship.y))
        speeds.append(math.hypot(env.spaceship.vx, env.spaceship.vy))
        clearances.append(clearance())
        rewards.append(float(reward))

    result = {
        "position": (asteroid_x, asteroid_y),
        "trajectory": np.asarray(positions),
        "speeds": speeds,
        "clearances": clearances,
        "rewards": rewards,
        "steps": len(rewards),
        "total_reward": sum(rewards),
        "goal": bool(info.get("goal_reached", False)),
        "collision": bool(info.get("collision", False)),
        "out_of_bounds": bool(info.get("out_of_bounds", False)),
        "timeout": bool(truncated),
    }

    env.close()
    return result


def plot_results(results):
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))

    for col, result in enumerate(results):
        asteroid_x, asteroid_y = result["position"]
        trajectory = result["trajectory"]
        steps = range(len(result["speeds"]))

        if result["goal"]:
            status = "SUCCESS"
        elif result["collision"]:
            status = "COLLISION"
        elif result["out_of_bounds"]:
            status = "OUT OF BOUNDS"
        else:
            status = "TIMEOUT"

        # Row 1: trajectories
        ax = axes[0, col]

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            marker="o",
            markersize=3,
            linewidth=2,
            label="PPO trajectory",
        )

        ax.add_patch(
            plt.Circle(
                (asteroid_x, asteroid_y),
                radius=8.0,
                color="red",
                alpha=0.25,
                label="Collision zone",
            )
        )

        ax.add_patch(
            plt.Circle(
                (asteroid_x, asteroid_y),
                radius=5.0,
                color="gray",
                alpha=0.8,
                label="Asteroid",
            )
        )

        ax.scatter(
            10, 50,
            color="blue",
            s=70,
            label="Start",
            zorder=5,
        )

        ax.scatter(
            90, 50,
            color="green",
            marker="*",
            s=180,
            label="Goal",
            zorder=5,
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_title(
            f"Asteroid ({asteroid_x:.0f}, {asteroid_y:.0f})\n"
            f"{status} | {result['steps']} steps"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(fontsize=7)

        # Row 2: speed
        ax = axes[1, col]
        ax.plot(steps, result["speeds"])
        ax.set_title("Spaceship speed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Speed")
        ax.grid(True)

        # Row 3: clearance
        ax = axes[2, col]
        ax.plot(steps, result["clearances"])
        ax.axhline(
            0,
            color="red",
            linestyle="--",
            label="Collision threshold",
        )
        ax.set_title("Distance between surfaces")
        ax.set_xlabel("Step")
        ax.set_ylabel("Clearance")
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.suptitle(
        "GALACTICA-RL: PPO with Randomized Asteroid Training",
        fontsize=16,
    )
    plt.tight_layout()
    plt.show()


def main():
    agent = PPOAgent()
    agent.load(MODEL_PATH)

    results = []

    print("\nGALACTICA-RL: RANDOM ASTEROID PPO EVALUATION")
    print("=" * 60)

    for asteroid_x, asteroid_y in SCENARIOS:
        result = run_scenario(agent, asteroid_x, asteroid_y)
        results.append(result)

        print(f"\nAsteroid: {result['position']}")
        print("Goal reached:", result["goal"])
        print("Collision:", result["collision"])
        print("Out of bounds:", result["out_of_bounds"])
        print("Timeout:", result["timeout"])
        print("Steps:", result["steps"])
        print("Total reward:", round(result["total_reward"], 3))
        print(
            "Minimum clearance:",
            round(min(result["clearances"]), 3),
        )

    plot_results(results)


if __name__ == "__main__":
    main()
