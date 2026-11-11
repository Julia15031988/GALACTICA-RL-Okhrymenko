import math

import matplotlib.pyplot as plt
import numpy as np

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.one_asteroid_env import OneAsteroidEnv


def main():
    env = OneAsteroidEnv()

    agent = PPOAgent()
    agent.load("models/ppo_one_asteroid_100k")

    obs, _ = env.reset(seed=42)

    positions = [(env.spaceship.x, env.spaceship.y)]
    speeds = [math.hypot(env.spaceship.vx, env.spaceship.vy)]
    distances = []
    rewards = []

    asteroid = env.asteroids[0]

    def clearance():
        return (
            math.hypot(
                env.spaceship.x - asteroid.x,
                env.spaceship.y - asteroid.y,
            )
            - env.spaceship.radius
            - asteroid.radius
        )

    distances.append(clearance())

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action = agent.predict(obs)

        obs, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32)
        )

        positions.append((env.spaceship.x, env.spaceship.y))
        speeds.append(
            math.hypot(env.spaceship.vx, env.spaceship.vy)
        )
        distances.append(clearance())
        rewards.append(reward)

    positions = np.array(positions)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Trajectory
    ax = axes[0, 0]
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        marker="o",
        markersize=3,
        label="Spaceship",
    )

    ax.add_patch(
        plt.Circle(
            (asteroid.x, asteroid.y),
            asteroid.radius + env.spaceship.radius,
            color="red",
            alpha=0.25,
            label="Collision zone",
        )
    )

    ax.scatter(10, 50, color="blue", label="Start")
    ax.scatter(
        env.goal_x,
        env.goal_y,
        color="green",
        marker="*",
        s=180,
        label="Goal",
    )

    ax.set_xlim(0, env.field_width)
    ax.set_ylim(0, env.field_height)
    ax.set_aspect("equal")
    ax.set_title("Spaceship trajectory")
    ax.legend()
    ax.grid(True)

    # 2. Speed
    axes[0, 1].plot(speeds)
    axes[0, 1].set_title("Speed over time")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Speed")
    axes[0, 1].grid(True)

    # 3. Asteroid clearance
    axes[1, 0].plot(distances)
    axes[1, 0].axhline(
        0, color="red", linestyle="--", label="Collision limit"
    )
    axes[1, 0].set_title("Clearance from asteroid")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Distance between surfaces")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Reward
    axes[1, 1].plot(
        range(1, len(rewards) + 1),
        rewards,
    )
    axes[1, 1].set_title("Reward per step")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    print("\nRESULTS")
    print("Steps:", len(rewards))
    print("Total reward:", round(sum(rewards), 3))
    print("Goal reached:", info.get("goal_reached"))
    print("Collision:", info.get("collision"))
    print("Out of bounds:", info.get("out_of_bounds"))
    print("Timeout:", truncated)
    print("Minimum clearance:", round(min(distances), 3))

    env.close()


if __name__ == "__main__":
    main()
