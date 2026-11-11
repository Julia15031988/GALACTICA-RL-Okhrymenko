import math

import matplotlib.pyplot as plt
import numpy as np

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.asteroid import AsteroidState
from galactica.environment.one_asteroid_env import OneAsteroidEnv


MODEL_PATH = "models/ppo_one_asteroid_100k"

# Same asteroid, different positions.
SCENARIOS = [
    (50.0, 50.0),
    (50.0, 60.0),
    (50.0, 40.0),
]

MAX_STEPS = 500


def run_scenario(agent, asteroid_x, asteroid_y):
    env = OneAsteroidEnv()

    observation, _ = env.reset(seed=42)

    # Change the asteroid position AFTER reset.
    # Otherwise reset() would restore the original position.
    env.asteroids = [
        AsteroidState(
            x=asteroid_x,
            y=asteroid_y,
            radius=5.0,
            vx=0.0,
            vy=0.0,
        )
    ]

    # Important: refresh radar observations.
    observation = env._get_observation()

    positions = [(env.spaceship.x, env.spaceship.y)]
    speeds = [math.hypot(env.spaceship.vx, env.spaceship.vy)]

    rewards = []
    clearances = []

    def get_clearance():
        asteroid = env.asteroids[0]

        return (
            math.hypot(
                env.spaceship.x - asteroid.x,
                env.spaceship.y - asteroid.y,
            )
            - env.spaceship.radius
            - asteroid.radius
        )

    clearances.append(get_clearance())

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        # Our PPOAgent.predict() returns the action directly.
        action = agent.predict(observation)

        observation, reward, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32)
        )

        positions.append((env.spaceship.x, env.spaceship.y))
        speeds.append(math.hypot(env.spaceship.vx, env.spaceship.vy))
        rewards.append(float(reward))
        clearances.append(get_clearance())

        if len(rewards) >= MAX_STEPS:
            break

    result = {
        "asteroid_position": (asteroid_x, asteroid_y),
        "positions": np.asarray(positions),
        "speeds": speeds,
        "rewards": rewards,
        "clearances": clearances,
        "steps": len(rewards),
        "total_reward": sum(rewards),
        "goal_reached": info.get("goal_reached", False),
        "collision": info.get("collision", False),
        "out_of_bounds": info.get("out_of_bounds", False),
        "timeout": truncated,
        "minimum_clearance": min(clearances),
    }

    env.close()
    return result


def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for ax, result in zip(axes, results):
        asteroid_x, asteroid_y = result["asteroid_position"]
        positions = result["positions"]

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            marker="o",
            markersize=3,
            linewidth=2,
            label="PPO trajectory",
        )

        # Collision zone = asteroid radius + spaceship radius.
        collision_zone = plt.Circle(
            (asteroid_x, asteroid_y),
            radius=8.0,
            color="red",
            alpha=0.25,
            label="Collision zone",
        )
        ax.add_patch(collision_zone)

        # Actual asteroid.
        asteroid = plt.Circle(
            (asteroid_x, asteroid_y),
            radius=5.0,
            color="gray",
            alpha=0.75,
            label="Asteroid",
        )
        ax.add_patch(asteroid)

        ax.scatter(
            10, 50,
            color="blue",
            s=65,
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

        if result["goal_reached"]:
            status = "SUCCESS"
        elif result["collision"]:
            status = "COLLISION"
        elif result["out_of_bounds"]:
            status = "OUT OF BOUNDS"
        else:
            status = "TIMEOUT"

        ax.set_title(
            f"Asteroid: ({asteroid_x:.0f}, {asteroid_y:.0f})\n"
            f"{status} | Steps: {result['steps']}"
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.suptitle(
        "PPO Generalization Test: Different Asteroid Positions",
        fontsize=14,
    )

    plt.tight_layout()
    plt.show()


def main():
    agent = PPOAgent()
    agent.load(MODEL_PATH)

    results = []

    print("\nPPO ASTEROID POSITION TEST")
    print("=" * 65)

    for asteroid_x, asteroid_y in SCENARIOS:
        result = run_scenario(agent, asteroid_x, asteroid_y)
        results.append(result)

        print(
            f"\nAsteroid: ({asteroid_x:.0f}, {asteroid_y:.0f})"
        )
        print("Goal reached:", result["goal_reached"])
        print("Collision:", result["collision"])
        print("Out of bounds:", result["out_of_bounds"])
        print("Timeout:", result["timeout"])
        print("Steps:", result["steps"])
        print(
            "Total reward:",
            round(result["total_reward"], 3),
        )
        print(
            "Minimum clearance:",
            round(result["minimum_clearance"], 3),
        )

    plot_results(results)


if __name__ == "__main__":
    main()
