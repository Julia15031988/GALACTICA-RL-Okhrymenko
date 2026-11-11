import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.space_env import SpaceEnv


def main():
    env = SpaceEnv(asteroid_count=0)

    agent = PPOAgent(
        model_path="models/ppo_empty_100k.zip"
    )

    observation, _ = env.reset(seed=42)

    xs = [float(observation[0])]
    ys = [float(observation[1])]
    speeds = [0.0]

    total_reward = 0.0
    info = {}

    for step in range(env.max_steps):
        action = agent.predict(observation)

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        x = float(observation[0])
        y = float(observation[1])

        vx = float(observation[2])
        vy = float(observation[3])

        xs.append(x)
        ys.append(y)
        speeds.append(math.hypot(vx, vy))

        total_reward += float(reward)

        if terminated or truncated:
            break

    print("\n========== PPO EMPTY FIELD ==========")
    print(f"Steps: {len(xs) - 1}")
    print(f"Goal reached: {info['goal_reached']}")
    print(f"Collision: {info['collision']}")
    print(f"Out of bounds: {info['out_of_bounds']}")
    print(f"Final position: ({xs[-1]:.2f}, {ys[-1]:.2f})")
    print(f"Final speed: {speeds[-1]:.2f}")
    print(f"Total reward: {total_reward:.3f}")

    # Plot 1: spaceship trajectory.
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(
        xs,
        ys,
        color="royalblue",
        linewidth=2.5,
        marker=".",
        label="PPO trajectory",
    )

    ax.scatter(
        xs[0], ys[0],
        color="blue",
        s=100,
        label="Start",
        zorder=5,
    )

    ax.scatter(
        xs[-1], ys[-1],
        color="red",
        s=100,
        label="Final position",
        zorder=5,
    )

    ax.scatter(
        env.goal_x,
        env.goal_y,
        color="green",
        marker="*",
        s=250,
        label="Goal",
        zorder=5,
    )

    ax.add_patch(
        Circle(
            (env.goal_x, env.goal_y),
            env.goal_radius,
            color="green",
            alpha=0.15,
        )
    )

    ax.set_xlim(0, env.field_width)
    ax.set_ylim(0, env.field_height)
    ax.set_aspect("equal")
    ax.set_title("GALACTICA-RL: PPO Navigation Without Asteroids")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(alpha=0.3)
    ax.legend()

    # Plot 2: spaceship speed.
    fig2, ax2 = plt.subplots(figsize=(9, 4))

    ax2.plot(
        range(len(speeds)),
        speeds,
        marker="o",
        color="darkorange",
    )

    ax2.set_title("Spaceship Speed During Flight")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Speed")
    ax2.grid(alpha=0.3)

    plt.show()
    env.close()


if __name__ == "__main__":
    main()
