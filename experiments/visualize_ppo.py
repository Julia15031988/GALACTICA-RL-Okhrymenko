import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.space_env import SpaceEnv


def main():
    env = SpaceEnv()

    agent = PPOAgent(
        model_path="models/ppo_galactica_shaped_100k.zip"
    )

    observation, _ = env.reset(seed=42)

    xs = [float(observation[0])]
    ys = [float(observation[1])]

    total_reward = 0.0
    info = {}
    terminated = False
    truncated = False

    print("\n========== PPO RADAR DIAGNOSTICS ==========")

    for step in range(env.max_steps):
        # Radar readings occupy observation indices 7-22.
        radar = observation[7:23]

        speed = math.hypot(
            float(observation[2]),
            float(observation[3]),
        )

        action = agent.predict(observation)

        if step < 20:
            print(
                f"Step {step:3d} | "
                f"Position: ({observation[0]:6.2f}, "
                f"{observation[1]:6.2f}) | "
                f"Speed: {speed:5.2f} | "
                f"Front radar: {radar[0]:.3f} | "
                f"Nearest radar: {radar.min():.3f} | "
                f"Steering: {action[0]:+.3f} | "
                f"Thrust action: {action[1]:+.3f}"
            )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        xs.append(float(observation[0]))
        ys.append(float(observation[1]))
        total_reward += float(reward)

        if terminated or truncated:
            break

    print("===========================================\n")

    print("========== PPO FLIGHT RESULTS ==========")
    print("Seed: 42")
    print(f"Steps: {len(xs) - 1}")
    print(f"Start: ({xs[0]:.2f}, {ys[0]:.2f})")
    print(f"End: ({xs[-1]:.2f}, {ys[-1]:.2f})")

    final_speed = math.hypot(
        float(observation[2]),
        float(observation[3]),
    )

    print(f"Final speed: {final_speed:.2f}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Collision: {info.get('collision', False)}")
    print(f"Goal reached: {info.get('goal_reached', False)}")
    print(f"Out of bounds: {info.get('out_of_bounds', False)}")
    print(f"Time limit reached: {truncated}")
    print(f"Distance to goal: {info.get('distance_to_goal')}")
    print("========================================\n")

    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw asteroids.
    for asteroid in env.asteroids:
        circle = Circle(
            (asteroid.x, asteroid.y),
            asteroid.radius,
            color="gray",
            alpha=0.55,
        )
        ax.add_patch(circle)

    # Draw the spaceship trajectory.
    ax.plot(
        xs,
        ys,
        color="royalblue",
        linewidth=2,
        label="PPO trajectory",
        zorder=3,
    )

    ax.scatter(
        xs[0],
        ys[0],
        color="blue",
        marker="o",
        s=100,
        label="Start",
        zorder=5,
    )

    ax.scatter(
        xs[-1],
        ys[-1],
        color="red",
        marker="x",
        s=130,
        label="End",
        zorder=6,
    )

    # Draw the goal and its radius.
    goal_circle = Circle(
        (env.goal_x, env.goal_y),
        env.goal_radius,
        color="green",
        alpha=0.15,
    )
    ax.add_patch(goal_circle)

    ax.scatter(
        env.goal_x,
        env.goal_y,
        color="green",
        marker="*",
        s=250,
        label="Goal",
        zorder=5,
    )

    # Draw the spaceship collision radius at the end.
    spaceship_circle = Circle(
        (xs[-1], ys[-1]),
        env.spaceship.radius,
        fill=False,
        edgecolor="red",
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(spaceship_circle)

    ax.set_xlim(0, env.field_width)
    ax.set_ylim(0, env.field_height)
    ax.set_aspect("equal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        "GALACTICA-RL | PPO trajectory | seed=42"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

    env.close()


if __name__ == "__main__":
    main()
