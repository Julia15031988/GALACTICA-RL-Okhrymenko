import math

from galactica.agents.heuristic_agent import HeuristicAgent
from galactica.environment.space_env import SpaceEnv


def main():
    env = SpaceEnv()
    agent = HeuristicAgent()

    observation, _ = env.reset(seed=42)

    total_reward = 0.0
    steps = 0

    while True:
        action = agent.predict(observation)
        print(
            f"MODE: {agent.mode} | "
            f"AVOID DIRECTION: {agent.avoid_direction}"
        )
        # Radar readings BEFORE the agent makes its decision.
        radar = observation[7:23]

        print(
            f"RADAR | "
            f"Front: {radar[0]:.2f} | "
            f"Left 45°: {radar[2]:.2f} | "
            f"Right 45°: {radar[14]:.2f} | "
            f"Speed: {math.hypot(observation[2], observation[3]):.2f}"
        )

        observation, reward, terminated, truncated, info = env.step(
            action
        )
        print(
            f"Step {steps + 1:3d} | "
            f"Position: ({env.spaceship.x:.2f}, "
            f"{env.spaceship.y:.2f}) | "
            f"Speed: {math.hypot(env.spaceship.vx, env.spaceship.vy):.2f} | "
            f"Angle: {env.spaceship.angle:.2f} | "
            f"Steering: {action[0]:.2f} | "
            f"Thrust: {action[1]:.2f} | "
            f"Front radar: {observation[7]:.2f}"
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    speed = math.hypot(
        env.spaceship.vx,
        env.spaceship.vy,
    )

    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Goal reached: {info.get('goal_reached')}")
    print(f"Collision: {info.get('collision')}")
    print(f"Out of bounds: {info.get('out_of_bounds')}")
    print(f"Distance to goal: {info.get('distance_to_goal')}")
    print(f"Remaining fuel: {env.spaceship.fuel:.2f}")
    print(f"Final speed: {speed:.2f}")

    env.close()


if __name__ == "__main__":
    main()
