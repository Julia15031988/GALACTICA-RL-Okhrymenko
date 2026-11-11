import numpy as np

from galactica.environment.space_env import SpaceEnv
from galactica.agents.heuristic_agent import HeuristicAgent


def evaluate_heuristic(num_episodes: int = 100):
    successes = 0
    collisions = 0
    out_of_bounds = 0
    total_steps = 0
    total_rewards = 0.0

    for seed in range(num_episodes):
        env = SpaceEnv()
        agent = HeuristicAgent()  # Fresh memory for each episode

        observation, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        steps = 0
        episode_reward = 0.0

        while not (terminated or truncated):
            action = agent.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            steps += 1
            episode_reward += reward

        successes += int(info.get("goal_reached", False))
        collisions += int(info.get("collision", False))
        out_of_bounds += int(info.get("out_of_bounds", False))
        total_steps += steps
        total_rewards += episode_reward

        env.close()

    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {successes / num_episodes:.1%}")
    print(f"Collisions: {collisions}")
    print(f"Out of bounds: {out_of_bounds}")
    print(f"Average steps: {total_steps / num_episodes:.2f}")
    print(f"Average reward: {total_rewards / num_episodes:.3f}")


if __name__ == "__main__":
    evaluate_heuristic()
