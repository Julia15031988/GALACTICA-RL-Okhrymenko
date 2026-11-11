from pathlib import Path

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.space_env import SpaceEnv


def main():
    model_path = Path("models/ppo_galactica_shaped_100k.zip")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    agent = PPOAgent(model_path=str(model_path))

    successes = 0
    collisions = 0
    out_of_bounds = 0
    total_steps = 0
    total_reward = 0.0

    for seed in range(100):
        env = SpaceEnv()
        observation, _ = env.reset(seed=seed)

        terminated = False
        truncated = False
        steps = 0
        episode_reward = 0.0

        while not (terminated or truncated):
            action = agent.predict(observation)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            steps += 1
            episode_reward += reward

        successes += int(info.get("goal_reached", False))
        collisions += int(info.get("collision", False))
        out_of_bounds += int(info.get("out_of_bounds", False))

        total_steps += steps
        total_reward += episode_reward

        env.close()

    print(f"Successes: {successes}/100")
    print(f"Collisions: {collisions}")
    print(f"Out of bounds: {out_of_bounds}")
    print(f"Average steps: {total_steps / 100:.2f}")
    print(f"Average reward: {total_reward / 100:.3f}")


if __name__ == "__main__":
    main()
