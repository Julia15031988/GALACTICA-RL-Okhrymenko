from pathlib import Path

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.space_env import SpaceEnv


def main():
    # Create an environment without asteroids.
    env = SpaceEnv(asteroid_count=0)

    # Create a new PPO agent.
    agent = PPOAgent()
    agent.create_model(env, seed=42)

    print("Starting PPO training on empty field...")

    # Train for 100,000 timesteps.
    agent.train(total_timesteps=100_000)

    # Save separately from previous models.
    model_path = Path("models/ppo_empty_100k")
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent.save(str(model_path))

    env.close()

    print("PPO EMPTY FIELD TRAINING COMPLETED")
    print(f"Model saved to: {model_path}.zip")


if __name__ == "__main__":
    main()
