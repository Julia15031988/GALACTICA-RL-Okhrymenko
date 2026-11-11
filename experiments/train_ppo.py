from pathlib import Path

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.space_env import SpaceEnv


def main():
    env = SpaceEnv()
    agent = PPOAgent()

    agent.create_model(env, seed=42)

    print("Starting PPO training...")

    agent.train(total_timesteps=100_000)

    model_path = Path("models/ppo_galactica_shaped_100k")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    agent.save(str(model_path))

    env.close()

    print("PPO TRAINING COMPLETED")
    print(f"Model saved to: {model_path}.zip")


if __name__ == "__main__":
    main()
