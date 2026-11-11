from pathlib import Path

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.random_one_asteroid_env import (
    RandomOneAsteroidEnv,
)


def main():
    env = RandomOneAsteroidEnv()

    agent = PPOAgent()
    agent.create_model(env, seed=42)

    print("=" * 55)
    print("GALACTICA-RL: RANDOM ASTEROID EXPERIMENT")
    print("=" * 55)
    print("Environment: RandomOneAsteroidEnv")
    print("Asteroids: 1")
    print("Training timesteps: 100,000")
    print("Starting PPO training...\n")

    agent.train(total_timesteps=100_000)

    model_path = Path("models/ppo_random_asteroid_100k")
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent.save(str(model_path))

    env.close()

    print("\n" + "=" * 55)
    print("PPO RANDOM ASTEROID TRAINING COMPLETED")
    print(f"Model saved to: {model_path}.zip")
    print("=" * 55)


if __name__ == "__main__":
    main()
