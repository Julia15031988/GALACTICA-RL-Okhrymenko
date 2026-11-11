from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from galactica.environment.space_env_v2 import SpaceEnvV2


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "results" / "ppo_v2_empty"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Training configuration
TOTAL_TIMESTEPS = 100_000
SEED = 42


def main():
    print("=" * 60)
    print("GALACTICA-RL V2.1: PPO TRAINING")
    print("=" * 60)

    print("Environment: EMPTY SPACE")
    print("Radar rays: 72")
    print("Observation size: 81")
    print("Actions: 3")
    print("Total timesteps:", TOTAL_TIMESTEPS)

    # Create environment without asteroids
    env = SpaceEnvV2(asteroid_count=0)

    # Monitor episode rewards and lengths
    env = Monitor(
        env,
        filename=str(LOGS_DIR / "monitor.csv"),
        info_keywords=(
            "collision",
            "goal_reached",
            "out_of_bounds",
            "distance_to_goal",
        ),
    )

    # Save intermediate checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(MODELS_DIR),
        name_prefix="ppo_v2_empty_checkpoint",
    )

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=SEED,
        device="auto",
        tensorboard_log=None,
    )

    print("\nStarting PPO training...\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=False,
    )

    # Save final model
    model_path = MODELS_DIR / "ppo_v2_empty_100k"

    model.save(str(model_path))

    print("\nTRAINING COMPLETED")
    print("Model saved:", model_path.with_suffix(".zip"))

    env.close()

    # Quick evaluation on the same empty scenario
    print("\nQUICK EVALUATION")
    print("-" * 60)

    eval_env = SpaceEnvV2(asteroid_count=0)

    success_count = 0
    episode_count = 10

    for episode in range(episode_count):
        obs, _ = eval_env.reset(seed=100 + episode)

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = (
                eval_env.step(action)
            )

            total_reward += reward
            steps += 1

        success = info.get("goal_reached", False)

        if success:
            success_count += 1

        print(
            f"Episode {episode + 1}: "
            f"success={success}, "
            f"steps={steps}, "
            f"reward={total_reward:.3f}, "
            f"fuel={eval_env.spaceship.fuel:.2f}, "
            f"out_of_bounds={info.get('out_of_bounds')}"
        )

    print("\nRESULTS")
    print(
        f"Success rate: "
        f"{success_count}/{episode_count} "
        f"({success_count / episode_count * 100:.1f}%)"
    )

    eval_env.close()

    print("\nGALACTICA-RL V2.1: TRAINING FINISHED")


if __name__ == "__main__":
    main()
