import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    """PPO-based reinforcement learning agent."""

    def __init__(self, model_path: str | None = None):
        self.model = None

        if model_path is not None:
            self.load(model_path)

    def create_model(self, env, seed: int = 42):
        """Create a new PPO model for training."""
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            verbose=1,
            seed=seed,
        )

    def train(self, total_timesteps: int = 100_000):
        """Train the PPO agent."""
        if self.model is None:
            raise RuntimeError("Create a model before training.")

        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict an action using the trained policy."""
        if self.model is None:
            raise RuntimeError("PPO model is not initialized.")

        action, _ = self.model.predict(
            observation,
            deterministic=True,
        )
        return np.asarray(action, dtype=np.float32)

    def save(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise RuntimeError("No PPO model to save.")

        self.model.save(path)

    def load(self, path: str):
        """Load a trained PPO model."""
        self.model = PPO.load(path)
