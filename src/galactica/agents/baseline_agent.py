import numpy as np


class RandomAgent:
    """Baseline agent that selects random actions."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return a random steering and thrust action."""

        action = self.rng.uniform(
            low=-1.0,
            high=1.0,
            size=2,
        )

        return action.astype(np.float32)
