from dataclasses import dataclass


@dataclass
class EpisodeResult:
    """Results of a single navigation episode."""

    seed: int
    steps: int
    total_reward: float
    goal_reached: bool
    collision: bool
    out_of_bounds: bool
    fuel_used: float


def calculate_metrics(results: list[EpisodeResult]) -> dict:
    """Calculate evaluation metrics across multiple episodes."""

    if not results:
        raise ValueError("Results list cannot be empty.")

    total_episodes = len(results)

    success_rate = sum(
        result.goal_reached for result in results
    ) / total_episodes

    collision_rate = sum(
        result.collision for result in results
    ) / total_episodes

    average_steps = sum(
        result.steps for result in results
    ) / total_episodes

    average_reward = sum(
        result.total_reward for result in results
    ) / total_episodes

    average_fuel_used = sum(
        result.fuel_used for result in results
    ) / total_episodes

    return {
        "episodes": total_episodes,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "average_steps": average_steps,
        "average_reward": average_reward,
        "average_fuel_used": average_fuel_used,
    }
