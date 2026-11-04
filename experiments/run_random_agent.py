from galactica.agents.baseline_agent import RandomAgent
from galactica.environment.space_env import SpaceEnv
from galactica.evaluation.metrics import (
    EpisodeResult,
    calculate_metrics,
)


def run_episode(seed: int) -> EpisodeResult:
    """Run one reproducible navigation episode."""

    env = SpaceEnv()
    agent = RandomAgent(seed=seed)

    observation, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    while True:
        action = agent.predict(observation)

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    result = EpisodeResult(
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        goal_reached=info["goal_reached"],
        collision=info["collision"],
        out_of_bounds=info["out_of_bounds"],
        fuel_used=100.0 - env.spaceship.fuel,
    )

    env.close()
    return result


def main():
    results = []

    for seed in range(100):
        result = run_episode(seed)
        results.append(result)

    metrics = calculate_metrics(results)

    print("=== RANDOM AGENT: 100 EPISODES ===")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Collision rate: {metrics['collision_rate']:.2%}")
    print(f"Average steps: {metrics['average_steps']:.2f}")
    print(f"Average reward: {metrics['average_reward']:.3f}")
    print(f"Average fuel used: {metrics['average_fuel_used']:.2f}")


if __name__ == "__main__":
    main()
