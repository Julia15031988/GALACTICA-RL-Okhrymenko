from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.space_env import SpaceEnv


def main():
    env = SpaceEnv(asteroid_count=0)

    agent = PPOAgent(
        model_path="models/ppo_empty_100k.zip"
    )

    episodes = 100

    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0

    total_steps = 0
    total_reward = 0.0

    for seed in range(episodes):
        observation, _ = env.reset(seed=seed)

        episode_reward = 0.0

        for step in range(env.max_steps):
            action = agent.predict(observation)

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += float(reward)

            if terminated or truncated:
                break

        successes += int(info["goal_reached"])
        collisions += int(info["collision"])
        out_of_bounds += int(info["out_of_bounds"])
        timeouts += int(truncated)

        total_steps += step + 1
        total_reward += episode_reward

    print("\n========== PPO EMPTY FIELD EVALUATION ==========")
    print(f"Episodes: {episodes}")
    print(f"Successes: {successes}")
    print(f"Collisions: {collisions}")
    print(f"Out of bounds: {out_of_bounds}")
    print(f"Timeouts: {timeouts}")
    print(f"Average steps: {total_steps / episodes:.2f}")
    print(f"Average reward: {total_reward / episodes:.3f}")
    print(f"Success rate: {successes / episodes * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
