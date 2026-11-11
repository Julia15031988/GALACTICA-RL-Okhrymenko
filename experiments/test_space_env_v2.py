import numpy as np
from stable_baselines3.common.env_checker import check_env

from galactica.environment.space_env_v2 import SpaceEnvV2


def main():
    env = SpaceEnvV2(asteroid_count=20)

    print("TEST 1: Gymnasium compatibility")
    check_env(env, warn=True)
    print("PASSED")

    print("\nTEST 2: Observation")

    observation, info = env.reset(seed=42)

    print("Shape:", observation.shape)
    print("Values:", np.round(observation, 3))

    assert observation.shape == (25,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)

    print("PASSED")

    print("\nTEST 3: Four control actions")

    assert env.action_space.shape == (4,)
    print("Action space:", env.action_space)

    print("PASSED")

    print("\nTEST 4: Forward thrust")

    env.reset(seed=42)

    action = np.array(
        [0.0, 1.0, -1.0, 0.0],
        dtype=np.float32,
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("Position:", env.spaceship.x, env.spaceship.y)
    print("Velocity:", env.spaceship.vx, env.spaceship.vy)
    print("Reward:", reward)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    print("PASSED")

    print("\nTEST 5: Random actions")

    env.reset(seed=42)

    for step in range(100):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break

    print("Steps:", step + 1)
    print("Info:", info)
    print("PASSED")

    env.close()

    print("\nALL SPACE ENV V2 TESTS PASSED!")


if __name__ == "__main__":
    main()
