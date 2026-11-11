import numpy as np

from galactica.agents.ppo_agent import PPOAgent
from galactica.environment.asteroid import AsteroidState
from galactica.environment.random_one_asteroid_env import (
    RandomOneAsteroidEnv,
)


MODELS = {
    "PPO Fixed": "models/ppo_one_asteroid_100k",
    "PPO Random": "models/ppo_random_asteroid_100k",
}

POSITIONS = [
    (50.0, 50.0),
    (50.0, 60.0),
    (50.0, 40.0),
    (50.0, 70.0),
    (50.0, 30.0),
]


def get_observation(x, y):
    env = RandomOneAsteroidEnv()
    env.reset(seed=42)

    env.asteroids = [
        AsteroidState(
            x=x,
            y=y,
            radius=5.0,
            vx=0.0,
            vy=0.0,
        )
    ]

    observation = env._get_observation()

    env.close()
    return observation


def main():
    agents = {}

    for name, path in MODELS.items():
        agent = PPOAgent()
        agent.load(path)
        agents[name] = agent

    print("\nGALACTICA-RL: RADAR AND POLICY DIAGNOSTICS")
    print("=" * 72)

    for x, y in POSITIONS:
        observation = get_observation(x, y)
        radar = observation[7:23]

        print(f"\nASTEROID: ({x:.0f}, {y:.0f})")
        print("Front radar:", round(float(radar[0]), 4))
        print("Nearest radar:", round(float(np.min(radar)), 4))
        print("Radar readings:", np.round(radar, 3))

        for name, agent in agents.items():
            action = np.asarray(
                agent.predict(observation),
                dtype=float,
            )

            steering = float(action[0])
            thrust_command = float(action[1])

            # Environment maps [-1, 1] to [0, 1].
            actual_thrust = (thrust_command + 1.0) / 2.0

            print(f"\n{name}:")
            print("  Steering:", round(steering, 4))
            print("  Thrust command:", round(thrust_command, 4))
            print("  Actual thrust:", round(actual_thrust, 4))

    print("\n" + "=" * 72)
    print("DIAGNOSTICS COMPLETED")


if __name__ == "__main__":
    main()
