from stable_baselines3.common.env_checker import check_env

from galactica.environment.space_env import SpaceEnv


def test_space_env_compatibility():
    env = SpaceEnv()

    check_env(env, warn=True)

    env.close()
