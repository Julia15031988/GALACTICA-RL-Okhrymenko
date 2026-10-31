from galactica.environment.space_env import SpaceEnv


def test_reset_generates_asteroids():
    env = SpaceEnv()

    observation, info = env.reset(seed=42)

    assert observation.shape == (7,)
    assert len(env.asteroids) == 20
    assert isinstance(info, dict)

def test_reset_reproducibility():
    env = SpaceEnv()

    observation1, _ = env.reset(seed=42)
    asteroids1 = list(env.asteroids)

    observation2, _ = env.reset(seed=42)
    asteroids2 = list(env.asteroids)

    assert (observation1 == observation2).all()
    assert asteroids1 == asteroids2
