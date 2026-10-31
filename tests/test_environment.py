from galactica.environment.space_env import SpaceEnv


def test_reset_generates_asteroids():
    env = SpaceEnv()

    observation, info = env.reset(seed=42)

    assert observation.shape == (25,)
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

def test_goal_reached():
    env = SpaceEnv()
    env.reset(seed=42)

    env.asteroids = []

    env.spaceship.x = env.goal_x
    env.spaceship.y = env.goal_y

    observation, reward, terminated, truncated, info = env.step(
        [0.0, -1.0]
    )

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    assert info["goal_reached"] is True

def test_episode_truncated_after_max_steps():
    env = SpaceEnv()
    env.reset(seed=42)

    env.asteroids = []
    env.max_steps = 3

    for _ in range(3):
        observation, reward, terminated, truncated, info = env.step(
            [0.0, -1.0]
        )

    assert terminated is False
    assert truncated is True
    assert env.current_step == 3
