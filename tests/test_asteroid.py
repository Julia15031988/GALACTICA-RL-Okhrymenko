import pytest

from galactica.environment.asteroid import AsteroidState


def test_asteroid_creation():
    asteroid = AsteroidState(
        x=10.0,
        y=20.0,
        radius=5.0,
    )

    assert asteroid.x == 10.0
    assert asteroid.y == 20.0
    assert asteroid.radius == 5.0
    assert asteroid.vx == 0.0
    assert asteroid.vy == 0.0


def test_moving_asteroid():
    asteroid = AsteroidState(
        x=10.0,
        y=20.0,
        radius=5.0,
        vx=2.0,
        vy=-1.0,
    )

    assert asteroid.vx == 2.0
    assert asteroid.vy == -1.0


def test_invalid_asteroid_radius():
    with pytest.raises(ValueError, match="radius"):
        AsteroidState(
            x=10.0,
            y=20.0,
            radius=-5.0,
        )
