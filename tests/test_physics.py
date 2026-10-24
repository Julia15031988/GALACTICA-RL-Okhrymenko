import pytest
from galactica.environment.asteroid import AsteroidState
from galactica.environment.physics import move_asteroid


def test_move_asteroid():
    asteroid = AsteroidState(
        x=10.0,
        y=20.0,
        radius=3.0,
        vx=2.0,
        vy=-1.0,
    )

    moved = move_asteroid(asteroid, dt=2.0)

    assert moved.x == 14.0
    assert moved.y == 18.0

    # Original asteroid must remain unchanged
    assert asteroid.x == 10.0
    assert asteroid.y == 20.0


def test_stationary_asteroid():
    asteroid = AsteroidState(
        x=10.0,
        y=20.0,
        radius=3.0,
    )

    moved = move_asteroid(asteroid, dt=5.0)

    assert moved.x == 10.0
    assert moved.y == 20.0


def test_negative_time_step():
    asteroid = AsteroidState(
        x=10.0,
        y=20.0,
        radius=3.0,
    )

    with pytest.raises(ValueError):
        move_asteroid(asteroid, dt=-1.0)
