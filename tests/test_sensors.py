
import numpy as np

from galactica.environment.asteroid import AsteroidState
from galactica.environment.sensors import scan_radar
from galactica.environment.spaceship import SpaceshipState


def test_radar_without_asteroids():
    ship = SpaceshipState(x=10.0, y=10.0, radius=2.0)

    result = scan_radar(ship, [])

    assert result.shape == (16,)
    assert np.all(result == 1.0)


def test_radar_detects_asteroid_ahead():
    ship = SpaceshipState(x=10.0, y=10.0, radius=2.0)
    asteroid = AsteroidState(x=30.0, y=10.0, radius=5.0)

    result = scan_radar(
        ship,
        [asteroid],
        num_rays=16,
        max_distance=50.0,
    )

    assert np.isclose(result[0], 0.3)
    assert result[8] == 1.0

def test_radar_detects_nearest_asteroid():
    ship = SpaceshipState(x=10.0, y=10.0, radius=2.0)

    near = AsteroidState(x=20.0, y=10.0, radius=2.0)
    far = AsteroidState(x=40.0, y=10.0, radius=2.0)

    result = scan_radar(
        ship,
        [far, near],
        max_distance=50.0,
    )

    # Nearest surface: 20 - 10 - 2 = 8
    assert np.isclose(result[0], 8.0 / 50.0)


def test_radar_rotates_with_spaceship():
    ship = SpaceshipState(
        x=10.0,
        y=10.0,
        radius=2.0,
        angle=np.pi / 2,
    )

    asteroid = AsteroidState(
        x=10.0,
        y=30.0,
        radius=5.0,
    )

    result = scan_radar(
        ship,
        [asteroid],
        max_distance=50.0,
    )

    # Ray 0 now points upward along Y
    assert np.isclose(result[0], 0.3)
    assert result[8] == 1.0
