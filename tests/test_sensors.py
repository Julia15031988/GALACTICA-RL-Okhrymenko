import numpy as np
import pytest

from galactica.environment.asteroid import AsteroidState
from galactica.environment.sensors import scan_radar
from galactica.environment.spaceship import SpaceshipState


def test_radar_without_asteroids():
    ship = SpaceshipState(
        x=10.0,
        y=10.0,
        radius=2.0,
    )

    result = scan_radar(ship, [])

    assert result.shape == (16,)

    # Ray 0 points right: wall is beyond radar range.
    assert result[0] == pytest.approx(1.0)

    # Ray 8 points left: wall is 10 units
    # from the center, minus ship radius 2.
    assert result[8] == pytest.approx(8.0 / 50.0)

    # Ray 12 points down: same free distance.
    assert result[12] == pytest.approx(8.0 / 50.0)


def test_radar_detects_asteroid_ahead():
    ship = SpaceshipState(
        x=10.0,
        y=10.0,
        radius=2.0,
    )

    asteroid = AsteroidState(
        x=30.0,
        y=10.0,
        radius=5.0,
    )

    result = scan_radar(
        ship,
        [asteroid],
        num_rays=16,
        max_distance=50.0,
    )

    assert result[0] == pytest.approx(13.0 / 50.0)


def test_radar_detects_nearest_asteroid():
    ship = SpaceshipState(
        x=10.0,
        y=10.0,
        radius=2.0,
    )

    near = AsteroidState(
        x=20.0,
        y=10.0,
        radius=2.0,
    )

    far = AsteroidState(
        x=40.0,
        y=10.0,
        radius=2.0,
    )

    result = scan_radar(
        ship,
        [far, near],
        max_distance=50.0,
    )

    # Nearest asteroid:
    # 20 - 10 - 2 (asteroid) - 2 (ship) = 6
    assert result[0] == pytest.approx(6.0 / 50.0)


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

    # Ray 0 points along positive Y after rotation.
    assert result[0] == pytest.approx(13.0 / 50.0)

def test_radar_detects_all_four_walls():
    ship = SpaceshipState(
        x=50.0,
        y=50.0,
        radius=3.0,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )

    readings = scan_radar(
        spaceship=ship,
        asteroids=[],
        num_rays=16,
        max_distance=50.0,
    )

    expected = (50.0 - 3.0) / 50.0

    assert readings[0] == pytest.approx(expected)
    assert readings[4] == pytest.approx(expected)
    assert readings[8] == pytest.approx(expected)
    assert readings[12] == pytest.approx(expected)


def test_radar_measures_distance_from_spaceship_body():
    ship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )

    readings = scan_radar(
        spaceship=ship,
        asteroids=[],
        num_rays=16,
        max_distance=50.0,
    )

    # The rear wall is 10 units from the ship center.
    # The ship radius is 3, so the free distance is 7.
    assert readings[8] == pytest.approx(7.0 / 50.0)


def test_radar_selects_nearest_obstacle():
    ship = SpaceshipState(
        x=50.0,
        y=50.0,
        radius=3.0,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )

    asteroid = AsteroidState(
        x=70.0,
        y=50.0,
        radius=5.0,
    )

    readings = scan_radar(
        spaceship=ship,
        asteroids=[asteroid],
        num_rays=16,
        max_distance=50.0,
    )

    # Asteroid surface: 70 - 5 = 65
    # Ship center: 50
    # Ship body radius: 3
    # Free distance: 65 - 50 - 3 = 12
    assert readings[0] == pytest.approx(12.0 / 50.0)

def test_radar_detects_diagonal_wall():
    ship = SpaceshipState(
        x=90.0,
        y=90.0,
        radius=3.0,
        angle=0.0,
    )

    readings = scan_radar(
        spaceship=ship,
        asteroids=[],
        num_rays=16,
        max_distance=50.0,
    )

    # Ray 2 points diagonally at 45 degrees.
    # Distance from center to wall along ray:
    # 10 / cos(45 degrees)
    center_distance = 10.0 / np.cos(np.pi / 4)

    # Subtract the spaceship radius.
    expected = (center_distance - 3.0) / 50.0

    assert readings[2] == pytest.approx(expected)


def test_radar_returns_zero_when_spaceship_touches_asteroid():
    ship = SpaceshipState(
        x=50.0,
        y=50.0,
        radius=3.0,
        angle=0.0,
    )

    asteroid = AsteroidState(
        x=58.0,
        y=50.0,
        radius=5.0,
    )

    readings = scan_radar(
        spaceship=ship,
        asteroids=[asteroid],
        num_rays=16,
        max_distance=50.0,
    )

    assert readings[0] == pytest.approx(0.0)
