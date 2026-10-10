import pytest

from galactica.environment.spaceship import SpaceshipState


def test_spaceship_creation():
    ship = SpaceshipState(x=10.0, y=20.0, radius=3.0)

    assert ship.x == 10.0
    assert ship.y == 20.0
    assert ship.radius == 3.0
    assert ship.vx == 0.0
    assert ship.fuel == 100.0


def test_spaceship_invalid_radius():
    with pytest.raises(ValueError, match="radius"):
        SpaceshipState(x=0.0, y=0.0, radius=-2.0)


def test_spaceship_invalid_fuel():
    with pytest.raises(ValueError, match="Fuel"):
        SpaceshipState(
            x=0.0,
            y=0.0,
            radius=3.0,
            fuel=-10.0,
        )
