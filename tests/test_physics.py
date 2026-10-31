import pytest
import math

from galactica.environment.asteroid import AsteroidState
from galactica.environment.physics import move_asteroid
from galactica.environment.physics import apply_spaceship_action
from galactica.environment.spaceship import SpaceshipState


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

def test_apply_spaceship_action():
    spaceship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
        angle=0.0,
        fuel=100.0,
    )

    moved = apply_spaceship_action(
        spaceship=spaceship,
        steering=0.0,
        thrust=1.0,
        dt=1.0,
    )

    assert moved.x == 11.0
    assert moved.y == 50.0
    assert moved.vx == 1.0
    assert moved.vy == 0.0
    assert moved.fuel == 99.0

    assert spaceship.x == 10.0
    assert spaceship.fuel == 100.0

def test_spaceship_turns():
    spaceship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
    )

    moved = apply_spaceship_action(
        spaceship=spaceship,
        steering=1.0,
        thrust=1.0,
        dt=1.0,
        max_turn_rate=math.pi / 2,
    )

    assert math.isclose(moved.angle, math.pi / 2)
    assert math.isclose(moved.x, 10.0, abs_tol=1e-9)
    assert math.isclose(moved.y, 51.0)

def test_spaceship_inertia():
    spaceship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
        vx=2.0,
        vy=0.0,
    )

    moved = apply_spaceship_action(
        spaceship=spaceship,
        steering=0.0,
        thrust=0.0,
        dt=1.0,
    )

    assert moved.x == 12.0
    assert moved.vx == 2.0
    assert moved.fuel == 100.0

def test_spaceship_without_fuel():
    spaceship = SpaceshipState(
        x=10.0,
        y=50.0,
        radius=3.0,
        vx=2.0,
        fuel=0.0,
    )

    moved = apply_spaceship_action(
        spaceship=spaceship,
        steering=0.0,
        thrust=1.0,
        dt=1.0,
    )

    assert moved.x == 12.0
    assert moved.vx == 2.0
    assert moved.fuel == 0.0
