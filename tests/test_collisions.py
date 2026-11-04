from dataclasses import replace
from galactica.environment.collisions import check_path_collision
from galactica.environment.asteroid import AsteroidState
from galactica.environment.collisions import check_collision
from galactica.environment.spaceship import SpaceshipState


def test_collision():
    spaceship = SpaceshipState(x=0, y=0, radius=5)
    asteroid = AsteroidState(x=6, y=0, radius=3)

    assert check_collision(spaceship, asteroid) is True


def test_no_collision():
    spaceship = SpaceshipState(x=0, y=0, radius=5)
    asteroid = AsteroidState(x=20, y=0, radius=3)

    assert check_collision(spaceship, asteroid) is False


def test_touching_objects():
    spaceship = SpaceshipState(x=0, y=0, radius=5)
    asteroid = AsteroidState(x=8, y=0, radius=3)

    assert check_collision(spaceship, asteroid) is True

def test_collision_can_be_missed_between_positions():
    from galactica.environment.asteroid import AsteroidState
    from galactica.environment.spaceship import SpaceshipState
    from galactica.environment.collisions import check_collision

    ship = SpaceshipState(
        x=0.0, y=50.0, radius=3.0,
        vx=0.0, vy=0.0, angle=0.0, fuel=100.0,
    )

    asteroid = AsteroidState(
        x=10.0, y=50.0, radius=2.0,
        vx=0.0, vy=0.0,
    )

    assert check_collision(ship, asteroid) is False

    ship.x = 20.0

    assert check_collision(ship, asteroid) is False

def test_path_collision_detects_asteroid_between_positions():
    start = SpaceshipState(
        x=0.0,
        y=50.0,
        radius=3.0,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        fuel=100.0,
    )

    end = replace(start, x=20.0)

    asteroid = AsteroidState(
        x=10.0,
        y=50.0,
        radius=2.0,
        vx=0.0,
        vy=0.0,
    )

    assert check_path_collision(start, end, asteroid) is True
