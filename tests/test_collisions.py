
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
