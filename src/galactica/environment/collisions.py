import math

from galactica.environment.asteroid import AsteroidState
from galactica.environment.spaceship import SpaceshipState


def check_collision(
    spaceship: SpaceshipState,
    asteroid: AsteroidState,
) -> bool:
    distance = math.hypot(
        spaceship.x - asteroid.x,
        spaceship.y - asteroid.y,
    )

    return distance <= spaceship.radius + asteroid.radius
