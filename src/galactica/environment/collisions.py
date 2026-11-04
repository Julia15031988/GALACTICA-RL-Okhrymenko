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

def check_path_collision(
    start: SpaceshipState,
    end: SpaceshipState,
    asteroid: AsteroidState,
) -> bool:
    """Check collision along the spaceship's movement segment."""

    dx = end.x - start.x
    dy = end.y - start.y

    length_squared = dx**2 + dy**2

    if length_squared == 0:
        return check_collision(start, asteroid)

    t = (
        (asteroid.x - start.x) * dx
        + (asteroid.y - start.y) * dy
    ) / length_squared

    t = max(0.0, min(1.0, t))

    closest_x = start.x + t * dx
    closest_y = start.y + t * dy

    distance = math.hypot(
        asteroid.x - closest_x,
        asteroid.y - closest_y,
    )

    return distance <= start.radius + asteroid.radius
