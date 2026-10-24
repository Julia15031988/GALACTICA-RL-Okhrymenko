import numpy as np

from galactica.environment.spaceship import SpaceshipState
from galactica.environment.asteroid import AsteroidState


def generate_asteroid(
    rng: np.random.Generator,
    width: float,
    height: float,
    min_radius: float,
    max_radius: float,
) -> AsteroidState:
    """Generate a random stationary asteroid."""

    if width <= 0 or height <= 0:
        raise ValueError("Field dimensions must be positive.")

    if min_radius <= 0 or max_radius < min_radius:
        raise ValueError("Invalid asteroid radius range.")

    radius = rng.uniform(min_radius, max_radius)

    if width < 2 * radius or height < 2 * radius:
        raise ValueError("Asteroid does not fit inside the field.")

    x = rng.uniform(radius, width - radius)
    y = rng.uniform(radius, height - radius)

    return AsteroidState(
        x=x,
        y=y,
        radius=radius,
    )

def generate_asteroid_field(
    rng: np.random.Generator,
    spaceship: SpaceshipState,
    count: int,
    width: float,
    height: float,
    min_radius: float,
    max_radius: float,
    safety_margin: float = 5.0,
    max_attempts: int = 1000,
) -> list[AsteroidState]:
    """Generate asteroids outside the spaceship's safety zone."""

    if count < 0:
        raise ValueError("Asteroid count cannot be negative.")

    if safety_margin < 0:
        raise ValueError("Safety margin cannot be negative.")

    if max_attempts <= 0:
        raise ValueError("Max attempts must be positive.")

    asteroids = []

    for _ in range(count):
        for _ in range(max_attempts):
            asteroid = generate_asteroid(
                rng=rng,
                width=width,
                height=height,
                min_radius=min_radius,
                max_radius=max_radius,
            )

            distance = np.hypot(
                spaceship.x - asteroid.x,
                spaceship.y - asteroid.y,
            )

            min_distance = (
                spaceship.radius
                + asteroid.radius
                + safety_margin
            )

            if distance >= min_distance:
                overlaps = any(
                    np.hypot(
                        asteroid.x - other.x,
                        asteroid.y - other.y,
                    ) < asteroid.radius + other.radius
                    for other in asteroids
                )

                if not overlaps:
                    asteroids.append(asteroid)
                    break

        else:
            raise ValueError(
                "Could not place all asteroids safely."
            )

    return asteroids
