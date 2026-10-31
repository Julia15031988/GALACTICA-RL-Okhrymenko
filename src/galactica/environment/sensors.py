import math

import numpy as np

from galactica.environment.asteroid import AsteroidState
from galactica.environment.spaceship import SpaceshipState


def scan_radar(
    spaceship: SpaceshipState,
    asteroids: list[AsteroidState],
    num_rays: int = 16,
    max_distance: float = 50.0,
) -> np.ndarray:
    """Measure distances to asteroids along radar rays."""

    if num_rays <= 0:
        raise ValueError("Number of rays must be positive.")

    if max_distance <= 0:
        raise ValueError("Radar range must be positive.")

    distances = np.ones(num_rays, dtype=np.float32)

    for i in range(num_rays):
        angle = spaceship.angle + 2 * math.pi * i / num_rays

        dx = math.cos(angle)
        dy = math.sin(angle)

        for asteroid in asteroids:
            ox = spaceship.x - asteroid.x
            oy = spaceship.y - asteroid.y

            b = ox * dx + oy * dy
            c = ox**2 + oy**2 - asteroid.radius**2

            discriminant = b**2 - c

            if discriminant < 0:
                continue

            distance = -b - math.sqrt(discriminant)

            if distance < 0:
                distance = 0.0 if c <= 0 else -b + math.sqrt(discriminant)

            if 0 <= distance <= max_distance:
                normalized = distance / max_distance
                distances[i] = min(distances[i], normalized)

    return distances
