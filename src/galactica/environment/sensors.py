
import math

import numpy as np

from galactica.environment.asteroid import AsteroidState
from galactica.environment.spaceship import SpaceshipState


def scan_radar(
    spaceship: SpaceshipState,
    asteroids: list[AsteroidState],
    num_rays: int = 16,
    max_distance: float = 50.0,
    field_width: float = 100.0,
    field_height: float = 100.0,
) -> np.ndarray:
    """
    Measure normalized distances from the spaceship body
    to the nearest asteroid or field boundary along radar rays.

    Returns values in [0, 1]:
        0.0 - obstacle touches the spaceship body
        1.0 - no obstacle within radar range
    """

    if num_rays <= 0:
        raise ValueError("Number of rays must be positive.")

    if max_distance <= 0:
        raise ValueError("Radar range must be positive.")

    if field_width <= 0 or field_height <= 0:
        raise ValueError("Field dimensions must be positive.")

    distances = np.ones(num_rays, dtype=np.float32)

    for i in range(num_rays):
        angle = spaceship.angle + 2 * math.pi * i / num_rays

        dx = math.cos(angle)
        dy = math.sin(angle)

        nearest_distance = max_distance

        # 1. Check asteroid intersections
        for asteroid in asteroids:
            ox = spaceship.x - asteroid.x
            oy = spaceship.y - asteroid.y

            b = ox * dx + oy * dy
            c = ox**2 + oy**2 - asteroid.radius**2

            discriminant = b**2 - c

            if discriminant < 0:
                continue

            sqrt_discriminant = math.sqrt(discriminant)

            if c <= 0:
                distance = 0.0
            else:
                distance = -b - sqrt_discriminant

                if distance < 0:
                    distance = -b + sqrt_discriminant

            if distance >= 0:
                body_distance = max(
                    0.0,
                    distance - spaceship.radius,
                )

                nearest_distance = min(
                    nearest_distance,
                    body_distance,
                )

        # 2. Check vertical field boundaries
        if dx > 0:
            wall_distance = (field_width - spaceship.x) / dx

        elif dx < 0:
            wall_distance = -spaceship.x / dx

        else:
            wall_distance = math.inf

        if wall_distance >= 0:
            body_distance = max(
                0.0,
                wall_distance - spaceship.radius,
            )

            nearest_distance = min(
                nearest_distance,
                body_distance,
            )

        # 3. Check horizontal field boundaries
        if dy > 0:
            wall_distance = (field_height - spaceship.y) / dy

        elif dy < 0:
            wall_distance = -spaceship.y / dy

        else:
            wall_distance = math.inf

        if wall_distance >= 0:
            body_distance = max(
                0.0,
                wall_distance - spaceship.radius,
            )

            nearest_distance = min(
                nearest_distance,
                body_distance,
            )

        # 4. Normalize radar reading
        distances[i] = min(
            1.0,
            nearest_distance / max_distance,
        )

    return distances
