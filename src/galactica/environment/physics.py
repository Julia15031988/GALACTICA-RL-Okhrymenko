from dataclasses import replace

from galactica.environment.asteroid import AsteroidState


def move_asteroid(
    asteroid: AsteroidState,
    dt: float,
) -> AsteroidState:
    if dt < 0:
        raise ValueError("Time step cannot be negative.")

    return replace(
        asteroid,
        x=asteroid.x + asteroid.vx * dt,
        y=asteroid.y + asteroid.vy * dt,
    )
