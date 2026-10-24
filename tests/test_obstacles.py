import numpy as np

from galactica.environment.obstacles import (
    generate_asteroid,
    generate_asteroid_field,
)
from galactica.environment.spaceship import SpaceshipState


def test_generate_asteroid():
    rng = np.random.default_rng(seed=42)

    asteroid = generate_asteroid(
        rng=rng,
        width=100.0,
        height=100.0,
        min_radius=2.0,
        max_radius=8.0,
    )

    assert 2.0 <= asteroid.radius <= 8.0
    assert asteroid.radius <= asteroid.x <= 100.0 - asteroid.radius
    assert asteroid.radius <= asteroid.y <= 100.0 - asteroid.radius
    assert asteroid.vx == 0.0
    assert asteroid.vy == 0.0

def test_generate_asteroid_field():
    rng = np.random.default_rng(seed=42)

    spaceship = SpaceshipState(
        x=50.0,
        y=50.0,
        radius=5.0,
    )

    asteroids = generate_asteroid_field(
        rng=rng,
        spaceship=spaceship,
        count=20,
        width=100.0,
        height=100.0,
        min_radius=2.0,
        max_radius=5.0,
        safety_margin=5.0,
    )

    assert len(asteroids) == 20

    for asteroid in asteroids:
        distance = np.hypot(
            spaceship.x - asteroid.x,
            spaceship.y - asteroid.y,
        )

        assert distance >= (
            spaceship.radius + asteroid.radius + 5.0
        )

def test_asteroid_field_reproducibility():
    spaceship = SpaceshipState(
        x=50.0,
        y=50.0,
        radius=5.0,
    )

    rng1 = np.random.default_rng(seed=42)
    rng2 = np.random.default_rng(seed=42)

    field1 = generate_asteroid_field(
        rng=rng1,
        spaceship=spaceship,
        count=20,
        width=100.0,
        height=100.0,
        min_radius=2.0,
        max_radius=5.0,
    )

    field2 = generate_asteroid_field(
        rng=rng2,
        spaceship=spaceship,
        count=20,
        width=100.0,
        height=100.0,
        min_radius=2.0,
        max_radius=5.0,
    )

    assert field1 == field2

def test_asteroids_do_not_overlap():
    rng = np.random.default_rng(seed=42)

    spaceship = SpaceshipState(
        x=50.0,
        y=50.0,
        radius=5.0,
    )

    asteroids = generate_asteroid_field(
        rng=rng,
        spaceship=spaceship,
        count=20,
        width=100.0,
        height=100.0,
        min_radius=2.0,
        max_radius=5.0,
    )

    for i in range(len(asteroids)):
        for j in range(i + 1, len(asteroids)):
            first = asteroids[i]
            second = asteroids[j]

            distance = np.hypot(
                first.x - second.x,
                first.y - second.y,
            )

            assert distance >= first.radius + second.radius
