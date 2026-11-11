import math

import numpy as np

from galactica.environment.space_env_v2 import SpaceEnvV2
from galactica.environment.sensors import scan_radar
from galactica.environment.asteroid import AsteroidState


def get_radar(env, num_rays):
    """Return radar readings for the requested number of rays."""

    return np.asarray(
        scan_radar(
            spaceship=env.spaceship,
            asteroids=env.asteroids,
            num_rays=num_rays,
            max_distance=50.0,
            field_width=env.field_width,
            field_height=env.field_height,
        ),
        dtype=np.float32,
    )


def test_radar():
    print("GALACTICA-RL: RADAR 16 VS 32 VS 72")
    print("=" * 60)

    env = SpaceEnvV2(asteroid_count=0)
    env.reset(seed=42)

    # Place spaceship in the center of the field.
    env.spaceship.x = 50.0
    env.spaceship.y = 50.0
    env.spaceship.angle = 0.0

    # Test configuration.
    asteroid_distance = 20.0
    asteroid_radius = 1.0

    ray_options = [16, 32, 72]

    # Baseline radar readings without asteroids.
    env.asteroids = []

    empty_readings = {
        rays: get_radar(env, rays)
        for rays in ray_options
    }

    detected_counts = {
        rays: 0
        for rays in ray_options
    }

    missed_directions = {
        rays: []
        for rays in ray_options
    }

    print("\nTesting 360 asteroid directions...")

    # Move one asteroid around the spaceship.
    for degrees in range(360):
        angle = math.radians(degrees)

        asteroid_x = (
            env.spaceship.x
            + asteroid_distance * math.cos(angle)
        )

        asteroid_y = (
            env.spaceship.y
            + asteroid_distance * math.sin(angle)
        )

        env.asteroids = [
            AsteroidState(
                x=asteroid_x,
                y=asteroid_y,
                radius=asteroid_radius,
                vx=0.0,
                vy=0.0,
            )
        ]

        # Compare all three radar configurations.
        for rays in ray_options:
            radar_data = get_radar(env, rays)

            detected = np.any(
                radar_data < empty_readings[rays] - 1e-5
            )

            if detected:
                detected_counts[rays] += 1
            else:
                missed_directions[rays].append(degrees)

    # Print results.
    print("\nRESULTS")
    print("-" * 60)

    for rays in ray_options:
        detected = detected_counts[rays]
        missed = len(missed_directions[rays])

        percentage = detected / 360 * 100

        print(
            f"{rays} rays: "
            f"detected {detected}/360 "
            f"({percentage:.1f}%), "
            f"missed {missed}"
        )

    print("\nDETECTION IMPROVEMENT")
    print("-" * 60)

    improvement_32 = (
        detected_counts[32] - detected_counts[16]
    )

    improvement_72 = (
        detected_counts[72] - detected_counts[32]
    )

    print(
        f"32 vs 16 rays: "
        f"+{improvement_32} detected directions"
    )

    print(
        f"72 vs 32 rays: "
        f"+{improvement_72} detected directions"
    )

    # More rays should not reduce detection coverage
    # in this nested-ray configuration.
    assert (
        detected_counts[72]
        >= detected_counts[32]
        >= detected_counts[16]
    ), "Unexpected radar detection results."

    print("\nALL RADAR COMPARISON TESTS PASSED!")

    env.close()


if __name__ == "__main__":
    test_radar()
