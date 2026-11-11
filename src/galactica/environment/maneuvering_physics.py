import math


def apply_maneuvering_action(
    spaceship,
    steering: float,
    main_thrust: float,
    reverse_thrust: float,
    lateral_thrust: float,
    dt: float = 1.0,
):
    """
    Apply independent forward, reverse and lateral thrusters.

    The ship keeps its velocity when engines are off.
    No atmospheric drag is applied.
    """

    steering = max(-1.0, min(1.0, steering))
    main_thrust = max(0.0, min(1.0, main_thrust))
    reverse_thrust = max(0.0, min(1.0, reverse_thrust))
    lateral_thrust = max(-1.0, min(1.0, lateral_thrust))

    turn_rate = 1.0
    main_acceleration = 1.0
    reverse_acceleration = 1.0
    lateral_acceleration = 0.6

    spaceship.angle += steering * turn_rate * dt

    forward_x = math.cos(spaceship.angle)
    forward_y = math.sin(spaceship.angle)

    side_x = -math.sin(spaceship.angle)
    side_y = math.cos(spaceship.angle)

    forward_force = (
        main_thrust * main_acceleration
        - reverse_thrust * reverse_acceleration
    )

    side_force = lateral_thrust * lateral_acceleration

    acceleration_x = (
        forward_force * forward_x
        + side_force * side_x
    )

    acceleration_y = (
        forward_force * forward_y
        + side_force * side_y
    )

    fuel_cost = (
        main_thrust
        + reverse_thrust
        + abs(lateral_thrust)
    ) * dt

    if spaceship.fuel <= 0:
        acceleration_x = 0.0
        acceleration_y = 0.0
        fuel_cost = 0.0

    elif fuel_cost > spaceship.fuel:
        scale = spaceship.fuel / fuel_cost
        acceleration_x *= scale
        acceleration_y *= scale
        fuel_cost = spaceship.fuel

    spaceship.vx += acceleration_x * dt
    spaceship.vy += acceleration_y * dt

    spaceship.x += spaceship.vx * dt
    spaceship.y += spaceship.vy * dt

    spaceship.fuel = max(
        0.0,
        spaceship.fuel - fuel_cost,
    )
