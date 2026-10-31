from dataclasses import replace
import math

from galactica.environment.spaceship import SpaceshipState
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

def move_spaceship(
    spaceship: SpaceshipState,
    ax: float,
    ay: float,
    dt: float,
) -> SpaceshipState:
    """Move spaceship using acceleration and time step."""

    if dt < 0:
        raise ValueError("Time step cannot be negative.")

    new_vx = spaceship.vx + ax * dt
    new_vy = spaceship.vy + ay * dt

    new_x = spaceship.x + new_vx * dt
    new_y = spaceship.y + new_vy * dt

    return replace(
        spaceship,
        x=new_x,
        y=new_y,
        vx=new_vx,
        vy=new_vy,
    )

def apply_spaceship_action(
    spaceship: SpaceshipState,
    steering: float,
    thrust: float,
    dt: float,
    max_turn_rate: float = 1.0,
    max_acceleration: float = 1.0,
    fuel_consumption: float = 1.0,
) -> SpaceshipState:
    """Apply steering and engine thrust to the spaceship."""

    if dt < 0:
        raise ValueError("Time step cannot be negative.")

    if not -1.0 <= steering <= 1.0:
        raise ValueError("Steering must be between -1 and 1.")

    if not 0.0 <= thrust <= 1.0:
        raise ValueError("Thrust must be between 0 and 1.")

    if max_turn_rate < 0 or max_acceleration < 0 or fuel_consumption < 0:
        raise ValueError("Physical parameters cannot be negative.")

    new_angle = spaceship.angle + steering * max_turn_rate * dt

    requested_fuel = thrust * fuel_consumption * dt
    actual_fuel = min(spaceship.fuel, requested_fuel)

    effective_thrust = (
        thrust * actual_fuel / requested_fuel
        if requested_fuel > 0
        else 0.0
    )

    ax = math.cos(new_angle) * effective_thrust * max_acceleration
    ay = math.sin(new_angle) * effective_thrust * max_acceleration

    rotated_ship = replace(
        spaceship,
        angle=new_angle,
        fuel=spaceship.fuel - actual_fuel,
    )

    return move_spaceship(
        spaceship=rotated_ship,
        ax=ax,
        ay=ay,
        dt=dt,
    )

