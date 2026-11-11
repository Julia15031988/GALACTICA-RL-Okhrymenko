
import math

import numpy as np


class VelocityController:
    """
    Converts desired world-frame velocity into
    spaceship engine commands.

    Input:
        desired_vx, desired_vy

    Output:
        [steering, longitudinal_thrust, lateral_thrust]
    """

    def __init__(
        self,
        kp: float = 0.5,
        max_desired_speed: float = 5.0,
    ):
        if kp <= 0:
            raise ValueError("kp must be positive.")

        if max_desired_speed <= 0:
            raise ValueError(
                "max_desired_speed must be positive."
            )

        self.kp = kp
        self.max_desired_speed = max_desired_speed

    def calculate_action(
        self,
        spaceship,
        desired_vx: float,
        desired_vy: float,
    ) -> np.ndarray:
        """
        Calculate three engine commands.

        steering:
            [-1, 1]

        longitudinal_thrust:
            -1 = reverse
             0 = no thrust
             1 = forward

        lateral_thrust:
            [-1, 1]
        """

        desired_velocity = np.array(
            [desired_vx, desired_vy],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(desired_velocity)):
            raise ValueError(
                "Desired velocity must be finite."
            )

        # ---------------------------------
        # 1. Limit requested speed
        # ---------------------------------

        desired_speed = float(
            np.linalg.norm(desired_velocity)
        )

        if desired_speed > self.max_desired_speed:
            desired_velocity *= (
                self.max_desired_speed / desired_speed
            )

        # ---------------------------------
        # 2. Calculate velocity error
        # ---------------------------------

        error_x = (
            desired_velocity[0] - spaceship.vx
        )

        error_y = (
            desired_velocity[1] - spaceship.vy
        )

        # Proportional velocity control
        acceleration_x = self.kp * error_x
        acceleration_y = self.kp * error_y

        # ---------------------------------
        # 3. Convert world acceleration
        #    into spaceship-local axes
        # ---------------------------------

        angle = spaceship.angle

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # Forward/backward acceleration
        forward_acceleration = (
            acceleration_x * cos_angle
            + acceleration_y * sin_angle
        )

        # Left/right acceleration
        lateral_acceleration = (
            -acceleration_x * sin_angle
            + acceleration_y * cos_angle
        )

        # ---------------------------------
        # 4. Convert acceleration requests
        #    into engine commands
        # ---------------------------------

        # Physics acceleration:
        # longitudinal = 1.0
        # lateral = 0.6

        longitudinal_thrust = float(
            np.clip(
                forward_acceleration / 1.0,
                -1.0,
                1.0,
            )
        )

        lateral_thrust = float(
            np.clip(
                lateral_acceleration / 0.6,
                -1.0,
                1.0,
            )
        )

        # Initial controller uses fixed orientation.
        # It can maneuver using forward, reverse,
        # and lateral thrusters.
        steering = 0.0

        return np.array(
            [
                steering,
                longitudinal_thrust,
                lateral_thrust,
            ],
            dtype=np.float32,
        )
