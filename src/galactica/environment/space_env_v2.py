import math
from copy import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from galactica.environment.spaceship import SpaceshipState
from galactica.environment.obstacles import generate_asteroid_field
from galactica.environment.collisions import check_path_collision
from galactica.environment.sensors import scan_radar
from galactica.environment.maneuvering_physics import (
    apply_maneuvering_action,
)


class SpaceEnvV2(gym.Env):
    """
    GALACTICA-RL V2.2.

    2D space navigation environment with:
    - Main and reverse thrusters
    - Lateral maneuvering thrusters
    - Inertia-based movement
    - 72-ray radar
    - Normalized/scaled observations
    - Three continuous control actions
    - Controlled-speed goal arrival
    - Fuel and time efficiency rewards
    """

    metadata = {"render_modes": []}

    def __init__(self, asteroid_count: int = 20):
        super().__init__()

        if asteroid_count < 0:
            raise ValueError(
                "Asteroid count cannot be negative."
            )

        self.asteroid_count = asteroid_count

        # Environment dimensions
        self.field_width = 100.0
        self.field_height = 100.0

        self.max_steps = 500
        self.max_fuel = 100.0

        # Radar configuration
        self.num_rays = 72
        self.radar_max_distance = 50.0

        # Observation scaling
        self.velocity_scale = 10.0

        # Goal arrival configuration
        self.goal_speed_threshold = 1.0

        # Reward configuration
        self.progress_reward_scale = 0.01
        self.fuel_penalty_scale = 0.001
        self.time_penalty = 0.001

        self.goal_success_reward = 2.0
        self.failure_penalty = -1.0

        # Encourage braking near the goal.
        self.braking_reward_scale = 0.005
        self.braking_zone_radius = 25.0

        # Observation:
        # 7 spaceship parameters
        # 72 radar readings
        # 2 relative goal coordinates
        #
        # Total: 81

        observation_low = np.array(
            [
                0.0,       # x
                0.0,       # y
                -np.inf,   # vx
                -np.inf,   # vy
                -1.0,      # angle
                0.0,       # fuel
                0.0,       # radius
            ]
            + [0.0] * self.num_rays
            + [
                -1.0,      # goal dx
                -1.0,      # goal dy
            ],
            dtype=np.float32,
        )

        observation_high = np.array(
            [
                1.0,       # x
                1.0,       # y
                np.inf,    # vx
                np.inf,    # vy
                1.0,       # angle
                1.0,       # fuel
                1.0,       # radius
            ]
            + [1.0] * self.num_rays
            + [
                1.0,       # goal dx
                1.0,       # goal dy
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )

        # Three continuous actions:
        #
        # action[0] = steering
        # action[1] = longitudinal thrust
        # action[2] = lateral thrust

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        self.spaceship = SpaceshipState(
            x=10.0,
            y=50.0,
            radius=3.0,
            vx=0.0,
            vy=0.0,
            angle=0.0,
            fuel=self.max_fuel,
        )

        # Goal configuration
        self.goal_x = 90.0
        self.goal_y = 50.0
        self.goal_radius = 5.0

        self._validate_start_position()
        self._validate_field_boundaries()

        # Generate asteroid field
        if self.asteroid_count == 0:
            self.asteroids = []
        else:
            self.asteroids = generate_asteroid_field(
                rng=self.np_random,
                spaceship=self.spaceship,
                count=self.asteroid_count,
                width=self.field_width,
                height=self.field_height,
                min_radius=2.0,
                max_radius=5.0,
                safety_margin=5.0,
                goal_x=self.goal_x,
                goal_y=self.goal_y,
                goal_radius=self.goal_radius,
            )

        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # Validate action shape and values
        if (
            action.shape != (3,)
            or not np.all(np.isfinite(action))
        ):
            raise ValueError(
                "Action must contain three finite values."
            )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        self.current_step += 1

        # Save previous state because physics mutates
        # the spaceship object in place.
        previous_spaceship = copy(self.spaceship)

        previous_distance = math.hypot(
            self.goal_x - previous_spaceship.x,
            self.goal_y - previous_spaceship.y,
        )

        previous_speed = math.hypot(
            previous_spaceship.vx,
            previous_spaceship.vy,
        )

        previous_fuel = previous_spaceship.fuel

        # ---------------------------------
        # Decode three control actions
        # ---------------------------------

        steering = float(action[0])

        # Positive = forward
        # Negative = reverse
        # Zero = no longitudinal thrust
        longitudinal_thrust = float(action[1])

        main_thrust = max(
            0.0,
            longitudinal_thrust,
        )

        reverse_thrust = max(
            0.0,
            -longitudinal_thrust,
        )

        # Signed lateral thrust
        lateral_thrust = float(action[2])

        # ---------------------------------
        # Apply maneuvering physics
        # ---------------------------------

        apply_maneuvering_action(
            spaceship=self.spaceship,
            steering=steering,
            main_thrust=main_thrust,
            reverse_thrust=reverse_thrust,
            lateral_thrust=lateral_thrust,
            dt=1.0,
        )

        # ---------------------------------
        # Collision detection
        # ---------------------------------

        collision = any(
            check_path_collision(
                previous_spaceship,
                self.spaceship,
                asteroid,
            )
            for asteroid in self.asteroids
        )

        out_of_bounds = (
            not self._is_spaceship_inside_field()
        )

        # ---------------------------------
        # Distance and speed
        # ---------------------------------

        distance_to_goal = math.hypot(
            self.goal_x - self.spaceship.x,
            self.goal_y - self.spaceship.y,
        )

        current_speed = math.hypot(
            self.spaceship.vx,
            self.spaceship.vy,
        )

        # Reaching the goal requires BOTH:
        # 1. Being inside the goal radius
        # 2. Moving slowly enough
        goal_reached = bool(
            distance_to_goal <= self.goal_radius
            and current_speed <= self.goal_speed_threshold
        )

        # ---------------------------------
        # Reward calculation
        # ---------------------------------

        progress = (
            previous_distance - distance_to_goal
        )

        fuel_used = max(
            0.0,
            previous_fuel - self.spaceship.fuel,
        )

        reward = (
            progress * self.progress_reward_scale
            - fuel_used * self.fuel_penalty_scale
            - self.time_penalty
        )

        # Braking reward near the goal.
        #
        # Reward a reduction in speed only when
        # the spaceship is close to the target.
        if distance_to_goal <= self.braking_zone_radius:
            speed_reduction = max(
                0.0,
                previous_speed - current_speed,
            )

            reward += (
                speed_reduction
                * self.braking_reward_scale
            )

        # Terminal rewards take priority.
        if collision or out_of_bounds:
            reward = self.failure_penalty

        elif goal_reached:
            reward += self.goal_success_reward

        # ---------------------------------
        # Episode termination
        # ---------------------------------

        terminated = bool(
            collision
            or goal_reached
            or out_of_bounds
        )

        truncated = bool(
            self.current_step >= self.max_steps
            and not terminated
        )

        # ---------------------------------
        # Diagnostic information
        # ---------------------------------

        info = {
            "collision": bool(collision),
            "goal_reached": bool(goal_reached),
            "out_of_bounds": bool(out_of_bounds),
            "distance_to_goal": float(
                distance_to_goal
            ),
            "speed": float(current_speed),
            "fuel_used": float(fuel_used),
        }

        observation = self._get_observation()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def _get_observation(self):
        """
        Build an 81-dimensional observation vector.

        Layout:
        0-6   : spaceship state
        7-78  : 72 radar readings
        79-80 : relative goal position
        """

        ship = self.spaceship

        # Wrap angle into [-pi, pi)
        wrapped_angle = (
            (ship.angle + math.pi)
            % (2.0 * math.pi)
        ) - math.pi

        # ---------------------------------
        # Normalized spaceship state
        # ---------------------------------

        spaceship_data = np.array(
            [
                ship.x / self.field_width,
                ship.y / self.field_height,

                ship.vx / self.velocity_scale,
                ship.vy / self.velocity_scale,

                wrapped_angle / math.pi,

                ship.fuel / self.max_fuel,

                ship.radius / self.field_width,
            ],
            dtype=np.float32,
        )

        # ---------------------------------
        # 72-ray radar
        # ---------------------------------

        radar_data = np.asarray(
            scan_radar(
                spaceship=ship,
                asteroids=self.asteroids,
                num_rays=self.num_rays,
                max_distance=self.radar_max_distance,
                field_width=self.field_width,
                field_height=self.field_height,
            ),
            dtype=np.float32,
        )

        # ---------------------------------
        # Relative goal position
        # ---------------------------------

        goal_data = np.array(
            [
                (
                    self.goal_x - ship.x
                ) / self.field_width,

                (
                    self.goal_y - ship.y
                ) / self.field_height,
            ],
            dtype=np.float32,
        )

        observation = np.concatenate(
            [
                spaceship_data,
                radar_data,
                goal_data,
            ]
        ).astype(np.float32)

        return observation

    def _validate_start_position(self):
        """Ensure spaceship and goal do not overlap."""

        distance = math.hypot(
            self.goal_x - self.spaceship.x,
            self.goal_y - self.spaceship.y,
        )

        minimum_distance = (
            self.spaceship.radius
            + self.goal_radius
        )

        if distance <= minimum_distance:
            raise ValueError(
                "Spaceship and goal safety zones "
                "must not overlap."
            )

    def _is_spaceship_inside_field(self):
        """Check whether the entire spaceship is inside."""

        ship = self.spaceship

        return (
            ship.radius
            <= ship.x
            <= self.field_width - ship.radius

            and ship.radius
            <= ship.y
            <= self.field_height - ship.radius
        )

    def _validate_field_boundaries(self):
        """Validate initial spaceship and goal positions."""

        if not self._is_spaceship_inside_field():
            raise ValueError(
                "Spaceship must be fully inside the field."
            )

        goal_inside = (
            self.goal_radius
            <= self.goal_x
            <= self.field_width - self.goal_radius

            and self.goal_radius
            <= self.goal_y
            <= self.field_height - self.goal_radius
        )

        if not goal_inside:
            raise ValueError(
                "Goal must be fully inside the field."
            )
