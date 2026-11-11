import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from galactica.environment.spaceship import SpaceshipState
from galactica.environment.obstacles import generate_asteroid_field
from galactica.environment.physics import apply_spaceship_action
from galactica.environment.collisions import check_path_collision
from galactica.environment.sensors import scan_radar


class SpaceEnv(gym.Env):
    """2D space navigation environment."""

    def __init__(self, asteroid_count: int = 20):
        super().__init__()

        if asteroid_count < 0:
            raise ValueError("Asteroid count cannot be negative.")

        self.asteroid_count = asteroid_count

        self.field_width = 100.0
        self.field_height = 100.0
        self.max_steps = 500

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
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
            fuel=100.0,
        )

        self.goal_x = 90.0
        self.goal_y = 50.0
        self.goal_radius = 5.0

        # Validate initial conditions.
        self._validate_start_position()
        self._validate_field_boundaries()

        # Generate asteroids only when requested.
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

    def _calculate_progress_reward(
        self,
        previous_distance: float,
        current_distance: float,
    ) -> float:
        """Calculate reward based on progress toward the goal."""
        progress = previous_distance - current_distance
        return float(progress * 0.01)

    def step(self, action):
        self.current_step += 1

        # Save the spaceship position before movement.
        previous_spaceship = self.spaceship

        previous_distance = math.hypot(
            self.goal_x - previous_spaceship.x,
            self.goal_y - previous_spaceship.y,
        )

        steering = float(action[0])

        # Convert thrust from [-1, 1] to [0, 1].
        thrust = (float(action[1]) + 1.0) / 2.0

        # Apply steering, thrust, and movement.
        self.spaceship = apply_spaceship_action(
            spaceship=self.spaceship,
            steering=steering,
            thrust=thrust,
            dt=1.0,
        )

        # Check the entire movement path for asteroid collisions.
        # This prevents tunneling through stationary asteroids.
        collision = any(
            check_path_collision(
                previous_spaceship,
                self.spaceship,
                asteroid,
            )
            for asteroid in self.asteroids
        )

        # Check whether the spaceship body leaves the field.
        out_of_bounds = not self._is_spaceship_inside_field()

        # Calculate distance to the goal.
        distance_to_goal = math.hypot(
            self.goal_x - self.spaceship.x,
            self.goal_y - self.spaceship.y,
        )

        goal_reached = distance_to_goal <= self.goal_radius

        # Calculate reward.
        if collision or out_of_bounds:
            reward = -1.0
        elif goal_reached:
            reward = 1.0
        else:
            reward = self._calculate_progress_reward(
                previous_distance=previous_distance,
                current_distance=distance_to_goal,
            )

        # End the episode on collision, goal, or boundary exit.
        terminated = collision or goal_reached or out_of_bounds

        # End the episode when the time limit is reached.
        truncated = (
            self.current_step >= self.max_steps
            and not terminated
        )

        observation = self._get_observation()

        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "out_of_bounds": out_of_bounds,
            "distance_to_goal": distance_to_goal,
        }

        return observation, reward, terminated, truncated, info

    def _validate_start_position(self):
        """Ensure the spaceship and goal do not overlap."""
        start_distance = math.hypot(
            self.goal_x - self.spaceship.x,
            self.goal_y - self.spaceship.y,
        )

        min_start_distance = (
            self.spaceship.radius + self.goal_radius
        )

        if start_distance <= min_start_distance:
            raise ValueError(
                "Spaceship and goal safety zones must not overlap."
            )

    def _is_spaceship_inside_field(self):
        """Check whether the spaceship is fully inside the field."""
        ship = self.spaceship

        return (
            ship.radius <= ship.x <= self.field_width - ship.radius
            and ship.radius <= ship.y <= self.field_height - ship.radius
        )

    def _validate_field_boundaries(self):
        """Validate initial spaceship and goal positions."""
        if not self._is_spaceship_inside_field():
            raise ValueError(
                "Spaceship must be fully inside the field."
            )

        if not (
            self.goal_radius
            <= self.goal_x
            <= self.field_width - self.goal_radius
            and self.goal_radius
            <= self.goal_y
            <= self.field_height - self.goal_radius
        ):
            raise ValueError(
                "Goal must be fully inside the field."
            )

    def _get_observation(self):
        """Build the observation vector for the agent."""
        spaceship_data = np.array(
            [
                self.spaceship.x,
                self.spaceship.y,
                self.spaceship.vx,
                self.spaceship.vy,
                self.spaceship.angle,
                self.spaceship.fuel,
                self.spaceship.radius,
            ],
            dtype=np.float32,
        )

        radar_data = scan_radar(
            spaceship=self.spaceship,
            asteroids=self.asteroids,
            num_rays=16,
            max_distance=50.0,
            field_width=self.field_width,
            field_height=self.field_height,
        )

        goal_data = np.array(
            [
                self.goal_x - self.spaceship.x,
                self.goal_y - self.spaceship.y,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [spaceship_data, radar_data, goal_data]
        )
