
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from galactica.environment.spaceship import SpaceshipState
from galactica.environment.obstacles import generate_asteroid_field
from galactica.environment.physics import apply_spaceship_action
from galactica.environment.collisions import check_collision
from galactica.environment.sensors import scan_radar


class SpaceEnv(gym.Env):
    """2D space navigation environment."""

    def __init__(self):
        super().__init__()

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

        self.asteroids = generate_asteroid_field(
            rng=self.np_random,
            spaceship=self.spaceship,
            count=20,
            width=100.0,
            height=100.0,
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
        self.current_step += 1

        steering = float(action[0])

        # Convert thrust from [-1, 1] to [0, 1]
        thrust = (float(action[1]) + 1.0) / 2.0

        self.spaceship = apply_spaceship_action(
            spaceship=self.spaceship,
            steering=steering,
            thrust=thrust,
            dt=1.0,
        )

        collision = any(
            check_collision(self.spaceship, asteroid)
            for asteroid in self.asteroids
        )

        distance_to_goal = math.hypot(
            self.goal_x - self.spaceship.x,
            self.goal_y - self.spaceship.y,
        )

        goal_reached = distance_to_goal <= self.goal_radius

        observation = self._get_observation()

        if collision:
            reward = -1.0
        elif goal_reached:
            reward = 1.0
        else:
            reward = 0.0

        terminated = collision or goal_reached

        truncated = (
            self.current_step >= self.max_steps
            and not terminated
        )

        info = {
            "collision": collision,
            "goal_reached": goal_reached,
            "distance_to_goal": distance_to_goal,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
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
