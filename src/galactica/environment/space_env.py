import gymnasium as gym
import numpy as np
from gymnasium import spaces
from galactica.environment.spaceship import SpaceshipState
from galactica.environment.obstacles import generate_asteroid_field


class SpaceEnv(gym.Env):
    """2D space navigation environment."""

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
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

        self.spaceship = SpaceshipState(
            x=10.0,
            y=50.0,
            radius=3.0,
            vx=0.0,
            vy=0.0,
            angle=0.0,
            fuel=100.0,
        )

        self.asteroids = generate_asteroid_field(
            rng=self.np_random,
            spaceship=self.spaceship,
            count=20,
            width=100.0,
            height=100.0,
            min_radius=2.0,
            max_radius=5.0,
            safety_margin=5.0,
        )

        observation = np.array(
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

        return observation, {}


