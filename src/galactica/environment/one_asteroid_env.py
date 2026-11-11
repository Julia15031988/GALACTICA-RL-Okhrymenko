from galactica.environment.space_env import SpaceEnv
from galactica.environment.asteroid import AsteroidState


class OneAsteroidEnv(SpaceEnv):
    """Environment with one fixed asteroid on the route."""

    def __init__(self):
        super().__init__(asteroid_count=0)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(
            seed=seed,
            options=options,
        )

        # Place one stationary asteroid on the direct route.
        self.asteroids = [
            AsteroidState(
                x=50.0,
                y=50.0,
                radius=5.0,
                vx=0.0,
                vy=0.0,
            )
        ]

        # Recalculate radar readings after adding asteroid.
        observation = self._get_observation()

        return observation, info
