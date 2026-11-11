from galactica.environment.asteroid import AsteroidState
from galactica.environment.space_env import SpaceEnv


class RandomOneAsteroidEnv(SpaceEnv):
    """
    Space environment with one stationary asteroid.

    The asteroid changes its position on every reset.
    """

    def __init__(self):
        super().__init__(asteroid_count=0)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(
            seed=seed,
            options=options,
        )

        # Randomize the asteroid position.
        # Keep it away from the start and goal.
        while True:
            asteroid_x = float(
                self.np_random.uniform(35.0, 65.0)
            )
            asteroid_y = float(
                self.np_random.uniform(30.0, 70.0)
            )

            start_distance = (
                (asteroid_x - self.spaceship.x) ** 2
                + (asteroid_y - self.spaceship.y) ** 2
            ) ** 0.5

            goal_distance = (
                (asteroid_x - self.goal_x) ** 2
                + (asteroid_y - self.goal_y) ** 2
            ) ** 0.5

            if (
                start_distance > 15.0
                and goal_distance > 15.0
            ):
                break

        self.asteroids = [
            AsteroidState(
                x=asteroid_x,
                y=asteroid_y,
                radius=5.0,
                vx=0.0,
                vy=0.0,
            )
        ]

        # Recalculate the observation so the radar
        # sees the asteroid at its new position.
        observation = self._get_observation()

        return observation, info
