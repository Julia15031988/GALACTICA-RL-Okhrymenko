
import math

import numpy as np


class HeuristicAgent:
    """Rule-based agent with obstacle avoidance and speed control."""

    def __init__(self):
        self.mode = "NAVIGATE"
        self.avoid_direction = 0.0

    def predict(self, observation: np.ndarray) -> np.ndarray:
        # Current spaceship orientation.
        angle = float(observation[4])

        # Relative position of the goal.
        goal_dx = float(observation[23])
        goal_dy = float(observation[24])

        # Direction toward the goal.
        target_angle = math.atan2(goal_dy, goal_dx)

        angle_difference = (
            target_angle - angle + math.pi
        ) % (2 * math.pi) - math.pi

        # Default steering: turn toward the goal.
        steering = float(
            np.clip(angle_difference, -1.0, 1.0)
        )

        # Radar observations.
        radar = observation[7:23]

        front_distance = float(radar[0])
        left_distance = float(radar[2])
        right_distance = float(radar[14])

        # Find the radar ray closest to the goal direction.
        relative_goal_angle = (
                                      target_angle - angle
                              ) % (2 * math.pi)

        ray_count = len(radar)
        ray_angle = 2 * math.pi / ray_count

        goal_ray_index = round(
            relative_goal_angle / ray_angle
        ) % ray_count

        goal_distance = float(radar[goal_ray_index])

        # Enter avoidance mode when an obstacle is ahead.
        if self.mode == "NAVIGATE" and front_distance < 0.4:
            if left_distance >= right_distance:
                self.avoid_direction = 1.0
            else:
                self.avoid_direction = -1.0

            self.mode = "AVOID"

        # Check whether the path toward the goal is clear.
        goal_path_clear = (
                goal_distance >= 0.9
                and float(radar[(goal_ray_index - 1) % ray_count]) >= 0.9
                and float(radar[(goal_ray_index + 1) % ray_count]) >= 0.9
        )

        # Check whether the spaceship is moving slowly enough
        # to safely reconsider its navigation direction.
        speed = math.hypot(
            float(observation[2]),
            float(observation[3]),
        )

        safe_to_resume = speed < 1.0

        # Leave avoidance mode when the goal corridor is clear.
        if (
                self.mode == "AVOID"
                and goal_path_clear
                and safe_to_resume
        ):
            self.mode = "NAVIGATE"
            self.avoid_direction = 0.0

        # Continue the avoidance maneuver while necessary.
        if self.mode == "AVOID":
            steering = self.avoid_direction

        # Current spaceship velocity.
        vx = float(observation[2])
        vy = float(observation[3])

        speed = math.hypot(vx, vy)

        # Direction opposite to current velocity.
        if speed > 0.0:
            braking_angle = math.atan2(-vy, -vx)
        else:
            braking_angle = angle

        # Brake at high speed, unless avoiding an obstacle.
        if speed >= 3.0 and self.mode != "AVOID":
            braking_difference = (
                braking_angle - angle + math.pi
            ) % (2 * math.pi) - math.pi

            steering = float(
                np.clip(braking_difference, -1.0, 1.0)
            )

        # Default: moderate thrust.
        # In SpaceEnv, action 0.0 means 50% thrust.
        thrust = 0.0

        # Active braking when speed is too high.
        if speed >= 3.0:
            thrust = -1.0

            braking_alignment = (
                braking_angle - angle + math.pi
            ) % (2 * math.pi) - math.pi

            if (
                self.mode != "AVOID"
                and abs(braking_alignment) < 0.2
            ):
                thrust = 1.0

        # Disable thrust when an obstacle is directly ahead.
        if front_distance < 0.4:
            thrust = -1.0

        return np.array(
            [steering, thrust],
            dtype=np.float32,
        )
