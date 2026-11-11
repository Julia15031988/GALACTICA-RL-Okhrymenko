import math
import numpy as np

from galactica.agents.heuristic_agent import HeuristicAgent
from galactica.environment.space_env import SpaceEnv


def test_heuristic_agent_turns_toward_goal():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship is facing right (0 radians).
    observation[4] = 0.0

    # Goal is above the ship.
    observation[23] = 0.0
    observation[24] = 10.0

    # No nearby obstacles.
    observation[7:23] = 1.0

    action = agent.predict(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32

    # Turn counterclockwise toward the goal.
    assert action[0] > 0.0

    # Fixed thrust command.
    assert action[1] == 0.0


def test_heuristic_agent_turns_in_environment():
    env = SpaceEnv()
    agent = HeuristicAgent()

    env.reset(seed=42)

    # Place the goal above the starting spaceship.
    env.goal_x = 10.0
    env.goal_y = 90.0

    # Remove asteroids to isolate goal steering.
    env.asteroids = []

    observation = env._get_observation()

    initial_angle = env.spaceship.angle

    action = agent.predict(observation)

    env.step(action)

    # Ship must turn counterclockwise.
    assert env.spaceship.angle > initial_angle

    env.close()


def test_heuristic_agent_avoids_obstacle_left():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship faces right.
    observation[4] = 0.0

    # Goal is directly ahead.
    observation[23] = 80.0
    observation[24] = 0.0

    # No nearby obstacles by default.
    observation[7:23] = 1.0

    # Obstacle directly ahead.
    observation[7] = 0.2

    # Left side is safer than right.
    observation[9] = 0.9
    observation[21] = 0.3

    action = agent.predict(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32

    # Turn left.
    assert action[0] > 0.0


def test_heuristic_agent_avoids_obstacle_right():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship faces right.
    observation[4] = 0.0

    # Goal is directly ahead.
    observation[23] = 80.0
    observation[24] = 0.0

    # No nearby obstacles by default.
    observation[7:23] = 1.0

    # Obstacle directly ahead.
    observation[7] = 0.2

    # Right side is safer than left.
    observation[9] = 0.3
    observation[21] = 0.9

    action = agent.predict(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32

    # Turn right.
    assert action[0] < 0.0

def test_heuristic_agent_stops_accelerating_at_high_speed():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship is moving faster than our speed threshold.
    observation[2] = 3.5
    observation[3] = 0.0

    # Goal is directly ahead.
    observation[23] = 80.0
    observation[24] = 0.0

    # Clear radar.
    observation[7:23] = 1.0

    action = agent.predict(observation)

    # Engine should be turned off.
    assert action[1] == -1.0


def test_heuristic_agent_stops_accelerating_near_obstacle():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Goal is directly ahead.
    observation[23] = 80.0
    observation[24] = 0.0

    # Clear radar except for a forward obstacle.
    observation[7:23] = 1.0
    observation[7] = 0.2

    action = agent.predict(observation)

    # Engine should be turned off near the obstacle.
    assert action[1] == -1.0

def test_heuristic_agent_turns_to_brake_at_high_speed():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship faces upward (90 degrees).
    observation[4] = math.pi / 2

    # Ship moves rapidly to the right.
    observation[2] = 4.0
    observation[3] = 0.0

    # Goal is straight ahead.
    observation[23] = 0.0
    observation[24] = 80.0

    # No nearby obstacles.
    observation[7:23] = 1.0

    action = agent.predict(observation)

    # Turn left toward the braking direction.
    assert action[0] > 0.0

    # Do not accelerate until facing the braking direction.
    assert action[1] == -1.0

def test_heuristic_agent_applies_braking_thrust():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship faces left.
    observation[4] = math.pi

    # Ship moves rapidly to the right.
    observation[2] = 4.0
    observation[3] = 0.0

    # Goal is to the right.
    observation[23] = 80.0
    observation[24] = 0.0

    # No nearby obstacles.
    observation[7:23] = 1.0

    action = agent.predict(observation)

    # Ship is already facing opposite to its velocity.
    assert abs(action[0]) < 0.01

    # Apply thrust to reduce forward velocity.
    assert action[1] == 1.0


def test_heuristic_agent_reduces_speed_in_environment():
    env = SpaceEnv()
    agent = HeuristicAgent()

    env.reset(seed=42)

    # Remove obstacles to isolate braking physics.
    env.asteroids = []

    # Ship moves right but faces left.
    env.spaceship.x = 50.0
    env.spaceship.y = 50.0
    env.spaceship.vx = 4.0
    env.spaceship.vy = 0.0
    env.spaceship.angle = math.pi

    observation = env._get_observation()

    initial_speed = math.hypot(
        env.spaceship.vx,
        env.spaceship.vy,
    )

    action = agent.predict(observation)

    # Agent should activate braking thrust.
    assert action[1] == 1.0

    env.step(action)

    final_speed = math.hypot(
        env.spaceship.vx,
        env.spaceship.vy,
    )

    # Physical velocity must actually decrease.
    assert final_speed < initial_speed

    env.close()

def test_heuristic_agent_prioritizes_obstacle_avoidance():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)

    # Ship faces right.
    observation[4] = 0.0

    # Ship moves quickly to the right.
    observation[2] = 4.0
    observation[3] = 0.0

    # Goal is directly ahead.
    observation[23] = 80.0
    observation[24] = 0.0

    # Clear radar by default.
    observation[7:23] = 1.0

    # Obstacle directly ahead.
    observation[7] = 0.2

    # Left side is safer.
    observation[9] = 0.9
    observation[21] = 0.3

    action = agent.predict(observation)

    # Obstacle avoidance must take priority.
    assert action[0] > 0.0

    # Do not accelerate toward the obstacle.
    assert action[1] == -1.0


def test_heuristic_agent_remembers_avoidance():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)
    observation[23] = 80.0
    observation[7:23] = 1.0

    # First observation: obstacle ahead, right side is safer.
    observation[7] = 0.12
    observation[9] = 0.81
    observation[21] = 1.0

    first_action = agent.predict(observation)

    assert first_action[0] < 0.0

    # After turning, the front ray becomes clear.
    observation[4] = -1.0
    observation[7] = 1.0
    observation[9] = 0.10
    observation[21] = 0.88

    second_action = agent.predict(observation)

    # Agent should continue avoiding instead of
    # immediately steering back toward the goal.
    assert second_action[0] < 0.0

def test_heuristic_agent_exits_avoidance():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)
    observation[23] = 80.0
    observation[7:23] = 1.0

    # Obstacle ahead: enter avoidance mode.
    observation[7] = 0.1
    observation[9] = 0.8
    observation[21] = 1.0

    agent.predict(observation)

    assert agent.mode == "AVOID"

    # Simulate a clear path after the maneuver.
    observation[4] = 0.0
    observation[7:23] = 1.0

    agent.predict(observation)

    assert agent.mode == "NAVIGATE"

def test_goal_direction_matches_radar_ray():
    import math

    ship_angle = 0.0
    goal_dx = 10.0
    goal_dy = 10.0

    goal_angle = math.atan2(goal_dy, goal_dx)

    relative_angle = (
        goal_angle - ship_angle
    ) % (2 * math.pi)

    ray_index = round(
        relative_angle / (2 * math.pi / 16)
    ) % 16

    assert ray_index == 2

def test_heuristic_agent_exits_avoidance_when_goal_path_is_clear():
    agent = HeuristicAgent()

    observation = np.zeros(25, dtype=np.float32)
    observation[7:23] = 1.0

    # Goal is directly ahead.
    observation[23] = 80.0
    observation[24] = 0.0

    # First: obstacle ahead, right side is safer.
    observation[7] = 0.1
    observation[9] = 0.8
    observation[21] = 1.0

    agent.predict(observation)

    assert agent.mode == "AVOID"

    # After maneuvering, the path toward the goal is clear.
    # An obstacle remains behind the spaceship.
    observation[7:23] = 1.0
    observation[15] = 0.2

    # Spaceship is stationary and facing the goal.
    observation[2] = 0.0
    observation[3] = 0.0
    observation[4] = 0.0

    agent.predict(observation)

    assert agent.mode == "NAVIGATE"
