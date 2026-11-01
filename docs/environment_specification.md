# GALACTICA-RL: Environment Specification

**Project:** Autonomous Space Navigation Through Deep Reinforcement Learning  
**Environment:** SpaceEnv  
**Version:** 0.1 — Initial 2D Environment  
**Status:** Initial implementation; validation and experimentation ongoing

## 1. Research Objective

GALACTICA-RL investigates autonomous spacecraft navigation through an asteroid field using reinforcement learning.

The primary research objective is to evaluate the performance of a Proximal Policy Optimization (PPO) agent against baseline navigation strategies under identical environmental conditions.

The evaluation will consider:

- Navigation success rate
- Collision rate
- Boundary violation rate
- Fuel consumption
- Number of steps required to reach the destination

No assumption is made that PPO will outperform the baseline agents.

## 2. Environment Geometry

The environment is a two-dimensional continuous space.

| Parameter | Value |
|---|---|
| Field width | 100 units |
| Field height | 100 units |
| Initial spacecraft position | (10, 50) |
| Spacecraft radius | 3 units |
| Goal position | (90, 50) |
| Goal radius | 5 units |
| Number of asteroids | 20 |
| Asteroid radius | 2–5 units |
| Maximum episode length | 500 steps |

The spacecraft, destination, and asteroids are represented by circular geometric objects.

## 3. Initial Conditions

At the beginning of each episode:

1. The spacecraft is initialized with zero velocity, zero orientation angle, and 100 units of fuel.
2. The destination is initialized at a fixed position.
3. The spacecraft and destination must not overlap.
4. Both objects must be fully contained within the environment boundaries.
5. Asteroids are generated using a seeded random number generator.
6. Asteroids must not overlap with the spacecraft, destination safety zone, or other asteroids.
7. An additional safety margin of 5 units is applied during asteroid placement.

The random seed supports reproducible environment initialization.

## 4. Spacecraft Dynamics

The spacecraft state includes position, velocity, orientation, radius, and remaining fuel.

The action space is continuous and consists of two components:

- **Steering:** controls spacecraft rotation.
- **Thrust:** controls forward propulsion.

Both action components are supplied in the range [-1, 1]. The thrust component is mapped to [0, 1] before being passed to the physics model.

The spacecraft can move in different directions through rotation and inertia. Dedicated reverse thrust is not implemented in this version.

The physics model uses a simulation time step of `dt = 1.0`.

## 5. Observation Space

The observation vector contains **25 numerical features**.

| Component | Features |
|---|---:|
| Spacecraft state | 7 |
| Radar readings | 16 |
| Relative destination coordinates | 2 |
| **Total** | **25** |

The spacecraft state includes position, velocity, orientation, remaining fuel, and radius.

Destination coordinates are represented relative to the spacecraft.

## 6. Radar Model — Version 1

The spacecraft is equipped with a 360-degree radar consisting of 16 uniformly distributed rays.

Each ray detects the nearest obstacle along its direction, considering:

- Asteroid surfaces
- Environment boundaries

Radar readings account for spacecraft radius using a body-relative approximation.

The maximum radar detection distance is 50 units.

Each radar reading is normalized to the interval [0, 1]:

- `0.0`: obstacle touching or intersecting the spacecraft body along the ray
- `1.0`: no obstacle detected within the radar range

Intermediate values represent normalized free distances.

The radar is orientation-relative: its rays rotate together with the spacecraft.

## 7. Collision and Boundary Conditions

An asteroid collision is detected using the distance between the centers of the spacecraft and asteroid, compared with the sum of their radii.

A boundary violation occurs when any part of the spacecraft body extends outside the rectangular environment.

Both conditions terminate the episode.

Boundary violations are tracked separately from asteroid collisions.

## 8. Reward Function

The initial reward model is sparse.

| Event | Reward |
|---|---:|
| Asteroid collision | -1.0 |
| Boundary violation | -1.0 |
| Destination reached without a simultaneous failure | +1.0 |
| Ordinary navigation step | 0.0 |

Collision and boundary penalties take priority over the destination reward.

Distance-based reward shaping is intentionally excluded from the initial implementation and may be evaluated as a separate research experiment.

## 9. Episode Termination

An episode terminates when:

- The spacecraft collides with an asteroid.
- The spacecraft reaches the destination.
- The spacecraft violates an environment boundary.

An episode is truncated when the maximum number of 500 steps is reached without a terminating event.

The environment returns the standard Gymnasium tuple:

`observation, reward, terminated, truncated, info`

The `info` dictionary includes:

- `collision`
- `goal_reached`
- `out_of_bounds`
- `distance_to_goal`

## 10. Testing and Reproducibility

The environment is tested using pytest.

The current test suite covers spacecraft behavior, asteroid generation, radar measurements, initial-position validation, episode termination, and selected boundary conditions.

Seeded initialization is used to support reproducible experiments.

Passing unit tests does not establish that every physically possible edge case has been validated.

## 11. Known Limitations and Future Work

The following topics require further investigation:

- Validation using Gymnasium and Stable-Baselines3 environment checkers
- High-speed motion and continuous collision detection
- Consistency of radar measurements near and outside field boundaries
- Reachability of the destination in randomly generated asteroid fields
- Fuel depletion behavior and its effect on episode outcomes
- Observation normalization and learning stability
- Baseline-agent implementation
- PPO training and controlled evaluation
- Dynamic obstacles and more complex navigation scenarios

## 12. Experimental Principles

PPO and baseline agents will be evaluated under comparable environment configurations.

Experimental comparisons should use consistent initial conditions, random seeds, reward definitions, and evaluation metrics.

Results will be reported without assuming superiority of either approach.

---

**Document purpose:** Define the initial environment model and provide a reproducible foundation for future reinforcement learning experiments.
