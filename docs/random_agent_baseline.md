# GALACTICA-RL — Random Agent Baseline Experiment

## Objective

Establish a reproducible baseline for evaluating autonomous spacecraft navigation in a two-dimensional asteroid field.

## Experimental Setup

- Environment: GALACTICA-RL SpaceEnv
- Agent: Random Agent
- Episodes: 100
- Seeds: 0–99
- Maximum episode length: 500 steps
- Reward: +1 for reaching the goal, −1 for collision or leaving the environment, 0 otherwise
- Action space: continuous steering and thrust
- Observation space: 25-dimensional vector including spacecraft state, radar measurements, and goal-relative position

## Results

| Metric | Value |
|---|---:|
| Episodes | 100 |
| Success rate | 0.00% |
| Collision rate | 87.00% |
| Average episode length | 11.68 steps |
| Average reward | −1.000 |
| Average fuel consumption | 5.66 units |

## Preliminary Interpretation

The Random Agent failed to reach the target in all 100 evaluated episodes. Collisions accounted for 87% of episode terminations.

The results indicate that unguided random control is insufficient for reliable navigation in the tested asteroid-field configuration.

These findings establish an initial performance reference for subsequent comparisons with heuristic navigation and Proximal Policy Optimization (PPO).

## Limitations

This experiment evaluates a single random-control policy over 100 seeded episodes. The remaining 13% of unsuccessful episodes have not yet been classified by termination cause in the aggregate report.

No statistical confidence intervals have been calculated. Comparisons with trained agents have not yet been performed.

## Next Steps

1. Record detailed termination reasons and per-episode results.
2. Implement and evaluate a radar-aware heuristic agent.
3. Train a PPO agent.
4. Compare all agents on the same held-out evaluation seeds using consistent metrics.
