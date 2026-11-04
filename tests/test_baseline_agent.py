import numpy as np

from galactica.agents.baseline_agent import RandomAgent


def test_random_agent_action():
    agent = RandomAgent(seed=42)

    observation = np.zeros(25, dtype=np.float32)

    action = agent.predict(observation)

    assert action.shape == (2,)
    assert action.dtype == np.float32
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

def test_random_agent_reproducibility():
    agent_1 = RandomAgent(seed=42)
    agent_2 = RandomAgent(seed=42)

    observation = np.zeros(25, dtype=np.float32)

    for _ in range(10):
        action_1 = agent_1.predict(observation)
        action_2 = agent_2.predict(observation)

        np.testing.assert_array_equal(action_1, action_2)
