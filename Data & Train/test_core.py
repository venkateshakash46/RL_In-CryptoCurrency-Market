import numpy as np
import pandas as pd
from src.portfolio_rl import covariance_state, annualized_metrics, DQNAgent

def test_covariance_shape_and_finite():
    rng = np.random.default_rng(1)
    p = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, .01, (220, 4)), axis=0)))
    s = covariance_state(p, 180, 180)
    assert s.shape == (4, 4)
    assert np.isfinite(s).all()
    assert np.allclose(s, s.T)

def test_metrics_are_finite():
    m = annualized_metrics(np.array([.01, -.02, .005, .003]))
    assert all(np.isfinite(v) for v in m.values())

def test_weights_sum_to_one_and_long_only():
    agent = DQNAgent(4, seed=1)
    actions = np.array([1, 0, 1, 2])
    q = np.array([[0, 1, 0], [1, 0, 0], [0, 2, 0], [1, 0, 0]], dtype=float)
    w = agent.actions_to_weights(actions, q)
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)
