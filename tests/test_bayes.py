import numpy as np
import pytest

from ordo.bayes import GammaPoissonBelief


def test_default_prior_is_gamma_1_1():
    belief = GammaPoissonBelief()
    assert belief.shape == 1.0
    assert belief.rate == 1.0


def test_custom_prior():
    belief = GammaPoissonBelief(shape=2.0, rate=3.0)
    assert belief.shape == 2.0
    assert belief.rate == 3.0


def test_observe_updates_shape_and_rate():
    belief = GammaPoissonBelief(shape=1.0, rate=1.0)

    belief.observe(2.0)

    assert belief.shape == 2.0
    assert belief.rate == 3.0


def test_observe_multiple_delays_accumulates():
    belief = GammaPoissonBelief(shape=1.0, rate=1.0)

    belief.observe(2.0)
    belief.observe(0.5)
    belief.observe(1.5)

    assert belief.shape == 4.0
    assert belief.rate == 5.0


def test_mean_and_variance_of_prior():
    belief = GammaPoissonBelief(shape=2.0, rate=4.0)

    assert belief.mean == 0.5
    assert belief.variance == 0.125


def test_mean_and_variance_after_observations():
    belief = GammaPoissonBelief(shape=1.0, rate=1.0)

    belief.observe(2.0)
    belief.observe(0.5)

    # shape=3.0, rate=3.5
    assert belief.mean == 3.0 / 3.5
    assert belief.variance == 3.0 / (3.5 ** 2)


def test_belief_converges_to_true_rate():
    true_lambda = 4.0
    rng = np.random.default_rng(seed=42)
    delays = rng.exponential(scale=1.0 / true_lambda, size=5000)

    belief = GammaPoissonBelief()
    for delay in delays:
        belief.observe(float(delay))

    assert belief.mean == pytest.approx(true_lambda, rel=0.05)
