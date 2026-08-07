import numpy as np


class GammaExponentialBelief:
    """
    Online Bayesian belief about an event rate, using the
    Gamma-Exponential conjugate model updated from observed inter-event
    delays
    """

    def __init__(self, shape: float = 1.0, rate: float = 1.0) -> None:
        self.shape = shape # alpha
        self.rate = rate # beta

    def observe(self, delay: float) -> None:
        """Update the belief with one observed inter-event delay."""
        self.shape += 1
        self.rate += delay

    @property
    def mean(self) -> float:
        """Posterior mean of lambda; the expected value of the Gamma distribution"""
        return self.shape / self.rate

    @property
    def variance(self) -> float:
        """Posterior variance of lambda."""
        return self.shape / self.rate ** 2

    @property
    def expected_delay(self) -> float:
        """Posterior mean estimate of expected delay (seconds / job)."""
        return self.rate / (self.shape - 1.0) if self.shape > 1.0 else float("inf")

    def sample_rate(self, rng=None) -> float:
        """Thompson Sampling: Draw a plausible service rate lambda from posterior distribution.

        `rng`, if given, is a `numpy.random.Generator` (e.g. from
        `np.random.default_rng(seed)`) used for reproducible draws.
        Defaults to the global `numpy.random` module state.
        """
        generator = rng if rng is not None else np.random
        return generator.gamma(shape=self.shape, scale=1.0 / self.rate)
    