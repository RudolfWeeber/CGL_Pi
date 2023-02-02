"""Estimate pi using various methods."""
import abc

import numpy as np


class PiEstimator(abc.ABC):
    """Abstract base class for pi estimators."""

    @abc.abstractmethod
    def estimate(self):
        """Estimate pi."""


class LeibnizPiEstimator(PiEstimator):
    """Estimate pi using the Leibniz formula."""

    def __init__(self, n):
        """Initialize the Leibniz formula estimator.

        Parameters
        ----------
        n: int
            Number of points to use.
        """
        self.n = n

    def estimate(self):
        """Estimate pi using the Leibniz formula."""
        data = np.arange(self.n)
        pi = 4 * (-1) ** data / (2 * data + 1)
        return float(np.sum(pi))

class MonteCarloPiEstimator(PiEstimator):
    """Estimate pi using the Monte Carlo method."""

    def __init__(self, n):
        """Initialize the Monte Carlo estimator.

        Parameters
        ----------
        n: int
            Number of random points to use.
        """
        self.n = n

    def estimate(self):
        """Estimate pi using the Monte Carlo method."""
        data = np.random.uniform(-0.5, 0.5, size=(self.n, 2))
        inside = len(np.argwhere(np.linalg.norm(data, axis=1) < 0.5))
        return 4 * inside / self.n

