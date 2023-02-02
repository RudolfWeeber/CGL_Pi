"""Estimate pi using various methods."""
import abc

import numpy as np

class PiEstimator(abc.ABC):
    """Abstract base class for pi estimators."""

    @abc.abstractmethod
    def estimate(self):
        """Estimate pi."""


class LeibnizPiEstimator(PiEstimator):


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