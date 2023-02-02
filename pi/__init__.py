"""Estimate pi using various methods."""
import abc

import numpy as np


class PiEstimator(abc.ABC):
    """Abstract base class for pi estimators."""

    @abc.abstractmethod
    def estimate(self):
        """Estimate pi."""
