"""Estimate pi using various methods."""
import abc

class PiEstimator(abc.ABC):
    """Abstract base class for pi estimators."""

    @abc.abstractmethod
    def estimate(self):
        """Estimate pi."""
