"""Test the pi module."""
import pytest

import numpy as np
import pi


def test_PiEstimator():
    """Test abstract base class PiEstimator."""
    with pytest.raises(TypeError):
        pi.PiEstimator()


@pytest.mark.parametrize("n", [100, 1000, 10000])
def test_LeibnizPiEstimator(n):
    """Test the Leibniz PiEstimator."""
    assert pi.LeibnizPiEstimator(n).estimate() == pytest.approx(np.pi, 0.01)

def test_MCPiEstimator():
    """Test the Monte Carlo PiEstimator."""
    assert pi.MonteCarloPiEstimator(1000000).estimate() == pytest.approx(np.pi, 0.01)

def test_MCPiEstimator_n_error():
    """Test the Monte Carlo PiEstimator with wrong input type."""
    with pytest.raises(TypeError):
        _ = pi.MonteCarloPiEstimator(10e5).estimate()
