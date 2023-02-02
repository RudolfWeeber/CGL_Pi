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