"""Test the pi module."""
import pytest

import pi


def test_PiEstimator():
    """Test abstract base class PiEstimator."""
    with pytest.raises(TypeError):
        pi.PiEstimator()
