"""Test the pi module."""
import subprocess

import numpy as np
import pytest

import pi


def test_PiEstimator():
    """Test abstract base class PiEstimator."""
    with pytest.raises(TypeError):
        pi.PiEstimator()
