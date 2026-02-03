"""Pytest configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def small_lattice_size():
    """Provide a small lattice size for quick tests."""
    return 8


@pytest.fixture
def rng(random_seed):
    """Provide a seeded random number generator."""
    return np.random.default_rng(random_seed)
