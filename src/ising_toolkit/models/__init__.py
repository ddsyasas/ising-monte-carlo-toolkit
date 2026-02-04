"""Ising model implementations."""

from ising_toolkit.models.base import IsingModel
from ising_toolkit.models.ising1d import Ising1D
from ising_toolkit.models.ising2d import Ising2D
from ising_toolkit.models.ising3d import Ising3D

__all__ = ["IsingModel", "Ising1D", "Ising2D", "Ising3D"]
