"""Ising model implementations."""

from typing import Dict, Type

from .base import IsingModel
from .ising1d import Ising1D
from .ising2d import Ising2D
from .ising3d import Ising3D

__all__ = ["IsingModel", "Ising1D", "Ising2D", "Ising3D", "create_model"]

# Registry of available model types
_MODEL_REGISTRY: Dict[str, Type[IsingModel]] = {
    "ising1d": Ising1D,
    "ising2d": Ising2D,
    "ising3d": Ising3D,
    "1d": Ising1D,
    "2d": Ising2D,
    "3d": Ising3D,
}


def create_model(model_type: str, **kwargs) -> IsingModel:
    """Factory function to create Ising models.

    Parameters
    ----------
    model_type : str
        Type of model to create. Valid options:
        - 'ising1d' or '1d': 1D chain
        - 'ising2d' or '2d': 2D square lattice
        - 'ising3d' or '3d': 3D cubic lattice
    **kwargs
        Arguments passed to the model constructor:
        - size: int (required) - Linear size of the lattice
        - temperature: float (required) - Temperature in units of J/kB
        - coupling: float (optional) - Coupling constant J (default: 1.0)
        - boundary: str (optional) - 'periodic' or 'fixed' (default: 'periodic')

    Returns
    -------
    IsingModel
        Instance of the requested model type.

    Raises
    ------
    ValueError
        If model_type is not recognized.

    Examples
    --------
    >>> model = create_model('2d', size=32, temperature=2.269)
    >>> model = create_model('ising3d', size=16, temperature=4.511, boundary='fixed')
    """
    model_type_lower = model_type.lower()

    if model_type_lower not in _MODEL_REGISTRY:
        valid_types = sorted(set(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Valid options are: {valid_types}"
        )

    model_class = _MODEL_REGISTRY[model_type_lower]
    return model_class(**kwargs)
