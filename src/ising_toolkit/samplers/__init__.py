"""Monte Carlo sampling algorithms."""

from typing import Dict, Optional, Type

from .base import Sampler
from .metropolis import MetropolisSampler
from .wolff import WolffSampler
from ising_toolkit.models.base import IsingModel

__all__ = ["Sampler", "MetropolisSampler", "WolffSampler", "create_sampler"]

# Registry of available sampler types
_SAMPLER_REGISTRY: Dict[str, Type[Sampler]] = {
    "metropolis": MetropolisSampler,
    "wolff": WolffSampler,
}


def create_sampler(
    algorithm: str,
    model: IsingModel,
    seed: Optional[int] = None,
) -> Sampler:
    """Factory function to create samplers.

    Parameters
    ----------
    algorithm : str
        Sampling algorithm to use. Valid options:
        - 'metropolis': Single-spin flip Metropolis algorithm
        - 'wolff': Wolff cluster algorithm (2D/3D only)
    model : IsingModel
        The Ising model to sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Sampler
        Instance of the requested sampler type.

    Raises
    ------
    ValueError
        If algorithm is not recognized.
    ConfigurationError
        If algorithm is incompatible with model (e.g., Wolff with 1D).

    Examples
    --------
    >>> from ising_toolkit.models import Ising2D
    >>> model = Ising2D(size=32, temperature=2.269)
    >>> sampler = create_sampler('metropolis', model, seed=42)
    >>> results = sampler.run(n_steps=10000)

    >>> # Wolff is more efficient near Tc
    >>> sampler = create_sampler('wolff', model, seed=42)
    >>> results = sampler.run(n_steps=10000)
    """
    algorithm_lower = algorithm.lower()

    if algorithm_lower not in _SAMPLER_REGISTRY:
        valid_algorithms = sorted(_SAMPLER_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Valid options are: {valid_algorithms}"
        )

    sampler_class = _SAMPLER_REGISTRY[algorithm_lower]
    return sampler_class(model, seed=seed)
