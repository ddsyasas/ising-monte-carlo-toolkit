"""1D Ising model implementation."""

from typing import Dict, Tuple

import numpy as np

from ising_toolkit.models.base import IsingModel
from ising_toolkit.utils import (
    validate_lattice_size,
    validate_initial_state,
    validate_temperature,
    DEFAULT_COUPLING,
)


class Ising1D(IsingModel):
    """1D Ising model on a chain.

    Implements the Ising model on a one-dimensional chain of spins
    with either periodic or fixed boundary conditions.

    The Hamiltonian is:
        H = -J * Σ_i s_i * s_{i+1}

    Parameters
    ----------
    size : int
        Number of spins in the chain. Must be >= 2.
    temperature : float
        Temperature in units of J/kB. Must be positive.
    coupling : float, optional
        Coupling constant J (default: 1.0). Positive for ferromagnetic.
    boundary : str, optional
        Boundary condition: 'periodic' or 'fixed' (default: 'periodic').

    Attributes
    ----------
    size : int
        Number of spins in the chain.
    acceptance_probs : dict
        Precomputed acceptance probabilities for Metropolis algorithm.

    Examples
    --------
    >>> model = Ising1D(size=100, temperature=2.0)
    >>> model.initialize('random')
    >>> energy = model.get_energy()
    >>> mag = model.get_magnetization()

    Notes
    -----
    The 1D Ising model has no phase transition at finite temperature
    (Tc = 0). It is exactly solvable and useful for testing and
    pedagogical purposes.
    """

    def __init__(
        self,
        size: int,
        temperature: float,
        coupling: float = DEFAULT_COUPLING,
        boundary: str = "periodic",
    ) -> None:
        """Initialize the 1D Ising model.

        Parameters
        ----------
        size : int
            Number of spins. Must be >= 2.
        temperature : float
            Temperature in units of J/kB. Must be positive.
        coupling : float, optional
            Coupling constant J (default: 1.0).
        boundary : str, optional
            Boundary condition: 'periodic' or 'fixed' (default: 'periodic').

        Raises
        ------
        ConfigurationError
            If parameters are invalid.
        """
        super().__init__(temperature, coupling, boundary)
        self._size = validate_lattice_size(size, min_size=2)
        self._spins = np.ones(self._size, dtype=np.int8)
        self._rng = np.random.default_rng()

        # Precompute acceptance probabilities
        self._acceptance_probs: Dict[int, float] = {}
        self._update_acceptance_probs()

    def _update_acceptance_probs(self) -> None:
        """Precompute Metropolis acceptance probabilities.

        For the 1D Ising model, the energy change from flipping a spin
        can only be -4J, -2J, 0, +2J, or +4J. We precompute exp(-β*ΔE)
        for positive ΔE values for efficiency.
        """
        self._acceptance_probs = {}
        for dE in [2, 4]:
            # Scale by coupling constant
            actual_dE = dE * self._coupling
            self._acceptance_probs[dE] = np.exp(-self.beta * actual_dE)

    def set_temperature(self, temperature: float) -> None:
        """Set a new temperature and update acceptance probabilities.

        Parameters
        ----------
        temperature : float
            New temperature value. Must be positive.

        Raises
        ------
        ConfigurationError
            If temperature is not positive.
        """
        self._temperature = validate_temperature(temperature)
        self._update_acceptance_probs()

    @property
    def size(self) -> int:
        """int: Number of spins in the chain."""
        return self._size

    @property
    def dimension(self) -> int:
        """int: Spatial dimension (1 for 1D chain)."""
        return 1

    @property
    def n_spins(self) -> int:
        """int: Total number of spins."""
        return self._size

    @property
    def n_neighbors(self) -> int:
        """int: Number of neighbors per spin (2 for periodic, varies for fixed)."""
        return 2 if self._boundary == "periodic" else 2

    @property
    def spins(self) -> np.ndarray:
        """np.ndarray: Current spin configuration."""
        return self._spins

    @property
    def acceptance_probs(self) -> Dict[int, float]:
        """dict: Precomputed acceptance probabilities for energy changes."""
        return self._acceptance_probs.copy()

    def initialize(self, state: str = "random") -> None:
        """Initialize the spin configuration.

        Parameters
        ----------
        state : str, optional
            Initial state type (default: 'random'):
            - 'random': Random spins with equal probability
            - 'up': All spins +1
            - 'down': All spins -1
            - 'checkerboard': Alternating +1/-1 pattern

        Raises
        ------
        ConfigurationError
            If state is not valid.
        """
        state = validate_initial_state(state)

        if state == "random":
            self._spins = self._rng.choice(
                np.array([-1, 1], dtype=np.int8), size=self._size
            )
        elif state == "up":
            self._spins = np.ones(self._size, dtype=np.int8)
        elif state == "down":
            self._spins = -np.ones(self._size, dtype=np.int8)
        elif state == "checkerboard":
            self._spins = np.array(
                [1 if i % 2 == 0 else -1 for i in range(self._size)],
                dtype=np.int8,
            )

    def get_energy(self) -> float:
        """Calculate the total energy.

        Returns
        -------
        float
            Total energy E = -J * Σ_i s_i * s_{i+1}

        Notes
        -----
        For periodic BC, includes the bond between last and first spin.
        For fixed BC, only includes N-1 bonds.
        """
        # Sum over all nearest-neighbor pairs
        energy = -self._coupling * np.sum(self._spins[:-1] * self._spins[1:])

        # Add boundary term for periodic BC
        if self._boundary == "periodic":
            energy -= self._coupling * self._spins[-1] * self._spins[0]

        return float(energy)

    def get_magnetization(self) -> float:
        """Calculate the total magnetization.

        Returns
        -------
        float
            Total magnetization M = Σ_i s_i
        """
        return float(np.sum(self._spins))

    def get_neighbor_sum(self, site: Tuple[int, ...]) -> int:
        """Calculate the sum of neighboring spins.

        Parameters
        ----------
        site : tuple of int
            Site index as a 1-tuple, e.g., (i,).

        Returns
        -------
        int
            Sum of neighboring spin values.
        """
        i = site[0]
        neighbor_sum = 0

        if self._boundary == "periodic":
            # Periodic: always 2 neighbors
            neighbor_sum = self._spins[(i - 1) % self._size]
            neighbor_sum += self._spins[(i + 1) % self._size]
        else:
            # Fixed BC: edge spins have only 1 neighbor
            if i > 0:
                neighbor_sum += self._spins[i - 1]
            if i < self._size - 1:
                neighbor_sum += self._spins[i + 1]

        return int(neighbor_sum)

    def flip_spin(self, site: Tuple[int, ...]) -> None:
        """Flip the spin at the specified site.

        Parameters
        ----------
        site : tuple of int
            Site index as a 1-tuple, e.g., (i,).
        """
        i = site[0]
        self._spins[i] = -self._spins[i]

    def get_energy_change(self, site: Tuple[int, ...]) -> float:
        """Calculate energy change from flipping a spin.

        Parameters
        ----------
        site : tuple of int
            Site index as a 1-tuple, e.g., (i,).

        Returns
        -------
        float
            Energy change ΔE = 2 * J * s_i * (s_{i-1} + s_{i+1})

        Notes
        -----
        Possible values are -4J, -2J, 0, +2J, +4J for periodic BC.
        For fixed BC at edges, possible values include -2J, 0, +2J.
        """
        i = site[0]
        neighbor_sum = self.get_neighbor_sum(site)
        return 2.0 * self._coupling * self._spins[i] * neighbor_sum

    def copy(self) -> "Ising1D":
        """Create a deep copy of the model.

        Returns
        -------
        Ising1D
            Independent copy with same parameters and spin configuration.
        """
        new_model = Ising1D(
            size=self._size,
            temperature=self._temperature,
            coupling=self._coupling,
            boundary=self._boundary,
        )
        new_model._spins = self._spins.copy()
        return new_model

    def random_site(self) -> Tuple[int]:
        """Select a random lattice site.

        Returns
        -------
        tuple of int
            Random site index as a 1-tuple.
        """
        return (self._rng.integers(0, self._size),)

    def set_seed(self, seed: int) -> None:
        """Set the random number generator seed.

        Parameters
        ----------
        seed : int
            Seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Ising1D(size={self._size}, "
            f"temperature={self._temperature:.4f}, "
            f"coupling={self._coupling:.4f}, "
            f"boundary='{self._boundary}')"
        )
