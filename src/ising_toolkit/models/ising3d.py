"""3D Ising model implementation."""

from typing import Dict, Tuple

import numpy as np

from ising_toolkit.models.base import IsingModel
from ising_toolkit.utils import (
    validate_lattice_size,
    validate_initial_state,
    validate_temperature,
    DEFAULT_COUPLING,
    CRITICAL_TEMP_3D,
)


class Ising3D(IsingModel):
    """3D Ising model on a cubic lattice.

    Implements the Ising model on a three-dimensional simple cubic lattice
    with either periodic or fixed boundary conditions.

    The Hamiltonian is:
        H = -J * Σ_{<i,j>} s_i * s_j

    where the sum is over all nearest-neighbor pairs on the cubic lattice.
    Each spin has 6 neighbors (±x, ±y, ±z directions).

    Parameters
    ----------
    size : int
        Linear size of the lattice. Total spins = size^3. Must be >= 2.
    temperature : float
        Temperature in units of J/kB. Must be positive.
    coupling : float, optional
        Coupling constant J (default: 1.0). Positive for ferromagnetic.
    boundary : str, optional
        Boundary condition: 'periodic' or 'fixed' (default: 'periodic').

    Attributes
    ----------
    size : int
        Linear size of the cubic lattice.
    shape : tuple
        Shape of the spin array (size, size, size).
    acceptance_probs : dict
        Precomputed acceptance probabilities for Metropolis algorithm.

    Examples
    --------
    >>> model = Ising3D(size=16, temperature=4.511)  # Near Tc
    >>> model.initialize('random')
    >>> energy = model.get_energy()
    >>> mag = model.get_magnetization()

    Notes
    -----
    The 3D Ising model has no exact solution. The critical temperature
    is approximately Tc ≈ 4.511 (in units of J/kB), determined from
    numerical simulations.
    """

    def __init__(
        self,
        size: int,
        temperature: float,
        coupling: float = DEFAULT_COUPLING,
        boundary: str = "periodic",
    ) -> None:
        """Initialize the 3D Ising model.

        Parameters
        ----------
        size : int
            Linear size of lattice. Total spins = size^3. Must be >= 2.
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
        self._spins = np.ones((self._size, self._size, self._size), dtype=np.int8)
        self._rng = np.random.default_rng()

        # Precompute acceptance probabilities
        self._acceptance_probs: Dict[int, float] = {}
        self._update_acceptance_probs()

    def _update_acceptance_probs(self) -> None:
        """Precompute Metropolis acceptance probabilities.

        For the 3D Ising model with 6 neighbors, the energy change from
        flipping a spin can be -12J, -8J, -4J, 0, +4J, +8J, or +12J.
        We precompute exp(-β*ΔE) for positive ΔE values.
        """
        self._acceptance_probs = {}
        for dE in [4, 8, 12]:
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
        """int: Linear size of the lattice."""
        return self._size

    @property
    def shape(self) -> Tuple[int, int, int]:
        """tuple: Shape of the spin array (size, size, size)."""
        return (self._size, self._size, self._size)

    @property
    def dimension(self) -> int:
        """int: Spatial dimension (3 for cubic lattice)."""
        return 3

    @property
    def n_spins(self) -> int:
        """int: Total number of spins (size^3)."""
        return self._size ** 3

    @property
    def n_neighbors(self) -> int:
        """int: Number of neighbors per spin (6 for cubic lattice)."""
        return 6

    @property
    def spins(self) -> np.ndarray:
        """np.ndarray: Current spin configuration with shape (size, size, size)."""
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
            - 'checkerboard': Alternating +1/-1 pattern (3D Néel state)

        Raises
        ------
        ConfigurationError
            If state is not valid.
        """
        state = validate_initial_state(state)
        L = self._size

        if state == "random":
            self._spins = self._rng.choice(
                np.array([-1, 1], dtype=np.int8), size=(L, L, L)
            )
        elif state == "up":
            self._spins = np.ones((L, L, L), dtype=np.int8)
        elif state == "down":
            self._spins = -np.ones((L, L, L), dtype=np.int8)
        elif state == "checkerboard":
            # Create 3D checkerboard pattern
            self._spins = np.ones((L, L, L), dtype=np.int8)
            for i in range(L):
                for j in range(L):
                    for k in range(L):
                        if (i + j + k) % 2 == 1:
                            self._spins[i, j, k] = -1

    def get_energy(self) -> float:
        """Calculate the total energy using vectorized operations.

        Returns
        -------
        float
            Total energy E = -J * Σ_{<i,j>} s_i * s_j

        Notes
        -----
        Uses efficient numpy operations to compute the sum over
        all nearest-neighbor pairs without explicit loops.
        For periodic BC, total bonds = 3 * N (N bonds in each direction).
        """
        # X-direction bonds
        energy = -self._coupling * np.sum(self._spins[:-1, :, :] * self._spins[1:, :, :])

        # Y-direction bonds
        energy -= self._coupling * np.sum(self._spins[:, :-1, :] * self._spins[:, 1:, :])

        # Z-direction bonds
        energy -= self._coupling * np.sum(self._spins[:, :, :-1] * self._spins[:, :, 1:])

        # Boundary terms for periodic BC
        if self._boundary == "periodic":
            # X-direction wrap
            energy -= self._coupling * np.sum(self._spins[-1, :, :] * self._spins[0, :, :])
            # Y-direction wrap
            energy -= self._coupling * np.sum(self._spins[:, -1, :] * self._spins[:, 0, :])
            # Z-direction wrap
            energy -= self._coupling * np.sum(self._spins[:, :, -1] * self._spins[:, :, 0])

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
            Site indices (i, j, k).

        Returns
        -------
        int
            Sum of the six neighboring spin values.
            Ranges from -6 to +6.
        """
        i, j, k = site
        L = self._size

        if self._boundary == "periodic":
            neighbor_sum = (
                self._spins[(i + 1) % L, j, k]
                + self._spins[(i - 1) % L, j, k]
                + self._spins[i, (j + 1) % L, k]
                + self._spins[i, (j - 1) % L, k]
                + self._spins[i, j, (k + 1) % L]
                + self._spins[i, j, (k - 1) % L]
            )
        else:
            # Fixed BC: only count existing neighbors
            neighbor_sum = 0
            if i < L - 1:
                neighbor_sum += self._spins[i + 1, j, k]
            if i > 0:
                neighbor_sum += self._spins[i - 1, j, k]
            if j < L - 1:
                neighbor_sum += self._spins[i, j + 1, k]
            if j > 0:
                neighbor_sum += self._spins[i, j - 1, k]
            if k < L - 1:
                neighbor_sum += self._spins[i, j, k + 1]
            if k > 0:
                neighbor_sum += self._spins[i, j, k - 1]

        return int(neighbor_sum)

    def flip_spin(self, site: Tuple[int, ...]) -> None:
        """Flip the spin at the specified site.

        Parameters
        ----------
        site : tuple of int
            Site indices (i, j, k).
        """
        i, j, k = site
        self._spins[i, j, k] = -self._spins[i, j, k]

    def get_energy_change(self, site: Tuple[int, ...]) -> float:
        """Calculate energy change from flipping a spin.

        Parameters
        ----------
        site : tuple of int
            Site indices (i, j, k) of the spin to potentially flip.

        Returns
        -------
        float
            Energy change ΔE = 2 * J * s_{i,j,k} * Σ_neighbors s

        Notes
        -----
        Possible values are -12J, -8J, -4J, 0, +4J, +8J, +12J for
        periodic BC with 6 neighbors.
        """
        i, j, k = site
        neighbor_sum = self.get_neighbor_sum(site)
        return 2.0 * self._coupling * self._spins[i, j, k] * neighbor_sum

    def copy(self) -> "Ising3D":
        """Create a deep copy of the model.

        Returns
        -------
        Ising3D
            Independent copy with same parameters and spin configuration.
        """
        new_model = Ising3D(
            size=self._size,
            temperature=self._temperature,
            coupling=self._coupling,
            boundary=self._boundary,
        )
        new_model._spins = self._spins.copy()
        return new_model

    def random_site(self) -> Tuple[int, int, int]:
        """Select a random lattice site.

        Returns
        -------
        tuple of int
            Random site indices (i, j, k).
        """
        i = self._rng.integers(0, self._size)
        j = self._rng.integers(0, self._size)
        k = self._rng.integers(0, self._size)
        return (i, j, k)

    def set_seed(self, seed: int) -> None:
        """Set the random number generator seed.

        Parameters
        ----------
        seed : int
            Seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def get_configuration_slice(self, axis: int = 2, index: int = 0) -> np.ndarray:
        """Get a 2D slice of the spin configuration.

        Parameters
        ----------
        axis : int, optional
            Axis perpendicular to the slice (default: 2, i.e., z-axis).
        index : int, optional
            Index along the axis (default: 0).

        Returns
        -------
        np.ndarray
            2D array suitable for visualization.

        Examples
        --------
        >>> model = Ising3D(size=16, temperature=4.0)
        >>> model.initialize('random')
        >>> slice_xy = model.get_configuration_slice(axis=2, index=8)
        >>> plt.imshow(slice_xy, cmap='binary')
        """
        if axis == 0:
            return self._spins[index, :, :].copy()
        elif axis == 1:
            return self._spins[:, index, :].copy()
        else:
            return self._spins[:, :, index].copy()

    def is_near_critical(self, tolerance: float = 0.1) -> bool:
        """Check if temperature is near the critical temperature.

        Parameters
        ----------
        tolerance : float, optional
            Relative tolerance for comparison (default: 0.1).

        Returns
        -------
        bool
            True if |T - Tc| / Tc < tolerance.
        """
        return abs(self._temperature - CRITICAL_TEMP_3D) / CRITICAL_TEMP_3D < tolerance

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Ising3D(size={self._size}, "
            f"n_spins={self.n_spins}, "
            f"temperature={self._temperature:.4f}, "
            f"coupling={self._coupling:.4f}, "
            f"boundary='{self._boundary}')"
        )
