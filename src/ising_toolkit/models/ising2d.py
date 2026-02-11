"""2D Ising model implementation with optional Numba acceleration."""

from typing import Dict, Optional, Tuple

import numpy as np

from ising_toolkit.models.base import IsingModel
from ising_toolkit.utils import (
    validate_lattice_size,
    validate_initial_state,
    validate_temperature,
    DEFAULT_COUPLING,
    CRITICAL_TEMP_2D,
)

# Try to import Numba kernels
try:
    from ising_toolkit.utils.numba_kernels import (
        NUMBA_AVAILABLE,
        _calculate_energy_2d,
        _calculate_magnetization_2d,
        _metropolis_sweep_2d,
        _wolff_cluster_2d,
        precompute_acceptance_probs_2d,
    )
except ImportError:
    NUMBA_AVAILABLE = False
    _calculate_energy_2d = None
    _calculate_magnetization_2d = None
    _metropolis_sweep_2d = None
    _wolff_cluster_2d = None
    precompute_acceptance_probs_2d = None


class Ising2D(IsingModel):
    """2D Ising model on a square lattice with optional Numba acceleration.

    Implements the Ising model on a two-dimensional square lattice
    with either periodic or fixed boundary conditions.

    The Hamiltonian is:
        H = -J * Σ_{<i,j>} s_i * s_j

    where the sum is over all nearest-neighbor pairs on the square lattice.

    Parameters
    ----------
    size : int
        Linear size of the lattice. Total spins = size^2. Must be >= 2.
    temperature : float
        Temperature in units of J/kB. Must be positive.
    coupling : float, optional
        Coupling constant J (default: 1.0). Positive for ferromagnetic.
    boundary : str, optional
        Boundary condition: 'periodic' or 'fixed' (default: 'periodic').
    use_numba : bool, optional
        Whether to use Numba-accelerated kernels when available (default: True).

    Attributes
    ----------
    size : int
        Linear size of the square lattice.
    shape : tuple
        Shape of the spin array (size, size).
    acceptance_probs : dict
        Precomputed acceptance probabilities for Metropolis algorithm.
    use_numba : bool
        Whether Numba acceleration is active.

    Examples
    --------
    >>> model = Ising2D(size=32, temperature=2.269)  # Near Tc
    >>> model.initialize('random')
    >>> energy = model.get_energy()
    >>> mag = model.get_magnetization()

    >>> # Explicitly disable Numba for comparison
    >>> model_slow = Ising2D(size=32, temperature=2.269, use_numba=False)

    Notes
    -----
    The 2D Ising model on a square lattice has an exact solution
    (Onsager, 1944) with critical temperature Tc = 2/ln(1+√2) ≈ 2.269.

    When Numba is available and enabled, the model uses JIT-compiled
    kernels for significant performance improvements (100x+ speedup).
    """

    def __init__(
        self,
        size: int,
        temperature: float,
        coupling: float = DEFAULT_COUPLING,
        boundary: str = "periodic",
        use_numba: bool = True,
    ) -> None:
        """Initialize the 2D Ising model.

        Parameters
        ----------
        size : int
            Linear size of lattice. Total spins = size^2. Must be >= 2.
        temperature : float
            Temperature in units of J/kB. Must be positive.
        coupling : float, optional
            Coupling constant J (default: 1.0).
        boundary : str, optional
            Boundary condition: 'periodic' or 'fixed' (default: 'periodic').
        use_numba : bool, optional
            Whether to use Numba acceleration (default: True).

        Raises
        ------
        ConfigurationError
            If parameters are invalid.
        """
        super().__init__(temperature, coupling, boundary)
        self._size = validate_lattice_size(size, min_size=2)
        self._spins = np.ones((self._size, self._size), dtype=np.int8)
        self._rng = np.random.default_rng()

        # Numba acceleration settings
        self._use_numba = use_numba and NUMBA_AVAILABLE and boundary == "periodic"
        if use_numba and not NUMBA_AVAILABLE:
            import warnings
            warnings.warn(
                "Numba not available, falling back to pure Python. "
                "Install numba for 100x+ speedup: pip install numba",
                RuntimeWarning,
                stacklevel=2
            )
        if use_numba and boundary != "periodic":
            import warnings
            warnings.warn(
                "Numba kernels only support periodic boundary conditions. "
                "Falling back to pure Python for fixed BC.",
                RuntimeWarning,
                stacklevel=2
            )

        # Precompute acceptance probabilities
        self._acceptance_probs: Dict[int, float] = {}
        self._acceptance_probs_array: Optional[np.ndarray] = None
        self._update_acceptance_probs()

    def _update_acceptance_probs(self) -> None:
        """Precompute Metropolis acceptance probabilities.

        For the 2D Ising model, the energy change from flipping a spin
        can only be -8J, -4J, 0, +4J, or +8J. We precompute exp(-β*ΔE)
        for positive ΔE values.
        """
        # Dictionary for Python implementation
        self._acceptance_probs = {}
        for dE in [4, 8]:
            actual_dE = dE * self._coupling
            self._acceptance_probs[dE] = np.exp(-self.beta * actual_dE)

        # Array for Numba implementation
        if self._use_numba and precompute_acceptance_probs_2d is not None:
            self._acceptance_probs_array = precompute_acceptance_probs_2d(
                self.beta * self._coupling
            )
        else:
            self._acceptance_probs_array = np.array([
                np.exp(-self.beta * self._coupling * 4),
                np.exp(-self.beta * self._coupling * 8),
                1.0
            ], dtype=np.float64)

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
    def use_numba(self) -> bool:
        """bool: Whether Numba acceleration is active."""
        return self._use_numba

    @property
    def size(self) -> int:
        """int: Linear size of the lattice."""
        return self._size

    @property
    def shape(self) -> Tuple[int, int]:
        """tuple: Shape of the spin array (size, size)."""
        return (self._size, self._size)

    @property
    def dimension(self) -> int:
        """int: Spatial dimension (2 for square lattice)."""
        return 2

    @property
    def n_spins(self) -> int:
        """int: Total number of spins (size^2)."""
        return self._size * self._size

    @property
    def n_neighbors(self) -> int:
        """int: Number of neighbors per spin (4 for square lattice)."""
        return 4

    @property
    def spins(self) -> np.ndarray:
        """np.ndarray: Current spin configuration with shape (size, size)."""
        return self._spins

    @property
    def acceptance_probs(self) -> Dict[int, float]:
        """dict: Precomputed acceptance probabilities for energy changes."""
        return self._acceptance_probs.copy()

    @property
    def acceptance_probs_array(self) -> np.ndarray:
        """np.ndarray: Acceptance probabilities array for Numba kernels."""
        return self._acceptance_probs_array

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
                np.array([-1, 1], dtype=np.int8), size=(self._size, self._size)
            )
        elif state == "up":
            self._spins = np.ones((self._size, self._size), dtype=np.int8)
        elif state == "down":
            self._spins = -np.ones((self._size, self._size), dtype=np.int8)
        elif state == "checkerboard":
            # Create checkerboard pattern
            self._spins = np.ones((self._size, self._size), dtype=np.int8)
            self._spins[1::2, ::2] = -1  # Odd rows, even columns
            self._spins[::2, 1::2] = -1  # Even rows, odd columns

    def get_energy(self) -> float:
        """Calculate the total energy.

        Uses Numba-accelerated kernel when available for ~100x speedup.

        Returns
        -------
        float
            Total energy E = -J * Σ_{<i,j>} s_i * s_j

        Notes
        -----
        Uses efficient numpy operations or Numba JIT-compiled code
        to compute the sum over all nearest-neighbor pairs.
        """
        # Using vectorized operations for efficiency:
        # E = -J * sum of (spin * neighbor) for all nearest-neighbor pairs
        if self._use_numba and _calculate_energy_2d is not None:
            # Numba-accelerated implementation
            return float(_calculate_energy_2d(self._spins)) * self._coupling
        else:
            # Pure Python/NumPy implementation
            return self._get_energy_python()

    def _get_energy_python(self) -> float:
        """Pure Python/NumPy energy calculation."""
        # Horizontal bonds (each site with its right neighbor)
        energy = -self._coupling * np.sum(self._spins[:, :-1] * self._spins[:, 1:])

        # Vertical bonds (each site with its bottom neighbor)
        energy -= self._coupling * np.sum(self._spins[:-1, :] * self._spins[1:, :])

        # Boundary terms for periodic BC
        if self._boundary == "periodic":
            # Right edge to left edge
            energy -= self._coupling * np.sum(self._spins[:, -1] * self._spins[:, 0])
            # Bottom edge to top edge
            energy -= self._coupling * np.sum(self._spins[-1, :] * self._spins[0, :])

        return float(energy)

    def get_magnetization(self) -> float:
        """Calculate the total magnetization.

        Returns
        -------
        float
            Total magnetization M = Σ_i s_i
        """
        if self._use_numba and _calculate_magnetization_2d is not None:
            return float(_calculate_magnetization_2d(self._spins))
        else:
            return float(np.sum(self._spins))

    def metropolis_sweep(self) -> int:
        """Perform one Metropolis sweep (N single-spin flip attempts).

        Uses Numba-accelerated kernel when available for ~100x speedup.

        Returns
        -------
        int
            Number of accepted spin flips.

        Notes
        -----
        This method is more efficient than calling step() on a sampler
        when only a single sweep is needed, as it avoids Python overhead.
        """
        if self._use_numba and _metropolis_sweep_2d is not None:
            return _metropolis_sweep_2d(
                self._spins, self.beta * self._coupling, self._acceptance_probs_array
            )
        else:
            return self._metropolis_sweep_python()

    def _metropolis_sweep_python(self) -> int:
        """Pure Python Metropolis sweep."""
        L = self._size
        n_accepted = 0

        for _ in range(L * L):
            i = self._rng.integers(0, L)
            j = self._rng.integers(0, L)

            # Neighbor sum
            if self._boundary == "periodic":
                nn_sum = (
                    self._spins[(i + 1) % L, j] +
                    self._spins[(i - 1) % L, j] +
                    self._spins[i, (j + 1) % L] +
                    self._spins[i, (j - 1) % L]
                )
            else:
                nn_sum = 0
                if i < L - 1:
                    nn_sum += self._spins[i + 1, j]
                if i > 0:
                    nn_sum += self._spins[i - 1, j]
                if j < L - 1:
                    nn_sum += self._spins[i, j + 1]
                if j > 0:
                    nn_sum += self._spins[i, j - 1]

            # Energy change for flipping spin s_i: ΔE = -2*J*s_i*Σ(neighbors)
            # Factor of 2 arises because flipping s_i -> -s_i changes each
            # pair interaction -J*s_i*s_j by 2*J*s_i*s_j
            dE = 2 * self._coupling * self._spins[i, j] * nn_sum

            # Metropolis acceptance: always accept if ΔE ≤ 0 (lower energy),
            # otherwise accept with probability exp(-β*ΔE)
            if dE <= 0:
                self._spins[i, j] *= -1
                n_accepted += 1
            else:
                dE_int = int(round(dE / self._coupling))
                if dE_int in self._acceptance_probs:
                    if self._rng.random() < self._acceptance_probs[dE_int]:
                        self._spins[i, j] *= -1
                        n_accepted += 1

        return n_accepted

    def wolff_step(self) -> int:
        """Perform one Wolff cluster update.

        Uses Numba-accelerated kernel when available.

        Returns
        -------
        int
            Size of the flipped cluster.

        Notes
        -----
        The Wolff algorithm is more efficient near the critical temperature
        as it avoids critical slowing down.
        """
        if self._use_numba and _wolff_cluster_2d is not None:
            L = self._size
            p_add = 1.0 - np.exp(-2.0 * self.beta * self._coupling)
            seed_i = np.random.randint(0, L)
            seed_j = np.random.randint(0, L)
            return _wolff_cluster_2d(self._spins, p_add, seed_i, seed_j)
        else:
            return self._wolff_step_python()

    def _wolff_step_python(self) -> int:
        """Pure Python Wolff cluster update."""
        L = self._size
        p_add = 1.0 - np.exp(-2.0 * self.beta * self._coupling)

        # Random seed site
        seed_i = self._rng.integers(0, L)
        seed_j = self._rng.integers(0, L)
        seed_spin = self._spins[seed_i, seed_j]

        # Track cluster membership
        in_cluster = np.zeros((L, L), dtype=bool)
        in_cluster[seed_i, seed_j] = True

        # Stack for BFS
        stack = [(seed_i, seed_j)]
        cluster_size = 1

        while stack:
            i, j = stack.pop()

            # Check neighbors
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni = (i + di) % L
                nj = (j + dj) % L

                if (not in_cluster[ni, nj] and
                    self._spins[ni, nj] == seed_spin and
                    self._rng.random() < p_add):

                    in_cluster[ni, nj] = True
                    stack.append((ni, nj))
                    cluster_size += 1

        # Flip cluster
        self._spins[in_cluster] *= -1

        return cluster_size

    def get_neighbor_sum(self, site: Tuple[int, ...]) -> int:
        """Calculate the sum of neighboring spins.

        Parameters
        ----------
        site : tuple of int
            Site indices (i, j).

        Returns
        -------
        int
            Sum of the four neighboring spin values.
            Ranges from -4 to +4.
        """
        i, j = site
        L = self._size

        if self._boundary == "periodic":
            neighbor_sum = (
                self._spins[(i + 1) % L, j]
                + self._spins[(i - 1) % L, j]
                + self._spins[i, (j + 1) % L]
                + self._spins[i, (j - 1) % L]
            )
        else:
            # Fixed BC: only count existing neighbors
            neighbor_sum = 0
            if i < L - 1:
                neighbor_sum += self._spins[i + 1, j]
            if i > 0:
                neighbor_sum += self._spins[i - 1, j]
            if j < L - 1:
                neighbor_sum += self._spins[i, j + 1]
            if j > 0:
                neighbor_sum += self._spins[i, j - 1]

        return int(neighbor_sum)

    def flip_spin(self, site: Tuple[int, ...]) -> None:
        """Flip the spin at the specified site.

        Parameters
        ----------
        site : tuple of int
            Site indices (i, j).
        """
        i, j = site
        self._spins[i, j] = -self._spins[i, j]

    def get_energy_change(self, site: Tuple[int, ...]) -> float:
        """Calculate energy change from flipping a spin.

        Parameters
        ----------
        site : tuple of int
            Site indices (i, j) of the spin to potentially flip.

        Returns
        -------
        float
            Energy change ΔE = 2 * J * s_{i,j} * Σ_neighbors s

        Notes
        -----
        Possible values are -8J, -4J, 0, +4J, +8J for periodic BC
        with 4 neighbors.
        """
        i, j = site
        neighbor_sum = self.get_neighbor_sum(site)
        return 2.0 * self._coupling * self._spins[i, j] * neighbor_sum

    def copy(self) -> "Ising2D":
        """Create a deep copy of the model.

        Returns
        -------
        Ising2D
            Independent copy with same parameters and spin configuration.
        """
        new_model = Ising2D(
            size=self._size,
            temperature=self._temperature,
            coupling=self._coupling,
            boundary=self._boundary,
            use_numba=self._use_numba,
        )
        new_model._spins = self._spins.copy()
        return new_model

    def random_site(self) -> Tuple[int, int]:
        """Select a random lattice site.

        Returns
        -------
        tuple of int
            Random site indices (i, j).
        """
        i = self._rng.integers(0, self._size)
        j = self._rng.integers(0, self._size)
        return (i, j)

    def set_seed(self, seed: int) -> None:
        """Set the random number generator seed.

        Parameters
        ----------
        seed : int
            Seed for reproducibility.
        """
        self._rng = np.random.default_rng(seed)

    def get_configuration_image(self) -> np.ndarray:
        """Get spin configuration as an image array.

        Returns
        -------
        np.ndarray
            2D array suitable for matplotlib imshow.
            Values are +1 (white/up) and -1 (black/down).

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> model = Ising2D(size=32, temperature=2.0)
        >>> model.initialize('random')
        >>> plt.imshow(model.get_configuration_image(), cmap='binary')
        >>> plt.show()
        """
        return self._spins.copy()

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
        return abs(self._temperature - CRITICAL_TEMP_2D) / CRITICAL_TEMP_2D < tolerance

    def __repr__(self) -> str:
        """Return string representation."""
        numba_str = "numba" if self._use_numba else "python"
        return (
            f"Ising2D(size={self._size}, "
            f"n_spins={self.n_spins}, "
            f"temperature={self._temperature:.4f}, "
            f"coupling={self._coupling:.4f}, "
            f"boundary='{self._boundary}', "
            f"backend='{numba_str}')"
        )
