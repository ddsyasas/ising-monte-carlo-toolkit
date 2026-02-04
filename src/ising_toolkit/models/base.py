"""Abstract base class for Ising models."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ising_toolkit.utils import (
    validate_temperature,
    validate_positive_float,
    validate_boundary_condition,
    DEFAULT_COUPLING,
)


class IsingModel(ABC):
    """Abstract base class for Ising model implementations.

    This class defines the interface that all Ising model implementations
    must follow. It provides abstract methods for core operations like
    computing energy, magnetization, and performing spin flips.

    The Ising model Hamiltonian is:
        H = -J * Σ_{<i,j>} s_i * s_j - h * Σ_i s_i

    where:
        - J is the coupling constant (ferromagnetic if J > 0)
        - s_i ∈ {-1, +1} are the spin variables
        - <i,j> denotes nearest-neighbor pairs
        - h is the external magnetic field (not included in base class)

    Parameters
    ----------
    temperature : float
        The temperature of the system in units of J/kB.
        Must be positive.
    coupling : float, optional
        The nearest-neighbor coupling constant J (default: 1.0).
        Positive values favor ferromagnetic alignment.
    boundary : str, optional
        Boundary condition type: 'periodic' or 'fixed' (default: 'periodic').

    Attributes
    ----------
    temperature : float
        The system temperature.
    coupling : float
        The coupling constant J.
    boundary : str
        The boundary condition type.
    beta : float
        Inverse temperature β = 1/(kB*T) = 1/T in natural units.

    Examples
    --------
    Subclasses must implement all abstract methods and properties:

    >>> class Ising2D(IsingModel):
    ...     # Implementation for 2D square lattice
    ...     pass
    """

    def __init__(
        self,
        temperature: float,
        coupling: float = DEFAULT_COUPLING,
        boundary: str = "periodic",
    ) -> None:
        """Initialize the Ising model base class.

        Parameters
        ----------
        temperature : float
            The temperature of the system in units of J/kB.
            Must be strictly positive.
        coupling : float, optional
            The nearest-neighbor coupling constant J (default: 1.0).
            Must be positive for ferromagnetic systems.
        boundary : str, optional
            Boundary condition: 'periodic' or 'fixed' (default: 'periodic').

        Raises
        ------
        ConfigurationError
            If temperature is not positive, coupling is not positive,
            or boundary condition is invalid.
        """
        self._temperature = validate_temperature(temperature)
        self._coupling = validate_positive_float(coupling, "coupling")
        self._boundary = validate_boundary_condition(boundary)

    @property
    def temperature(self) -> float:
        """float: The system temperature in units of J/kB."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the system temperature.

        Parameters
        ----------
        value : float
            New temperature value. Must be positive.

        Raises
        ------
        ConfigurationError
            If temperature is not positive.
        """
        self._temperature = validate_temperature(value)

    @property
    def coupling(self) -> float:
        """float: The nearest-neighbor coupling constant J."""
        return self._coupling

    @property
    def boundary(self) -> str:
        """str: The boundary condition type ('periodic' or 'fixed')."""
        return self._boundary

    @property
    def beta(self) -> float:
        """float: Inverse temperature β = 1/T in natural units (kB=1)."""
        return 1.0 / self._temperature

    # =========================================================================
    # Abstract Properties (must be implemented by subclasses)
    # =========================================================================

    @property
    @abstractmethod
    def dimension(self) -> int:
        """int: Spatial dimension of the lattice (e.g., 1, 2, or 3).

        This determines the geometry of the spin interactions.
        For example, dimension=2 corresponds to a 2D square lattice.
        """
        pass

    @property
    @abstractmethod
    def n_spins(self) -> int:
        """int: Total number of spins in the system.

        For a d-dimensional hypercubic lattice with linear size L,
        this equals L^d.
        """
        pass

    @property
    @abstractmethod
    def n_neighbors(self) -> int:
        """int: Number of nearest neighbors per spin.

        For a d-dimensional hypercubic lattice with periodic boundary
        conditions, this equals 2*d (e.g., 4 for 2D square lattice).
        """
        pass

    @property
    @abstractmethod
    def spins(self) -> np.ndarray:
        """np.ndarray: The current spin configuration.

        Returns a view or copy of the spin array. Each element is
        either +1 (spin up) or -1 (spin down).

        The shape depends on the lattice geometry:
        - 1D: (L,)
        - 2D: (Lx, Ly)
        - 3D: (Lx, Ly, Lz)
        """
        pass

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def initialize(self, state: str = "random") -> None:
        """Initialize the spin configuration.

        Parameters
        ----------
        state : str, optional
            Initial state type (default: 'random'):
            - 'random': Random spins (+1 or -1 with equal probability)
            - 'up': All spins +1
            - 'down': All spins -1
            - 'checkerboard': Alternating +1/-1 pattern

        Raises
        ------
        ConfigurationError
            If state is not a valid initial state type.

        Notes
        -----
        This method modifies the spin configuration in place.
        """
        pass

    @abstractmethod
    def get_energy(self) -> float:
        """Calculate the total energy of the current configuration.

        Returns
        -------
        float
            Total energy H = -J * Σ_{<i,j>} s_i * s_j

        Notes
        -----
        The sum is over all unique nearest-neighbor pairs.
        Each pair is counted once, not twice.
        """
        pass

    @abstractmethod
    def get_magnetization(self) -> float:
        """Calculate the total magnetization of the current configuration.

        Returns
        -------
        float
            Total magnetization M = Σ_i s_i

        Notes
        -----
        The magnetization ranges from -N to +N, where N is the
        number of spins. A value near ±N indicates an ordered
        (ferromagnetic) state, while a value near 0 indicates
        a disordered (paramagnetic) state.
        """
        pass

    @abstractmethod
    def get_neighbor_sum(self, site: Tuple[int, ...]) -> int:
        """Calculate the sum of neighboring spins at a given site.

        Parameters
        ----------
        site : tuple of int
            The lattice site indices. The length of the tuple
            should match the lattice dimension.

        Returns
        -------
        int
            Sum of spin values of all nearest neighbors.
            For a 2D square lattice, this ranges from -4 to +4.

        Notes
        -----
        This is used to compute the local energy and energy change
        for spin flip updates.
        """
        pass

    @abstractmethod
    def flip_spin(self, site: Tuple[int, ...]) -> None:
        """Flip the spin at the specified site.

        Parameters
        ----------
        site : tuple of int
            The lattice site indices where the spin should be flipped.

        Notes
        -----
        This operation changes s_i -> -s_i at the given site.
        The operation is performed in place.
        """
        pass

    @abstractmethod
    def get_energy_change(self, site: Tuple[int, ...]) -> float:
        """Calculate the energy change from flipping a spin.

        Parameters
        ----------
        site : tuple of int
            The lattice site indices of the spin to potentially flip.

        Returns
        -------
        float
            Energy change ΔE that would result from flipping the spin.
            ΔE = 2 * J * s_i * Σ_j s_j, where j are neighbors of i.

        Notes
        -----
        This does NOT actually flip the spin. It only computes what
        the energy change would be. This is used in the Metropolis
        acceptance criterion: accept if ΔE < 0 or with probability
        exp(-β*ΔE) if ΔE > 0.
        """
        pass

    @abstractmethod
    def copy(self) -> "IsingModel":
        """Create a deep copy of the model.

        Returns
        -------
        IsingModel
            A new instance with the same parameters and an independent
            copy of the spin configuration.

        Notes
        -----
        Modifications to the copy will not affect the original.
        """
        pass

    @abstractmethod
    def random_site(self) -> Tuple[int, ...]:
        """Select a random lattice site uniformly.

        Returns
        -------
        tuple of int
            Random site indices suitable for use with flip_spin(),
            get_neighbor_sum(), etc.

        Notes
        -----
        Each site has equal probability 1/N of being selected,
        where N is the total number of spins.
        """
        pass

    # =========================================================================
    # Concrete Methods (implemented here, available to all subclasses)
    # =========================================================================

    def get_energy_per_spin(self) -> float:
        """Calculate the energy per spin.

        Returns
        -------
        float
            Energy per spin e = E / N, where E is total energy
            and N is the number of spins.

        Notes
        -----
        This intensive quantity is useful for comparing systems
        of different sizes and approaches a well-defined
        thermodynamic limit as N -> ∞.
        """
        return self.get_energy() / self.n_spins

    def get_magnetization_per_spin(self) -> float:
        """Calculate the magnetization per spin.

        Returns
        -------
        float
            Magnetization per spin m = M / N, where M is total
            magnetization and N is the number of spins.
            This ranges from -1 to +1.

        Notes
        -----
        This intensive quantity (order parameter) is useful for
        identifying phase transitions. Below Tc, |m| > 0 indicates
        spontaneous magnetization. Above Tc, m ≈ 0 in the
        thermodynamic limit.
        """
        return self.get_magnetization() / self.n_spins

    def __repr__(self) -> str:
        """Return a string representation of the model.

        Returns
        -------
        str
            A string containing the class name and key parameters.
        """
        return (
            f"{self.__class__.__name__}("
            f"n_spins={self.n_spins}, "
            f"temperature={self.temperature:.4f}, "
            f"coupling={self.coupling:.4f}, "
            f"boundary='{self.boundary}')"
        )
