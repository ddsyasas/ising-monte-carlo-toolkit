"""Abstract base class for Monte Carlo samplers."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ising_toolkit.models.base import IsingModel
from ising_toolkit.io.results import SimulationResults
from ising_toolkit.utils import (
    validate_positive_integer,
    DEFAULT_MC_STEPS,
    DEFAULT_EQUILIBRATION,
    DEFAULT_MEASUREMENT_INTERVAL,
)


class Sampler(ABC):
    """Abstract base class for Monte Carlo samplers.

    This class defines the interface for Monte Carlo sampling algorithms
    used to simulate Ising models. Subclasses must implement the step()
    method which performs a single Monte Carlo update.

    The run() method provides a complete simulation workflow including
    equilibration, measurements, and optional configuration saving.

    Parameters
    ----------
    model : IsingModel
        The Ising model to simulate.
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.

    Attributes
    ----------
    model : IsingModel
        The Ising model being simulated.
    rng : np.random.Generator
        NumPy random number generator for stochastic updates.
    seed : int or None
        The random seed used (None if not specified).
    n_accepted : int
        Counter for accepted moves (updated by subclasses).
    n_attempted : int
        Counter for attempted moves (updated by subclasses).

    Examples
    --------
    Subclasses implement the step() method:

    >>> class MetropolisSampler(Sampler):
    ...     def step(self):
    ...         # Implementation here
    ...         pass
    ...
    >>> sampler = MetropolisSampler(model, seed=42)
    >>> results = sampler.run(n_steps=10000, equilibration=1000)
    """

    def __init__(self, model: IsingModel, seed: Optional[int] = None) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        model : IsingModel
            The Ising model to simulate.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.model = model
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Counters for acceptance statistics
        self.n_accepted = 0
        self.n_attempted = 0

    @property
    def acceptance_rate(self) -> float:
        """float: Fraction of proposed moves that were accepted."""
        if self.n_attempted == 0:
            return 0.0
        return self.n_accepted / self.n_attempted

    def reset_counters(self) -> None:
        """Reset acceptance counters to zero."""
        self.n_accepted = 0
        self.n_attempted = 0

    @abstractmethod
    def step(self) -> Optional[int]:
        """Perform one Monte Carlo step.

        A step typically consists of N attempted spin updates, where N
        is the number of spins, so that on average each spin has one
        opportunity to flip per step.

        Returns
        -------
        int or None
            Number of accepted moves in this step, or None if not tracked.

        Notes
        -----
        Subclasses must implement this method. The implementation should:
        1. Select spin(s) to update
        2. Compute the energy change
        3. Accept or reject according to the algorithm's criterion
        4. Update acceptance counters
        """
        pass

    def _equilibrate(
        self,
        n_steps: int,
        progress: bool = False,
    ) -> None:
        """Run equilibration phase without measurements.

        Parameters
        ----------
        n_steps : int
            Number of equilibration steps to perform.
        progress : bool, optional
            Whether to show progress bar (default: False).

        Notes
        -----
        Equilibration allows the system to relax from its initial
        configuration to thermal equilibrium. No measurements are
        taken during this phase.
        """
        iterator = range(n_steps)
        if progress:
            iterator = tqdm(
                iterator,
                desc="Equilibrating",
                unit="step",
                leave=False,
            )

        for _ in iterator:
            self.step()

    def _measure(self) -> Dict[str, float]:
        """Measure current observables.

        Returns
        -------
        dict
            Dictionary containing measured observables:
            - 'energy': Total energy
            - 'magnetization': Total magnetization
            - 'energy_per_spin': Energy per spin
            - 'magnetization_per_spin': Magnetization per spin
        """
        return {
            "energy": self.model.get_energy(),
            "magnetization": self.model.get_magnetization(),
            "energy_per_spin": self.model.get_energy_per_spin(),
            "magnetization_per_spin": self.model.get_magnetization_per_spin(),
        }

    def run(
        self,
        n_steps: int = DEFAULT_MC_STEPS,
        equilibration: int = DEFAULT_EQUILIBRATION,
        measurement_interval: int = DEFAULT_MEASUREMENT_INTERVAL,
        save_configurations: bool = False,
        configuration_interval: Optional[int] = None,
        progress: bool = True,
    ) -> SimulationResults:
        """Run a complete Monte Carlo simulation.

        Performs equilibration followed by production run with measurements.
        Optionally saves spin configurations at regular intervals.

        Parameters
        ----------
        n_steps : int, optional
            Number of production Monte Carlo steps (default: 100000).
        equilibration : int, optional
            Number of equilibration steps before measurements (default: 10000).
        measurement_interval : int, optional
            Steps between measurements (default: 10).
        save_configurations : bool, optional
            Whether to save spin configurations (default: False).
        configuration_interval : int, optional
            Steps between saved configurations. If None, defaults to
            10 * measurement_interval.
        progress : bool, optional
            Whether to show progress bars (default: True).

        Returns
        -------
        SimulationResults
            Object containing all measured data and simulation metadata.

        Examples
        --------
        >>> sampler = MetropolisSampler(model, seed=42)
        >>> results = sampler.run(
        ...     n_steps=50000,
        ...     equilibration=5000,
        ...     measurement_interval=10,
        ...     progress=True
        ... )
        >>> print(f"Mean energy: {results.energy_mean:.4f}")

        Notes
        -----
        The simulation proceeds in two phases:

        1. **Equilibration**: The system evolves for `equilibration` steps
           without taking measurements, allowing it to reach thermal
           equilibrium.

        2. **Production**: The system evolves for `n_steps` steps, with
           observables measured every `measurement_interval` steps.
        """
        # Validate parameters
        n_steps = validate_positive_integer(n_steps, "n_steps")
        equilibration = validate_positive_integer(equilibration, "equilibration")
        measurement_interval = validate_positive_integer(
            measurement_interval, "measurement_interval"
        )

        configuration_interval = (
            10 * measurement_interval if configuration_interval is None
            else validate_positive_integer(configuration_interval, "configuration_interval")
        )

        # Calculate number of measurements
        n_measurements = n_steps // measurement_interval

        # Pre-allocate arrays
        energies = np.empty(n_measurements, dtype=np.float64)
        magnetizations = np.empty(n_measurements, dtype=np.float64)
        configurations: List[np.ndarray] = []

        # Reset counters
        self.reset_counters()

        # Record start time
        start_time = time.perf_counter()

        # Equilibration phase
        if equilibration > 0:
            self._equilibrate(equilibration, progress=progress)

        # Reset counters after equilibration for production statistics
        self.reset_counters()

        # Production phase
        measurement_idx = 0
        iterator = range(n_steps)
        if progress:
            iterator = tqdm(
                iterator,
                desc="Sampling",
                unit="step",
            )

        for step in iterator:
            self.step()

            # Take measurement
            if (step + 1) % measurement_interval == 0:
                observables = self._measure()
                energies[measurement_idx] = observables["energy"]
                magnetizations[measurement_idx] = observables["magnetization"]
                measurement_idx += 1

            # Save configuration
            if save_configurations and (step + 1) % configuration_interval == 0:
                configurations.append(self.model.spins.copy())

        # Record end time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Build metadata
        metadata = self._build_metadata(
            n_steps=n_steps,
            equilibration=equilibration,
            measurement_interval=measurement_interval,
            configuration_interval=configuration_interval if save_configurations else None,
            elapsed_time=elapsed_time,
        )

        return SimulationResults(
            energy=energies,
            magnetization=magnetizations,
            metadata=metadata,
            configurations=configurations or None,
        )

    def _build_metadata(
        self,
        n_steps: int,
        equilibration: int,
        measurement_interval: int,
        configuration_interval: Optional[int],
        elapsed_time: float,
    ) -> Dict[str, Any]:
        """Build metadata dictionary for simulation results.

        Parameters
        ----------
        n_steps : int
            Number of production steps.
        equilibration : int
            Number of equilibration steps.
        measurement_interval : int
            Steps between measurements.
        configuration_interval : int or None
            Steps between saved configurations.
        elapsed_time : float
            Total simulation time in seconds.

        Returns
        -------
        dict
            Metadata dictionary.
        """
        # Determine model type and size
        model_type = self.model.__class__.__name__.lower()
        if hasattr(self.model, "shape"):
            size = self.model.shape
        elif hasattr(self.model, "size"):
            size = self.model.size
        else:
            size = self.model.n_spins

        return {
            "model_type": model_type,
            "size": size,
            "n_spins": self.model.n_spins,
            "temperature": self.model.temperature,
            "coupling": self.model.coupling,
            "boundary": self.model.boundary,
            "algorithm": self.__class__.__name__.lower(),
            "n_steps": n_steps,
            "equilibration": equilibration,
            "measurement_interval": measurement_interval,
            "configuration_interval": configuration_interval,
            "seed": self.seed,
            "elapsed_time": elapsed_time,
            "acceptance_rate": self.acceptance_rate,
            "n_accepted": self.n_accepted,
            "n_attempted": self.n_attempted,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"seed={self.seed})"
        )
