"""Wolff cluster algorithm sampler."""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from ising_toolkit.samplers.base import Sampler
from ising_toolkit.models.base import IsingModel
from ising_toolkit.io.results import SimulationResults
from ising_toolkit.utils import (
    validate_positive_integer,
    ConfigurationError,
    DEFAULT_MC_STEPS,
    DEFAULT_EQUILIBRATION,
    DEFAULT_MEASUREMENT_INTERVAL,
)


class WolffSampler(Sampler):
    """Wolff cluster algorithm Monte Carlo sampler.

    Implements the Wolff single-cluster algorithm for the Ising model.
    This algorithm builds and flips clusters of aligned spins, dramatically
    reducing critical slowing down near the phase transition.

    The algorithm:
    1. Select a random seed spin
    2. Build a cluster by adding aligned neighbors with probability
       p_add = 1 - exp(-2βJ)
    3. Flip all spins in the cluster

    Parameters
    ----------
    model : IsingModel
        The Ising model to simulate. Must be 2D or 3D.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    p_add : float
        Probability of adding an aligned neighbor to the cluster.
    cluster_sizes : list
        History of cluster sizes from recent steps.

    Raises
    ------
    ConfigurationError
        If model is 1D (Wolff is inefficient for 1D).

    Examples
    --------
    >>> from ising_toolkit.models import Ising2D
    >>> model = Ising2D(size=64, temperature=2.269)
    >>> model.initialize('random')
    >>> sampler = WolffSampler(model, seed=42)
    >>> results = sampler.run(n_steps=10000, equilibration=1000)

    Notes
    -----
    The Wolff algorithm is particularly efficient near the critical
    temperature where it eliminates critical slowing down. The dynamic
    critical exponent z ≈ 0 for Wolff, compared to z ≈ 2 for Metropolis.

    For 1D systems, use MetropolisSampler instead as there is no phase
    transition and cluster algorithms provide no benefit.

    References
    ----------
    Wolff, U. (1989). "Collective Monte Carlo Updating for Spin Systems".
    Physical Review Letters, 62(4), 361-364.
    """

    def __init__(self, model: IsingModel, seed: Optional[int] = None) -> None:
        """Initialize the Wolff sampler.

        Parameters
        ----------
        model : IsingModel
            The Ising model to simulate. Must be 2D or 3D.
        seed : int, optional
            Random seed for reproducibility.

        Raises
        ------
        ConfigurationError
            If model dimension is 1 (Wolff is not suitable for 1D).
        """
        # Check model dimension
        if model.dimension == 1:
            raise ConfigurationError(
                "WolffSampler requires 2D or 3D model. "
                "Use MetropolisSampler for 1D systems."
            )

        super().__init__(model, seed)

        # Calculate bond probability: p = 1 - exp(-2*beta*J)
        self._update_bond_probability()

        # Track cluster sizes
        self.cluster_sizes: List[int] = []

        # Also seed the model's RNG
        if seed is not None:
            self.model.set_seed(seed)

    def _update_bond_probability(self) -> None:
        """Update the bond addition probability based on current temperature."""
        self.p_add = 1.0 - np.exp(-2.0 * self.model.beta * self.model.coupling)

    def set_temperature(self, temperature: float) -> None:
        """Update temperature and recalculate bond probability.

        Parameters
        ----------
        temperature : float
            New temperature value.
        """
        self.model.set_temperature(temperature)
        self._update_bond_probability()

    def step(self) -> int:
        """Perform one Wolff cluster update.

        Builds a cluster starting from a random seed spin and flips
        all spins in the cluster.

        Returns
        -------
        int
            Size of the flipped cluster.

        Notes
        -----
        The cluster is built using a stack-based depth-first search.
        Each aligned neighbor is added with probability p_add.
        """
        # Select random seed spin
        seed_site = self.model.random_site()
        seed_spin = self._get_spin(seed_site)

        # Build cluster using depth-first search
        cluster: Set[Tuple[int, ...]] = set()
        stack: List[Tuple[int, ...]] = [seed_site]
        cluster.add(seed_site)

        while stack:
            site = stack.pop()

            # Check all neighbors
            for neighbor in self._get_neighbors(site):
                if neighbor not in cluster:
                    neighbor_spin = self._get_spin(neighbor)

                    # Only consider aligned neighbors
                    if neighbor_spin == seed_spin:
                        # Add with probability p_add
                        if self.rng.random() < self.p_add:
                            cluster.add(neighbor)
                            stack.append(neighbor)

        # Flip all spins in cluster
        for site in cluster:
            self.model.flip_spin(site)

        # Update statistics
        cluster_size = len(cluster)
        self.cluster_sizes.append(cluster_size)
        self.n_attempted += 1
        self.n_accepted += 1  # Wolff always accepts

        return cluster_size

    def _get_spin(self, site: Tuple[int, ...]) -> int:
        """Get spin value at a site.

        Parameters
        ----------
        site : tuple
            Site coordinates.

        Returns
        -------
        int
            Spin value (+1 or -1).
        """
        if self.model.dimension == 2:
            return int(self.model.spins[site[0], site[1]])
        else:  # 3D
            return int(self.model.spins[site[0], site[1], site[2]])

    def _get_neighbors(self, site: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get all neighbor sites with periodic boundary conditions.

        Parameters
        ----------
        site : tuple
            Site coordinates.

        Returns
        -------
        list of tuple
            List of neighbor site coordinates.
        """
        L = self.model.size
        neighbors = []

        if self.model.dimension == 2:
            i, j = site
            if self.model.boundary == "periodic":
                neighbors = [
                    ((i + 1) % L, j),
                    ((i - 1) % L, j),
                    (i, (j + 1) % L),
                    (i, (j - 1) % L),
                ]
            else:
                # Fixed boundary - only include valid neighbors
                if i < L - 1:
                    neighbors.append((i + 1, j))
                if i > 0:
                    neighbors.append((i - 1, j))
                if j < L - 1:
                    neighbors.append((i, j + 1))
                if j > 0:
                    neighbors.append((i, j - 1))

        else:  # 3D
            i, j, k = site
            if self.model.boundary == "periodic":
                neighbors = [
                    ((i + 1) % L, j, k),
                    ((i - 1) % L, j, k),
                    (i, (j + 1) % L, k),
                    (i, (j - 1) % L, k),
                    (i, j, (k + 1) % L),
                    (i, j, (k - 1) % L),
                ]
            else:
                if i < L - 1:
                    neighbors.append((i + 1, j, k))
                if i > 0:
                    neighbors.append((i - 1, j, k))
                if j < L - 1:
                    neighbors.append((i, j + 1, k))
                if j > 0:
                    neighbors.append((i, j - 1, k))
                if k < L - 1:
                    neighbors.append((i, j, k + 1))
                if k > 0:
                    neighbors.append((i, j, k - 1))

        return neighbors

    def get_mean_cluster_size(self) -> float:
        """Get mean cluster size from recent steps.

        Returns
        -------
        float
            Mean cluster size, or 0 if no steps performed.
        """
        if not self.cluster_sizes:
            return 0.0
        return float(np.mean(self.cluster_sizes))

    def reset_counters(self) -> None:
        """Reset acceptance counters and cluster size history."""
        super().reset_counters()
        self.cluster_sizes = []

    def run(
        self,
        n_steps: int = DEFAULT_MC_STEPS,
        equilibration: int = DEFAULT_EQUILIBRATION,
        measurement_interval: int = DEFAULT_MEASUREMENT_INTERVAL,
        save_configurations: bool = False,
        configuration_interval: Optional[int] = None,
        progress: bool = True,
    ) -> SimulationResults:
        """Run a complete Monte Carlo simulation with cluster size tracking.

        This overrides the base run() method to also track cluster sizes
        in the results metadata.

        Parameters
        ----------
        n_steps : int, optional
            Number of production Monte Carlo steps (default: 100000).
        equilibration : int, optional
            Number of equilibration steps (default: 10000).
        measurement_interval : int, optional
            Steps between measurements (default: 10).
        save_configurations : bool, optional
            Whether to save spin configurations (default: False).
        configuration_interval : int, optional
            Steps between saved configurations.
        progress : bool, optional
            Whether to show progress bars (default: True).

        Returns
        -------
        SimulationResults
            Results with additional cluster size statistics in metadata.
        """
        # Validate parameters
        n_steps = validate_positive_integer(n_steps, "n_steps")
        equilibration = validate_positive_integer(equilibration, "equilibration")
        measurement_interval = validate_positive_integer(
            measurement_interval, "measurement_interval"
        )

        if configuration_interval is None:
            configuration_interval = 10 * measurement_interval
        else:
            configuration_interval = validate_positive_integer(
                configuration_interval, "configuration_interval"
            )

        # Calculate number of measurements
        n_measurements = n_steps // measurement_interval

        # Pre-allocate arrays
        energies = np.empty(n_measurements, dtype=np.float64)
        magnetizations = np.empty(n_measurements, dtype=np.float64)
        cluster_sizes_record = np.empty(n_measurements, dtype=np.int64)
        configurations: List[np.ndarray] = []

        # Reset counters
        self.reset_counters()

        # Record start time
        import time
        start_time = time.perf_counter()

        # Equilibration phase
        if equilibration > 0:
            self._equilibrate(equilibration, progress=progress)

        # Reset counters after equilibration
        self.reset_counters()

        # Production phase
        measurement_idx = 0
        iterator = range(n_steps)
        if progress:
            iterator = tqdm(iterator, desc="Sampling (Wolff)", unit="step")

        for step_num in iterator:
            cluster_size = self.step()

            # Take measurement
            if (step_num + 1) % measurement_interval == 0:
                observables = self._measure()
                energies[measurement_idx] = observables["energy"]
                magnetizations[measurement_idx] = observables["magnetization"]
                cluster_sizes_record[measurement_idx] = cluster_size
                measurement_idx += 1

            # Save configuration
            if save_configurations and (step_num + 1) % configuration_interval == 0:
                configurations.append(self.model.spins.copy())

        # Record end time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Build metadata with cluster statistics
        metadata = self._build_metadata(
            n_steps=n_steps,
            equilibration=equilibration,
            measurement_interval=measurement_interval,
            configuration_interval=configuration_interval if save_configurations else None,
            elapsed_time=elapsed_time,
        )

        # Add cluster statistics to metadata
        metadata["mean_cluster_size"] = float(np.mean(cluster_sizes_record))
        metadata["std_cluster_size"] = float(np.std(cluster_sizes_record))
        metadata["max_cluster_size"] = int(np.max(cluster_sizes_record))
        metadata["min_cluster_size"] = int(np.min(cluster_sizes_record))

        return SimulationResults(
            energy=energies,
            magnetization=magnetizations,
            metadata=metadata,
            configurations=configurations if configurations else None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WolffSampler("
            f"model={self.model.__class__.__name__}, "
            f"T={self.model.temperature:.4f}, "
            f"p_add={self.p_add:.4f}, "
            f"seed={self.seed})"
        )
