"""Container classes for simulation results with memory-efficient storage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from ising_toolkit.utils import FileFormatError
from ising_toolkit.io.compression import (
    pack_spins,
    unpack_spins,
)


class SimulationResults:
    """Container for Monte Carlo simulation results.

    Stores time series data from Ising model simulations along with
    metadata about the simulation parameters. Supports saving to and
    loading from HDF5 format for efficient storage and retrieval.

    Parameters
    ----------
    energy : np.ndarray
        Energy time series from the simulation.
    magnetization : np.ndarray
        Magnetization time series from the simulation.
    metadata : dict
        Dictionary containing simulation parameters and metadata.

    Attributes
    ----------
    energy : np.ndarray
        Energy time series.
    magnetization : np.ndarray
        Magnetization time series.
    abs_magnetization : np.ndarray
        Absolute magnetization time series.
    configurations : list
        Optional list of saved spin configuration snapshots.
    metadata : dict
        Simulation parameters and metadata.

    Examples
    --------
    >>> results = SimulationResults(
    ...     energy=energy_data,
    ...     magnetization=mag_data,
    ...     metadata={'model_type': 'ising2d', 'size': 32, 'temperature': 2.269}
    ... )
    >>> results.save('simulation.h5')
    >>> loaded = SimulationResults.load('simulation.h5')
    """

    def __init__(
        self,
        energy: np.ndarray,
        magnetization: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        configurations: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Initialize simulation results.

        Parameters
        ----------
        energy : np.ndarray
            Energy time series.
        magnetization : np.ndarray
            Magnetization time series.
        metadata : dict, optional
            Simulation metadata. Will be initialized to empty dict if None.
        configurations : list, optional
            List of spin configuration snapshots.
        """
        self.energy = np.asarray(energy, dtype=np.float64)
        self.magnetization = np.asarray(magnetization, dtype=np.float64)
        self.abs_magnetization = np.abs(self.magnetization)
        self.metadata = metadata if metadata is not None else {}
        self.configurations = configurations if configurations is not None else []

        # Cache for computed statistics
        self._stats_cache: Dict[str, Any] = {}

    def _clear_cache(self) -> None:
        """Clear the statistics cache."""
        self._stats_cache = {}

    @property
    def n_samples(self) -> int:
        """int: Number of samples in the time series."""
        return len(self.energy)

    # =========================================================================
    # Energy statistics
    # =========================================================================

    @property
    def energy_mean(self) -> float:
        """float: Mean energy."""
        if "energy_mean" not in self._stats_cache:
            self._stats_cache["energy_mean"] = float(np.mean(self.energy))
        return self._stats_cache["energy_mean"]

    @property
    def energy_std(self) -> float:
        """float: Standard deviation of energy."""
        if "energy_std" not in self._stats_cache:
            self._stats_cache["energy_std"] = float(np.std(self.energy, ddof=1))
        return self._stats_cache["energy_std"]

    @property
    def energy_err(self) -> float:
        """float: Standard error of mean energy (bootstrap estimate)."""
        if "energy_err" not in self._stats_cache:
            self._stats_cache["energy_err"] = self._bootstrap_error(self.energy)
        return self._stats_cache["energy_err"]

    # =========================================================================
    # Magnetization statistics
    # =========================================================================

    @property
    def magnetization_mean(self) -> float:
        """float: Mean magnetization."""
        if "magnetization_mean" not in self._stats_cache:
            self._stats_cache["magnetization_mean"] = float(np.mean(self.magnetization))
        return self._stats_cache["magnetization_mean"]

    @property
    def magnetization_std(self) -> float:
        """float: Standard deviation of magnetization."""
        if "magnetization_std" not in self._stats_cache:
            self._stats_cache["magnetization_std"] = float(
                np.std(self.magnetization, ddof=1)
            )
        return self._stats_cache["magnetization_std"]

    # =========================================================================
    # Absolute magnetization statistics
    # =========================================================================

    @property
    def abs_magnetization_mean(self) -> float:
        """float: Mean absolute magnetization."""
        if "abs_magnetization_mean" not in self._stats_cache:
            self._stats_cache["abs_magnetization_mean"] = float(
                np.mean(self.abs_magnetization)
            )
        return self._stats_cache["abs_magnetization_mean"]

    @property
    def abs_magnetization_std(self) -> float:
        """float: Standard deviation of absolute magnetization."""
        if "abs_magnetization_std" not in self._stats_cache:
            self._stats_cache["abs_magnetization_std"] = float(
                np.std(self.abs_magnetization, ddof=1)
            )
        return self._stats_cache["abs_magnetization_std"]

    # =========================================================================
    # Statistical helpers
    # =========================================================================

    def _bootstrap_error(
        self,
        data: np.ndarray,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate standard error using bootstrap resampling.

        Parameters
        ----------
        data : np.ndarray
            Data array to estimate error for.
        n_bootstrap : int, optional
            Number of bootstrap samples (default: 1000).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        float
            Bootstrap estimate of standard error.
        """
        rng = np.random.default_rng(seed)
        n = len(data)

        if n == 0:
            return 0.0

        bootstrap_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)

        return float(np.std(bootstrap_means, ddof=1))

    # =========================================================================
    # I/O methods
    # =========================================================================

    def save(self, filepath: Union[str, Path]) -> None:
        """Save results to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to save the HDF5 file.

        Notes
        -----
        The file structure is:
        - /energy: energy time series
        - /magnetization: magnetization time series
        - /configurations/config_N: saved configurations
        - /metadata: attributes with simulation parameters
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "w") as f:
            # Save time series data
            f.create_dataset("energy", data=self.energy, compression="gzip")
            f.create_dataset("magnetization", data=self.magnetization, compression="gzip")

            # Save configurations if any
            if self.configurations:
                config_grp = f.create_group("configurations")
                for i, config in enumerate(self.configurations):
                    config_grp.create_dataset(
                        f"config_{i:06d}",
                        data=config,
                        compression="gzip"
                    )

            # Save metadata as attributes
            meta_grp = f.create_group("metadata")
            for key, value in self.metadata.items():
                if value is None:
                    meta_grp.attrs[key] = "None"
                elif isinstance(value, (list, tuple)):
                    meta_grp.attrs[key] = list(value)
                else:
                    meta_grp.attrs[key] = value

            # Add file metadata
            f.attrs["created"] = datetime.now().isoformat()
            f.attrs["format_version"] = "1.0"

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SimulationResults":
        """Load results from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.

        Returns
        -------
        SimulationResults
            Loaded simulation results.

        Raises
        ------
        FileFormatError
            If the file format is invalid or unreadable.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileFormatError(f"File not found: {filepath}")

        try:
            with h5py.File(filepath, "r") as f:
                # Load time series
                energy = np.array(f["energy"])
                magnetization = np.array(f["magnetization"])

                # Load configurations if present
                configurations = []
                if "configurations" in f:
                    config_grp = f["configurations"]
                    for key in sorted(config_grp.keys()):
                        configurations.append(np.array(config_grp[key]))

                # Load metadata
                metadata = {}
                if "metadata" in f:
                    for key, value in f["metadata"].attrs.items():
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        if isinstance(value, str) and value == "None":
                            value = None
                        elif isinstance(value, np.ndarray):
                            value = tuple(value.tolist())
                        metadata[key] = value

                return cls(
                    energy=energy,
                    magnetization=magnetization,
                    metadata=metadata,
                    configurations=configurations if configurations else None,
                )

        except Exception as e:
            if isinstance(e, FileFormatError):
                raise
            raise FileFormatError(f"Failed to load file: {e}")

    # =========================================================================
    # Conversion methods
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series to pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: step, energy, magnetization, abs_magnetization
        """
        return pd.DataFrame({
            "step": np.arange(self.n_samples),
            "energy": self.energy,
            "magnetization": self.magnetization,
            "abs_magnetization": self.abs_magnetization,
        })

    def get_statistics(self) -> Dict[str, float]:
        """Get all computed statistics as a dictionary.

        Returns
        -------
        dict
            Dictionary containing all mean, std, and error values.
        """
        return {
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "energy_err": self.energy_err,
            "magnetization_mean": self.magnetization_mean,
            "magnetization_std": self.magnetization_std,
            "abs_magnetization_mean": self.abs_magnetization_mean,
            "abs_magnetization_std": self.abs_magnetization_std,
            "n_samples": self.n_samples,
        }

    # =========================================================================
    # String representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        model_type = self.metadata.get("model_type", "unknown")
        size = self.metadata.get("size", "?")
        temperature = self.metadata.get("temperature", "?")

        lines = [
            "SimulationResults(",
            f"  model_type='{model_type}',",
            f"  size={size},",
            f"  temperature={temperature},",
            f"  n_samples={self.n_samples},",
            f"  energy_mean={self.energy_mean:.4f} ± {self.energy_err:.4f},",
            f"  |M|_mean={self.abs_magnetization_mean:.4f}",
            ")",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def save_compressed(
        self,
        filepath: Union[str, Path],
        compress_configurations: bool = True,
    ) -> None:
        """Save results to HDF5 file with optional spin compression.

        Parameters
        ----------
        filepath : str or Path
            Path to save the HDF5 file.
        compress_configurations : bool, optional
            If True, pack spin configurations using bit-packing (8x compression).
            Default is True.

        Notes
        -----
        Uses bit-packing for configurations: 8 spins per byte instead of
        1 spin per byte, reducing storage by approximately 8x.
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "w") as f:
            # Save time series data with gzip compression
            f.create_dataset("energy", data=self.energy, compression="gzip")
            f.create_dataset("magnetization", data=self.magnetization, compression="gzip")

            # Save configurations with optional bit-packing
            if self.configurations:
                config_grp = f.create_group("configurations")
                config_grp.attrs["compressed"] = compress_configurations

                for i, config in enumerate(self.configurations):
                    if compress_configurations:
                        packed = pack_spins(config)
                        config_grp.create_dataset(
                            f"config_{i:06d}",
                            data=packed,
                            compression="gzip"
                        )
                    else:
                        config_grp.create_dataset(
                            f"config_{i:06d}",
                            data=config,
                            compression="gzip"
                        )

                # Store original shape for unpacking
                if self.configurations:
                    config_grp.attrs["shape"] = self.configurations[0].shape

            # Save metadata as attributes
            meta_grp = f.create_group("metadata")
            for key, value in self.metadata.items():
                if value is None:
                    meta_grp.attrs[key] = "None"
                elif isinstance(value, (list, tuple)):
                    meta_grp.attrs[key] = list(value)
                else:
                    meta_grp.attrs[key] = value

            # Add file metadata
            f.attrs["created"] = datetime.now().isoformat()
            f.attrs["format_version"] = "1.1"
            f.attrs["compressed_configs"] = compress_configurations

    def add_configuration(
        self,
        config: np.ndarray,
        max_configurations: Optional[int] = None,
        decimation: int = 1,
    ) -> bool:
        """Add a configuration with optional limiting.

        Parameters
        ----------
        config : np.ndarray
            Spin configuration to add.
        max_configurations : int, optional
            Maximum configurations to keep. Oldest are dropped.
        decimation : int, optional
            Only add every Nth configuration.

        Returns
        -------
        bool
            True if configuration was added.
        """
        # Track call count for decimation
        if not hasattr(self, '_config_count'):
            self._config_count = 0
        self._config_count += 1

        # Apply decimation
        if decimation > 1 and (self._config_count - 1) % decimation != 0:
            return False

        # Add configuration
        self.configurations.append(config.copy())

        # Apply max limit
        if max_configurations is not None:
            while len(self.configurations) > max_configurations:
                self.configurations.pop(0)

        return True


class LazyResults:
    """Lazy-loading wrapper for large HDF5 result files.

    Only loads data when accessed, minimizing memory usage for large files.
    Supports context manager protocol for automatic cleanup.

    Parameters
    ----------
    filepath : str or Path
        Path to the HDF5 file.

    Attributes
    ----------
    filepath : Path
        Path to the HDF5 file.
    metadata : dict
        Simulation metadata (loaded immediately, small).
    n_samples : int
        Number of samples in the time series.
    n_configurations : int
        Number of stored configurations.

    Examples
    --------
    >>> with LazyResults('large_simulation.h5') as results:
    ...     print(f"Samples: {results.n_samples}")
    ...     print(f"Mean energy: {results.energy_mean}")
    ...     # Only loads energy data when accessed
    ...     for i, config in results.iter_configurations():
    ...         process(config)  # Configurations loaded one at a time

    >>> # Or use without context manager
    >>> results = LazyResults('simulation.h5')
    >>> energy = results.energy  # Loads on first access
    >>> results.close()
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileFormatError(f"File not found: {self.filepath}")

        self._file: Optional[h5py.File] = None
        self._energy: Optional[np.ndarray] = None
        self._magnetization: Optional[np.ndarray] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._config_shape: Optional[Tuple[int, ...]] = None
        self._configs_compressed: bool = False

        # Load metadata immediately (small)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from file."""
        with h5py.File(self.filepath, "r") as f:
            self._n_samples = len(f["energy"])

            # Count configurations
            if "configurations" in f:
                self._n_configurations = len(f["configurations"])
                config_grp = f["configurations"]
                self._configs_compressed = config_grp.attrs.get("compressed", False)
                if "shape" in config_grp.attrs:
                    self._config_shape = tuple(config_grp.attrs["shape"])
            else:
                self._n_configurations = 0

            # Load metadata
            self._metadata = {}
            if "metadata" in f:
                for key, value in f["metadata"].attrs.items():
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    if isinstance(value, str) and value == "None":
                        value = None
                    elif isinstance(value, np.ndarray):
                        value = tuple(value.tolist())
                    self._metadata[key] = value

    def _ensure_open(self) -> h5py.File:
        """Ensure file is open, open if needed."""
        if self._file is None:
            self._file = h5py.File(self.filepath, "r")
        return self._file

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "LazyResults":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def n_samples(self) -> int:
        """int: Number of samples in time series."""
        return self._n_samples

    @property
    def n_configurations(self) -> int:
        """int: Number of stored configurations."""
        return self._n_configurations

    @property
    def metadata(self) -> Dict[str, Any]:
        """dict: Simulation metadata."""
        return self._metadata.copy()

    @property
    def energy(self) -> np.ndarray:
        """np.ndarray: Energy time series (loaded on first access)."""
        if self._energy is None:
            f = self._ensure_open()
            self._energy = np.array(f["energy"])
        return self._energy

    @property
    def magnetization(self) -> np.ndarray:
        """np.ndarray: Magnetization time series (loaded on first access)."""
        if self._magnetization is None:
            f = self._ensure_open()
            self._magnetization = np.array(f["magnetization"])
        return self._magnetization

    @property
    def abs_magnetization(self) -> np.ndarray:
        """np.ndarray: Absolute magnetization time series."""
        return np.abs(self.magnetization)

    # Statistics computed on-demand
    @property
    def energy_mean(self) -> float:
        """float: Mean energy."""
        return float(np.mean(self.energy))

    @property
    def energy_std(self) -> float:
        """float: Standard deviation of energy."""
        return float(np.std(self.energy, ddof=1))

    @property
    def magnetization_mean(self) -> float:
        """float: Mean magnetization."""
        return float(np.mean(self.magnetization))

    @property
    def abs_magnetization_mean(self) -> float:
        """float: Mean absolute magnetization."""
        return float(np.mean(np.abs(self.magnetization)))

    def get_configuration(self, index: int) -> np.ndarray:
        """Load a single configuration by index.

        Parameters
        ----------
        index : int
            Configuration index (0-based).

        Returns
        -------
        np.ndarray
            Spin configuration.

        Raises
        ------
        IndexError
            If index is out of range.
        """
        if index < 0 or index >= self._n_configurations:
            raise IndexError(
                f"Configuration index {index} out of range "
                f"[0, {self._n_configurations})"
            )

        f = self._ensure_open()
        config_grp = f["configurations"]
        key = f"config_{index:06d}"

        data = np.array(config_grp[key])

        if self._configs_compressed:
            return unpack_spins(data, self._config_shape)
        return data

    def iter_configurations(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        step: int = 1,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over configurations without loading all into memory.

        Parameters
        ----------
        start : int, optional
            Starting index (default: 0).
        stop : int, optional
            Stopping index (default: n_configurations).
        step : int, optional
            Step size (default: 1).

        Yields
        ------
        tuple of (int, np.ndarray)
            Index and configuration array.

        Examples
        --------
        >>> with LazyResults('simulation.h5') as results:
        ...     for i, config in results.iter_configurations(step=10):
        ...         # Process every 10th configuration
        ...         analyze(config)
        """
        if stop is None:
            stop = self._n_configurations

        for i in range(start, stop, step):
            yield i, self.get_configuration(i)

    def to_simulation_results(self) -> "SimulationResults":
        """Convert to full SimulationResults (loads all data).

        Returns
        -------
        SimulationResults
            Fully loaded results object.

        Warning
        -------
        This loads all data into memory. For large files, prefer
        using lazy access methods.
        """
        configurations = None
        if self._n_configurations > 0:
            configurations = [self.get_configuration(i) for i in range(self._n_configurations)]

        return SimulationResults(
            energy=self.energy.copy(),
            magnetization=self.magnetization.copy(),
            metadata=self.metadata,
            configurations=configurations,
        )

    def get_memory_estimate(self) -> Dict[str, float]:
        """Estimate memory usage if fully loaded.

        Returns
        -------
        dict
            Dictionary with 'timeseries_mb', 'configurations_mb', 'total_mb'.
        """
        # Time series: float64
        ts_bytes = self._n_samples * 8 * 2  # energy + magnetization

        # Configurations: int8
        if self._n_configurations > 0 and self._config_shape is not None:
            config_bytes = self._n_configurations * int(np.prod(self._config_shape))
        else:
            config_bytes = 0

        return {
            'timeseries_mb': ts_bytes / (1024 * 1024),
            'configurations_mb': config_bytes / (1024 * 1024),
            'total_mb': (ts_bytes + config_bytes) / (1024 * 1024),
        }

    def __repr__(self) -> str:
        mem = self.get_memory_estimate()
        return (
            f"LazyResults(\n"
            f"  filepath='{self.filepath}',\n"
            f"  n_samples={self._n_samples},\n"
            f"  n_configurations={self._n_configurations},\n"
            f"  estimated_memory={mem['total_mb']:.1f}MB\n"
            f")"
        )
