"""Container classes for simulation results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from ising_toolkit.utils import FileFormatError


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
            f"SimulationResults(",
            f"  model_type='{model_type}',",
            f"  size={size},",
            f"  temperature={temperature},",
            f"  n_samples={self.n_samples},",
            f"  energy_mean={self.energy_mean:.4f} ± {self.energy_err:.4f},",
            f"  |M|_mean={self.abs_magnetization_mean:.4f}",
            f")",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
