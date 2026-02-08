"""Input/output utilities for saving and loading simulation data."""

from ising_toolkit.io.results import SimulationResults, LazyResults

from ising_toolkit.io.compression import (
    pack_spins,
    unpack_spins,
    pack_configurations,
    unpack_configurations,
    ConfigurationBuffer,
    get_compression_ratio,
    estimate_storage_size,
)

from ising_toolkit.io.config import (
    # Dataclasses
    SimulationConfig,
    AlgorithmConfig,
    RunConfig,
    OutputConfig,
    Config,
    # Functions
    load_config,
    merge_configs,
    create_sweep_configs,
    create_size_sweep_configs,
    get_default_config,
)

__all__ = [
    # Results
    "SimulationResults",
    "LazyResults",
    # Compression
    "pack_spins",
    "unpack_spins",
    "pack_configurations",
    "unpack_configurations",
    "ConfigurationBuffer",
    "get_compression_ratio",
    "estimate_storage_size",
    # Config dataclasses
    "SimulationConfig",
    "AlgorithmConfig",
    "RunConfig",
    "OutputConfig",
    "Config",
    # Config functions
    "load_config",
    "merge_configs",
    "create_sweep_configs",
    "create_size_sweep_configs",
    "get_default_config",
]
