"""General utility functions."""

from ising_toolkit.utils.exceptions import (
    IsingError,
    ConfigurationError,
    SimulationError,
    AnalysisError,
    FileFormatError,
)
from ising_toolkit.utils.validation import (
    validate_positive_integer,
    validate_positive_float,
    validate_temperature,
    validate_lattice_size,
    validate_boundary_condition,
    validate_initial_state,
)

__all__ = [
    # Exceptions
    "IsingError",
    "ConfigurationError",
    "SimulationError",
    "AnalysisError",
    "FileFormatError",
    # Validation
    "validate_positive_integer",
    "validate_positive_float",
    "validate_temperature",
    "validate_lattice_size",
    "validate_boundary_condition",
    "validate_initial_state",
]
