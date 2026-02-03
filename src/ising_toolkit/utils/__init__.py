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
from ising_toolkit.utils.constants import (
    # Physical constants
    DEFAULT_COUPLING,
    DEFAULT_EXTERNAL_FIELD,
    # 2D critical values
    CRITICAL_TEMP_2D,
    CRITICAL_EXPONENT_BETA_2D,
    CRITICAL_EXPONENT_GAMMA_2D,
    CRITICAL_EXPONENT_NU_2D,
    CRITICAL_EXPONENT_ALPHA_2D,
    # 3D critical values
    CRITICAL_TEMP_3D,
    CRITICAL_EXPONENT_BETA_3D,
    CRITICAL_EXPONENT_GAMMA_3D,
    CRITICAL_EXPONENT_NU_3D,
    # Simulation defaults
    DEFAULT_MC_STEPS,
    DEFAULT_EQUILIBRATION,
    DEFAULT_MEASUREMENT_INTERVAL,
    # Valid values
    VALID_BOUNDARIES,
    VALID_INITIAL_STATES,
    VALID_ALGORITHMS,
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
    # Physical constants
    "DEFAULT_COUPLING",
    "DEFAULT_EXTERNAL_FIELD",
    # 2D critical values
    "CRITICAL_TEMP_2D",
    "CRITICAL_EXPONENT_BETA_2D",
    "CRITICAL_EXPONENT_GAMMA_2D",
    "CRITICAL_EXPONENT_NU_2D",
    "CRITICAL_EXPONENT_ALPHA_2D",
    # 3D critical values
    "CRITICAL_TEMP_3D",
    "CRITICAL_EXPONENT_BETA_3D",
    "CRITICAL_EXPONENT_GAMMA_3D",
    "CRITICAL_EXPONENT_NU_3D",
    # Simulation defaults
    "DEFAULT_MC_STEPS",
    "DEFAULT_EQUILIBRATION",
    "DEFAULT_MEASUREMENT_INTERVAL",
    # Valid values
    "VALID_BOUNDARIES",
    "VALID_INITIAL_STATES",
    "VALID_ALGORITHMS",
]
