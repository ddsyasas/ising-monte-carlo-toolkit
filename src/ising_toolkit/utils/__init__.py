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
from ising_toolkit.utils.numba_kernels import (
    NUMBA_AVAILABLE,
    # Precomputation
    precompute_acceptance_probs_1d,
    precompute_acceptance_probs_2d,
    precompute_acceptance_probs_3d,
    # High-level wrappers
    metropolis_sweep_1d,
    metropolis_sweep_2d,
    metropolis_sweep_3d,
    wolff_step_1d,
    wolff_step_2d,
    wolff_step_3d,
    # Benchmarking
    run_benchmark,
    print_benchmark_results,
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
    # Numba kernels
    "NUMBA_AVAILABLE",
    "precompute_acceptance_probs_1d",
    "precompute_acceptance_probs_2d",
    "precompute_acceptance_probs_3d",
    "metropolis_sweep_1d",
    "metropolis_sweep_2d",
    "metropolis_sweep_3d",
    "wolff_step_1d",
    "wolff_step_2d",
    "wolff_step_3d",
    "run_benchmark",
    "print_benchmark_results",
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
