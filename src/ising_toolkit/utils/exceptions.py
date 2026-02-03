"""Custom exception classes for the Ising Monte Carlo toolkit."""

__all__ = [
    "IsingError",
    "ConfigurationError",
    "SimulationError",
    "AnalysisError",
    "FileFormatError",
]


class IsingError(Exception):
    """Base exception for all Ising Monte Carlo toolkit errors.

    All custom exceptions in this project inherit from this class,
    allowing users to catch all toolkit-specific errors with a single
    except clause.
    """

    pass


class ConfigurationError(IsingError):
    """Raised when invalid configuration parameters are provided.

    This includes invalid lattice sizes, temperatures, coupling constants,
    or any other simulation parameters that fall outside acceptable ranges
    or are of incorrect types.
    """

    pass


class SimulationError(IsingError):
    """Raised when an error occurs during Monte Carlo simulation.

    This includes failures in the sampling algorithms, convergence issues,
    numerical instabilities, or other runtime errors specific to the
    simulation process.
    """

    pass


class AnalysisError(IsingError):
    """Raised when an error occurs during data analysis.

    This includes errors in computing observables, statistical analysis
    failures, insufficient data for analysis, or invalid analysis
    parameters.
    """

    pass


class FileFormatError(IsingError):
    """Raised when encountering an invalid or unsupported file format.

    This includes errors when reading or writing simulation data,
    configuration files, or checkpoint files that do not conform
    to expected formats.
    """

    pass
