"""Validation functions for simulation parameters."""

from typing import Union

from ising_toolkit.utils.exceptions import ConfigurationError

__all__ = [
    "validate_positive_integer",
    "validate_positive_float",
    "validate_temperature",
    "validate_lattice_size",
    "validate_boundary_condition",
    "validate_initial_state",
]

VALID_BOUNDARY_CONDITIONS = ("periodic", "fixed")
VALID_INITIAL_STATES = ("random", "up", "down", "checkerboard")


def validate_positive_integer(value: Union[int, float], name: str) -> int:
    """Validate that a value is a positive integer.

    Parameters
    ----------
    value : int or float
        The value to validate. Floats that represent whole numbers
        (e.g., 5.0) are accepted and converted to int.
    name : str
        The name of the parameter, used in error messages.

    Returns
    -------
    int
        The validated positive integer.

    Raises
    ------
    ConfigurationError
        If the value is not a positive integer.

    Examples
    --------
    >>> validate_positive_integer(5, "steps")
    5
    >>> validate_positive_integer(-1, "steps")
    ConfigurationError: steps must be a positive integer, got -1
    """
    # Check if it's a number type
    if not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"{name} must be a positive integer, got {type(value).__name__}"
        )

    # Check for special float values
    if isinstance(value, float):
        if value != value:  # NaN check
            raise ConfigurationError(f"{name} must be a positive integer, got nan")
        if value == float("inf") or value == float("-inf"):
            raise ConfigurationError(f"{name} must be a positive integer, got {value}")

    # Check if it's a whole number
    if isinstance(value, float) and not value.is_integer():
        raise ConfigurationError(
            f"{name} must be a positive integer, got {value}"
        )

    int_value = int(value)

    # Check if positive
    if int_value <= 0:
        raise ConfigurationError(
            f"{name} must be a positive integer, got {int_value}"
        )

    return int_value


def validate_positive_float(value: Union[int, float], name: str) -> float:
    """Validate that a value is a positive float.

    Parameters
    ----------
    value : int or float
        The value to validate. Integers are accepted and converted to float.
    name : str
        The name of the parameter, used in error messages.

    Returns
    -------
    float
        The validated positive float.

    Raises
    ------
    ConfigurationError
        If the value is not a positive number.

    Examples
    --------
    >>> validate_positive_float(2.5, "coupling")
    2.5
    >>> validate_positive_float(0.0, "coupling")
    ConfigurationError: coupling must be a positive number, got 0.0
    """
    # Check if it's a number type
    if not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"{name} must be a positive number, got {type(value).__name__}"
        )

    # Exclude booleans (they are a subclass of int in Python)
    if isinstance(value, bool):
        raise ConfigurationError(
            f"{name} must be a positive number, got bool"
        )

    float_value = float(value)

    # Check for NaN
    if float_value != float_value:
        raise ConfigurationError(f"{name} must be a positive number, got nan")

    # Check for infinity
    if float_value == float("inf") or float_value == float("-inf"):
        raise ConfigurationError(f"{name} must be a positive number, got {float_value}")

    # Check if positive
    if float_value <= 0:
        raise ConfigurationError(
            f"{name} must be a positive number, got {float_value}"
        )

    return float_value


def validate_temperature(value: Union[int, float]) -> float:
    """Validate that a temperature value is positive.

    Temperature in the Ising model must be strictly positive to ensure
    well-defined Boltzmann statistics.

    Parameters
    ----------
    value : int or float
        The temperature value to validate.

    Returns
    -------
    float
        The validated positive temperature.

    Raises
    ------
    ConfigurationError
        If the temperature is not a positive number.

    Examples
    --------
    >>> validate_temperature(2.269)
    2.269
    >>> validate_temperature(0)
    ConfigurationError: temperature must be a positive number, got 0.0
    """
    return validate_positive_float(value, "temperature")


def validate_lattice_size(value: Union[int, float], min_size: int = 2) -> int:
    """Validate that a lattice size is a valid positive integer.

    Parameters
    ----------
    value : int or float
        The lattice size to validate.
    min_size : int, optional
        The minimum allowed lattice size (default is 2).

    Returns
    -------
    int
        The validated lattice size.

    Raises
    ------
    ConfigurationError
        If the lattice size is not an integer >= min_size.

    Examples
    --------
    >>> validate_lattice_size(16)
    16
    >>> validate_lattice_size(1)
    ConfigurationError: lattice_size must be at least 2, got 1
    """
    # First validate it's a positive integer
    int_value = validate_positive_integer(value, "lattice_size")

    # Check minimum size
    if int_value < min_size:
        raise ConfigurationError(
            f"lattice_size must be at least {min_size}, got {int_value}"
        )

    return int_value


def validate_boundary_condition(value: str) -> str:
    """Validate that a boundary condition is supported.

    Parameters
    ----------
    value : str
        The boundary condition to validate.

    Returns
    -------
    str
        The validated boundary condition (lowercase).

    Raises
    ------
    ConfigurationError
        If the boundary condition is not one of the supported values.

    Notes
    -----
    Supported boundary conditions:
    - 'periodic': Periodic boundary conditions (toroidal topology)
    - 'fixed': Fixed boundary conditions (spins at edges are fixed)

    Examples
    --------
    >>> validate_boundary_condition("periodic")
    'periodic'
    >>> validate_boundary_condition("open")
    ConfigurationError: boundary_condition must be one of ('periodic', 'fixed'), got 'open'
    """
    if not isinstance(value, str):
        raise ConfigurationError(
            f"boundary_condition must be a string, got {type(value).__name__}"
        )

    value_lower = value.lower()

    if value_lower not in VALID_BOUNDARY_CONDITIONS:
        raise ConfigurationError(
            f"boundary_condition must be one of {VALID_BOUNDARY_CONDITIONS}, "
            f"got '{value}'"
        )

    return value_lower


def validate_initial_state(value: str) -> str:
    """Validate that an initial state configuration is supported.

    Parameters
    ----------
    value : str
        The initial state configuration to validate.

    Returns
    -------
    str
        The validated initial state (lowercase).

    Raises
    ------
    ConfigurationError
        If the initial state is not one of the supported values.

    Notes
    -----
    Supported initial states:
    - 'random': Random spin configuration (+1 or -1 with equal probability)
    - 'up': All spins pointing up (+1)
    - 'down': All spins pointing down (-1)
    - 'checkerboard': Alternating pattern of +1 and -1 spins

    Examples
    --------
    >>> validate_initial_state("random")
    'random'
    >>> validate_initial_state("antiferromagnetic")
    ConfigurationError: initial_state must be one of ('random', 'up', 'down', 'checkerboard'), got 'antiferromagnetic'
    """
    if not isinstance(value, str):
        raise ConfigurationError(
            f"initial_state must be a string, got {type(value).__name__}"
        )

    value_lower = value.lower()

    if value_lower not in VALID_INITIAL_STATES:
        raise ConfigurationError(
            f"initial_state must be one of {VALID_INITIAL_STATES}, "
            f"got '{value}'"
        )

    return value_lower
