"""Tests for validation functions."""

import pytest

from ising_toolkit.utils.validation import (
    validate_positive_integer,
    validate_positive_float,
    validate_temperature,
    validate_lattice_size,
    validate_boundary_condition,
    validate_initial_state,
)
from ising_toolkit.utils.exceptions import ConfigurationError


class TestValidatePositiveInteger:
    """Tests for validate_positive_integer function."""

    def test_valid_integer(self):
        """Test that valid positive integers are accepted."""
        assert validate_positive_integer(1, "test") == 1
        assert validate_positive_integer(10, "test") == 10
        assert validate_positive_integer(1000000, "test") == 1000000

    def test_float_whole_number(self):
        """Test that floats representing whole numbers are accepted."""
        assert validate_positive_integer(5.0, "test") == 5
        assert validate_positive_integer(100.0, "test") == 100

    def test_zero_raises(self):
        """Test that zero raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(0, "test")

    def test_negative_raises(self):
        """Test that negative integers raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(-1, "test")
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(-100, "test")

    def test_non_integer_float_raises(self):
        """Test that non-integer floats raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(3.14, "test")
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(0.5, "test")

    def test_string_raises(self):
        """Test that strings raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer("5", "test")

    def test_none_raises(self):
        """Test that None raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(None, "test")

    def test_nan_raises(self):
        """Test that NaN raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(float("nan"), "test")

    def test_inf_raises(self):
        """Test that infinity raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(float("inf"), "test")
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_positive_integer(float("-inf"), "test")

    def test_error_message_includes_name(self):
        """Test that error message includes parameter name."""
        with pytest.raises(ConfigurationError, match="my_param"):
            validate_positive_integer(-1, "my_param")


class TestValidatePositiveFloat:
    """Tests for validate_positive_float function."""

    def test_valid_float(self):
        """Test that valid positive floats are accepted."""
        assert validate_positive_float(1.0, "test") == 1.0
        assert validate_positive_float(3.14, "test") == 3.14
        assert validate_positive_float(0.001, "test") == 0.001

    def test_valid_integer(self):
        """Test that positive integers are accepted and converted."""
        assert validate_positive_float(5, "test") == 5.0
        assert isinstance(validate_positive_float(5, "test"), float)

    def test_zero_raises(self):
        """Test that zero raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(0.0, "test")
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(0, "test")

    def test_negative_raises(self):
        """Test that negative numbers raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(-1.0, "test")
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(-0.001, "test")

    def test_string_raises(self):
        """Test that strings raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float("3.14", "test")

    def test_none_raises(self):
        """Test that None raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(None, "test")

    def test_bool_raises(self):
        """Test that booleans raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(True, "test")
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(False, "test")

    def test_nan_raises(self):
        """Test that NaN raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(float("nan"), "test")

    def test_inf_raises(self):
        """Test that infinity raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(float("inf"), "test")
        with pytest.raises(ConfigurationError, match="must be a positive number"):
            validate_positive_float(float("-inf"), "test")

    def test_error_message_includes_name(self):
        """Test that error message includes parameter name."""
        with pytest.raises(ConfigurationError, match="coupling"):
            validate_positive_float(-1.0, "coupling")


class TestValidateTemperature:
    """Tests for validate_temperature function."""

    def test_valid_temperature(self):
        """Test that valid temperatures are accepted."""
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.269) == 2.269  # Critical temperature
        assert validate_temperature(0.1) == 0.1

    def test_integer_temperature(self):
        """Test that integer temperatures are converted to float."""
        assert validate_temperature(2) == 2.0
        assert isinstance(validate_temperature(2), float)

    def test_zero_raises(self):
        """Test that zero temperature raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="temperature"):
            validate_temperature(0)

    def test_negative_raises(self):
        """Test that negative temperature raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="temperature"):
            validate_temperature(-1.0)


class TestValidateLatticeSize:
    """Tests for validate_lattice_size function."""

    def test_valid_size(self):
        """Test that valid lattice sizes are accepted."""
        assert validate_lattice_size(2) == 2
        assert validate_lattice_size(16) == 16
        assert validate_lattice_size(100) == 100

    def test_float_whole_number(self):
        """Test that float whole numbers are accepted."""
        assert validate_lattice_size(10.0) == 10

    def test_custom_min_size(self):
        """Test validation with custom minimum size."""
        assert validate_lattice_size(5, min_size=5) == 5
        assert validate_lattice_size(10, min_size=5) == 10

    def test_below_default_min_raises(self):
        """Test that size below default minimum raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="at least 2"):
            validate_lattice_size(1)

    def test_below_custom_min_raises(self):
        """Test that size below custom minimum raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="at least 4"):
            validate_lattice_size(3, min_size=4)

    def test_zero_raises(self):
        """Test that zero raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_lattice_size(0)

    def test_negative_raises(self):
        """Test that negative size raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_lattice_size(-10)

    def test_non_integer_raises(self):
        """Test that non-integer raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a positive integer"):
            validate_lattice_size(10.5)


class TestValidateBoundaryCondition:
    """Tests for validate_boundary_condition function."""

    def test_periodic(self):
        """Test that 'periodic' is accepted."""
        assert validate_boundary_condition("periodic") == "periodic"

    def test_fixed(self):
        """Test that 'fixed' is accepted."""
        assert validate_boundary_condition("fixed") == "fixed"

    def test_case_insensitive(self):
        """Test that validation is case-insensitive."""
        assert validate_boundary_condition("PERIODIC") == "periodic"
        assert validate_boundary_condition("Periodic") == "periodic"
        assert validate_boundary_condition("FIXED") == "fixed"

    def test_invalid_string_raises(self):
        """Test that invalid boundary conditions raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="boundary_condition"):
            validate_boundary_condition("open")
        with pytest.raises(ConfigurationError, match="boundary_condition"):
            validate_boundary_condition("free")

    def test_non_string_raises(self):
        """Test that non-strings raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a string"):
            validate_boundary_condition(1)
        with pytest.raises(ConfigurationError, match="must be a string"):
            validate_boundary_condition(None)
        with pytest.raises(ConfigurationError, match="must be a string"):
            validate_boundary_condition(["periodic"])


class TestValidateInitialState:
    """Tests for validate_initial_state function."""

    def test_random(self):
        """Test that 'random' is accepted."""
        assert validate_initial_state("random") == "random"

    def test_up(self):
        """Test that 'up' is accepted."""
        assert validate_initial_state("up") == "up"

    def test_down(self):
        """Test that 'down' is accepted."""
        assert validate_initial_state("down") == "down"

    def test_checkerboard(self):
        """Test that 'checkerboard' is accepted."""
        assert validate_initial_state("checkerboard") == "checkerboard"

    def test_case_insensitive(self):
        """Test that validation is case-insensitive."""
        assert validate_initial_state("RANDOM") == "random"
        assert validate_initial_state("Random") == "random"
        assert validate_initial_state("UP") == "up"
        assert validate_initial_state("CHECKERBOARD") == "checkerboard"

    def test_invalid_string_raises(self):
        """Test that invalid initial states raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="initial_state"):
            validate_initial_state("ferromagnetic")
        with pytest.raises(ConfigurationError, match="initial_state"):
            validate_initial_state("antiferromagnetic")

    def test_non_string_raises(self):
        """Test that non-strings raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="must be a string"):
            validate_initial_state(1)
        with pytest.raises(ConfigurationError, match="must be a string"):
            validate_initial_state(None)
        with pytest.raises(ConfigurationError, match="must be a string"):
            validate_initial_state(["random"])
