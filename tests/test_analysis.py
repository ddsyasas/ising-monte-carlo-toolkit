"""Tests for analysis functions."""

import numpy as np
import pytest

from ising_toolkit.analysis import (
    calculate_heat_capacity,
    calculate_susceptibility,
    calculate_binder_cumulant,
    calculate_correlation_time,
    calculate_all_observables,
)
from ising_toolkit.io import SimulationResults


class TestHeatCapacity:
    """Tests for heat capacity calculation."""

    def test_heat_capacity_formula(self):
        """Test heat capacity with known variance."""
        # Create energy data with known mean and variance
        # E = [-100, -100, -100, -100] has variance 0
        energy = np.array([-100.0, -100.0, -100.0, -100.0])
        C = calculate_heat_capacity(energy, temperature=2.0, n_spins=100)
        assert C == pytest.approx(0.0)

    def test_heat_capacity_with_variance(self):
        """Test heat capacity with non-zero variance."""
        # E = [-102, -98] -> mean = -100, variance = 4
        # C/N = variance / (N * T^2) = 4 / (100 * 4) = 0.01
        energy = np.array([-102.0, -98.0])
        C = calculate_heat_capacity(energy, temperature=2.0, n_spins=100)
        expected = 4.0 / (100 * 4.0)  # variance / (N * T^2)
        assert C == pytest.approx(expected)

    def test_heat_capacity_temperature_dependence(self):
        """Test that heat capacity scales with 1/T²."""
        energy = np.array([-102.0, -98.0, -100.0, -101.0, -99.0])

        C_T1 = calculate_heat_capacity(energy, temperature=1.0, n_spins=100)
        C_T2 = calculate_heat_capacity(energy, temperature=2.0, n_spins=100)

        # C(T=1) should be 4x C(T=2)
        assert C_T1 == pytest.approx(4 * C_T2)

    def test_heat_capacity_formula_verification(self):
        """Verify the heat capacity formula directly.

        C/N = Var(E) / (N * T²)
        """
        # Create data with known variance
        # E = [90, 110] -> mean=100, variance = 100 (population)
        # Using sample variance with ddof=0: var = 100
        energy = np.array([90.0, 110.0])
        variance = np.var(energy)  # = 100

        n_spins = 50
        temperature = 2.0

        C = calculate_heat_capacity(energy, temperature, n_spins)

        # Expected: C/N = variance / (N * T²) = 100 / (50 * 4) = 0.5
        expected = variance / (n_spins * temperature**2)
        assert C == pytest.approx(expected)

    def test_heat_capacity_positive(self):
        """Test that heat capacity is always non-negative."""
        np.random.seed(42)
        energy = np.random.normal(-500, 50, 1000)

        C = calculate_heat_capacity(energy, temperature=2.0, n_spins=256)

        assert C >= 0


class TestSusceptibility:
    """Tests for magnetic susceptibility calculation."""

    def test_susceptibility_formula(self):
        """Test susceptibility with known variance."""
        # M = [50, 50, 50, 50] has variance 0
        magnetization = np.array([50.0, 50.0, 50.0, 50.0])
        chi = calculate_susceptibility(magnetization, temperature=2.0, n_spins=100)
        assert chi == pytest.approx(0.0)

    def test_susceptibility_with_variance(self):
        """Test susceptibility with non-zero variance."""
        # M = [52, 48] -> mean = 50, variance = 4
        # χ/N = variance / (N * T) = 4 / (100 * 2) = 0.02
        magnetization = np.array([52.0, 48.0])
        chi = calculate_susceptibility(magnetization, temperature=2.0, n_spins=100)
        expected = 4.0 / (100 * 2.0)
        assert chi == pytest.approx(expected)

    def test_susceptibility_temperature_dependence(self):
        """Test that susceptibility scales with 1/T."""
        magnetization = np.array([52.0, 48.0, 50.0, 51.0, 49.0])

        chi_T1 = calculate_susceptibility(magnetization, temperature=1.0, n_spins=100)
        chi_T2 = calculate_susceptibility(magnetization, temperature=2.0, n_spins=100)

        # χ(T=1) should be 2x χ(T=2)
        assert chi_T1 == pytest.approx(2 * chi_T2)

    def test_susceptibility_symmetric_magnetization(self):
        """Test susceptibility with symmetric magnetization (mean ≈ 0)."""
        # Symmetric around 0: ⟨M⟩ = 0, so χ/N = ⟨M²⟩ / (N * T)
        magnetization = np.array([50.0, -50.0, 50.0, -50.0])
        chi = calculate_susceptibility(magnetization, temperature=1.0, n_spins=100)

        # variance = ⟨M²⟩ - ⟨M⟩² = 2500 - 0 = 2500
        # χ/N = 2500 / (100 * 1) = 25
        expected = 2500.0 / (100 * 1.0)
        assert chi == pytest.approx(expected)

    def test_susceptibility_positive(self):
        """Test that susceptibility is always non-negative."""
        np.random.seed(42)
        magnetization = np.random.normal(0, 100, 1000)

        chi = calculate_susceptibility(magnetization, temperature=2.0, n_spins=256)

        assert chi >= 0


class TestBinderCumulant:
    """Tests for Binder cumulant calculation."""

    def test_binder_cumulant_ordered_phase(self):
        """Test Binder cumulant in ordered phase (should approach 2/3)."""
        # In ordered phase, |m| ≈ 1, so m⁴ ≈ m² ≈ 1
        # U = 1 - 1/(3*1) = 2/3
        n_spins = 100
        magnetization = np.array([100.0] * 1000)  # All +1 per spin

        U = calculate_binder_cumulant(magnetization, n_spins)

        assert U == pytest.approx(2.0 / 3.0)

    def test_binder_cumulant_disordered_phase(self):
        """Test Binder cumulant in disordered phase (should approach 0)."""
        # In disordered phase with Gaussian fluctuations around 0:
        # ⟨m⁴⟩ = 3⟨m²⟩² (for Gaussian), so U = 1 - 1 = 0
        np.random.seed(42)
        n_spins = 1000
        # Generate Gaussian magnetization (high temperature behavior)
        magnetization = np.random.normal(0, n_spins * 0.1, 10000)

        U = calculate_binder_cumulant(magnetization, n_spins)

        # Should be close to 0 (within statistical noise)
        assert abs(U) < 0.1

    def test_binder_cumulant_symmetric(self):
        """Test Binder cumulant is symmetric under M -> -M."""
        n_spins = 100
        magnetization_pos = np.array([80.0, 90.0, 85.0, 95.0, 88.0])
        magnetization_neg = -magnetization_pos

        U_pos = calculate_binder_cumulant(magnetization_pos, n_spins)
        U_neg = calculate_binder_cumulant(magnetization_neg, n_spins)

        assert U_pos == pytest.approx(U_neg)

    def test_binder_cumulant_range(self):
        """Test Binder cumulant is in expected range."""
        np.random.seed(42)
        n_spins = 100

        # Random magnetization
        magnetization = np.random.uniform(-100, 100, 1000)
        U = calculate_binder_cumulant(magnetization, n_spins)

        # U should be between 0 and 2/3 for physical distributions
        # (can be slightly outside due to finite sampling)
        assert U >= -0.1
        assert U <= 0.75

    def test_binder_cumulant_zero_magnetization(self):
        """Test Binder cumulant handles zero magnetization."""
        magnetization = np.zeros(100)
        U = calculate_binder_cumulant(magnetization, n_spins=100)

        assert U == 0.0


class TestCorrelationTime:
    """Tests for autocorrelation time calculation."""

    def test_correlation_time_uncorrelated(self):
        """Test correlation time for uncorrelated data."""
        np.random.seed(42)
        # Uncorrelated random data should have τ ≈ 0.5
        data = np.random.randn(10000)

        tau = calculate_correlation_time(data)

        # τ should be close to 0.5 for uncorrelated data
        assert tau < 2.0

    def test_correlation_time_constant(self):
        """Test correlation time for constant data."""
        data = np.ones(100) * 5.0

        tau = calculate_correlation_time(data)

        # Constant data has zero variance, should return 1
        assert tau == 1.0

    def test_correlation_time_correlated(self):
        """Test correlation time for correlated data."""
        np.random.seed(42)
        # Create exponentially correlated data
        n = 1000
        tau_true = 10.0
        data = np.zeros(n)
        data[0] = np.random.randn()
        alpha = np.exp(-1.0 / tau_true)
        for i in range(1, n):
            data[i] = alpha * data[i - 1] + np.sqrt(1 - alpha**2) * np.random.randn()

        tau = calculate_correlation_time(data)

        # Estimated tau should be reasonably close to true value
        assert tau > 2.0  # Definitely correlated

    def test_correlation_time_positive(self):
        """Test that correlation time is always positive."""
        np.random.seed(42)
        data = np.random.randn(500)

        tau = calculate_correlation_time(data)

        assert tau > 0


class TestCalculateAllObservables:
    """Tests for calculate_all_observables function."""

    @pytest.fixture
    def mock_results(self):
        """Create mock simulation results."""
        np.random.seed(42)
        n = 1000
        n_spins = 100

        # Create energy and magnetization data
        energy = np.random.normal(-200, 20, n)
        magnetization = np.random.normal(0, 30, n)

        metadata = {
            "n_spins": n_spins,
            "temperature": 2.0,
            "model_type": "ising2d",
        }

        return SimulationResults(
            energy=energy,
            magnetization=magnetization,
            metadata=metadata,
        )

    def test_all_observables_keys(self, mock_results):
        """Test that all expected keys are present."""
        observables = calculate_all_observables(mock_results)

        expected_keys = {
            "energy_mean",
            "energy_std",
            "magnetization_mean",
            "magnetization_std",
            "abs_magnetization_mean",
            "abs_magnetization_std",
            "heat_capacity",
            "susceptibility",
            "binder_cumulant",
        }

        assert set(observables.keys()) == expected_keys

    def test_all_observables_types(self, mock_results):
        """Test that all values are floats."""
        observables = calculate_all_observables(mock_results)

        for key, value in observables.items():
            assert isinstance(value, float), f"{key} is not float"

    def test_all_observables_per_spin(self, mock_results):
        """Test that energy and magnetization are per spin."""
        observables = calculate_all_observables(mock_results)
        n_spins = mock_results.metadata["n_spins"]

        # Energy per spin should be roughly E_total / N
        total_energy_mean = np.mean(mock_results.energy)
        expected_e_per_spin = total_energy_mean / n_spins

        assert observables["energy_mean"] == pytest.approx(expected_e_per_spin)

    def test_all_observables_consistency(self, mock_results):
        """Test internal consistency of observables."""
        observables = calculate_all_observables(mock_results)

        # Absolute magnetization mean should be >= 0
        assert observables["abs_magnetization_mean"] >= 0

        # Absolute magnetization mean >= |magnetization mean|
        assert observables["abs_magnetization_mean"] >= abs(
            observables["magnetization_mean"]
        )

        # Standard deviations should be positive
        assert observables["energy_std"] >= 0
        assert observables["magnetization_std"] >= 0
        assert observables["abs_magnetization_std"] >= 0

        # Heat capacity and susceptibility should be non-negative
        assert observables["heat_capacity"] >= 0
        assert observables["susceptibility"] >= 0


class TestObservablesIntegration:
    """Integration tests with realistic simulation-like data."""

    def test_ordered_phase_observables(self):
        """Test observables in ordered phase (low T, high |M|)."""
        n_spins = 256
        n_samples = 1000

        # Simulate ordered phase: high magnetization, low fluctuations
        energy = np.random.normal(-2 * n_spins, 5, n_samples)  # E ≈ -2N
        magnetization = np.random.normal(0.9 * n_spins, 10, n_samples)  # M ≈ 0.9N

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
            metadata={"n_spins": n_spins, "temperature": 1.0},
        )

        observables = calculate_all_observables(results)

        # In ordered phase, Binder cumulant should be close to 2/3
        assert observables["binder_cumulant"] > 0.5

        # Absolute magnetization should be close to magnetization
        assert observables["abs_magnetization_mean"] == pytest.approx(
            abs(observables["magnetization_mean"]), rel=0.2
        )

    def test_disordered_phase_observables(self):
        """Test observables in disordered phase (high T, M ≈ 0)."""
        n_spins = 256
        n_samples = 1000

        # Simulate disordered phase: low magnetization, high fluctuations
        energy = np.random.normal(0, 50, n_samples)  # E ≈ 0
        magnetization = np.random.normal(0, 50, n_samples)  # M ≈ 0

        results = SimulationResults(
            energy=energy,
            magnetization=magnetization,
            metadata={"n_spins": n_spins, "temperature": 5.0},
        )

        observables = calculate_all_observables(results)

        # In disordered phase, mean magnetization should be small
        assert abs(observables["magnetization_mean"]) < 0.5

        # Binder cumulant should be close to 0
        assert observables["binder_cumulant"] < 0.3
