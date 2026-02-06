"""Tests for Monte Carlo samplers."""

import numpy as np
import pytest

from ising_toolkit.models import Ising1D, Ising2D
from ising_toolkit.samplers import MetropolisSampler
from ising_toolkit.io import SimulationResults


class TestMetropolisInitialization:
    """Tests for MetropolisSampler initialization."""

    def test_metropolis_initialization(self):
        """Test basic initialization."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")

        sampler = MetropolisSampler(model, seed=42)

        assert sampler.model is model
        assert sampler.seed == 42
        assert sampler.n_accepted == 0
        assert sampler.n_attempted == 0

    def test_metropolis_initialization_no_seed(self):
        """Test initialization without seed."""
        model = Ising2D(size=10, temperature=2.0)
        sampler = MetropolisSampler(model)

        assert sampler.seed is None
        assert sampler.rng is not None

    def test_metropolis_repr(self):
        """Test string representation."""
        model = Ising2D(size=10, temperature=2.269)
        sampler = MetropolisSampler(model, seed=42)

        repr_str = repr(sampler)

        assert "MetropolisSampler" in repr_str
        assert "Ising2D" in repr_str
        assert "2.269" in repr_str


class TestMetropolisStep:
    """Tests for Metropolis step functionality."""

    def test_metropolis_step_changes_spins(self):
        """Test that steps can change spin configuration."""
        model = Ising2D(size=10, temperature=2.5)
        model.initialize("up")
        sampler = MetropolisSampler(model, seed=42)

        initial_spins = model.spins.copy()

        # Run several steps
        for _ in range(10):
            sampler.step()

        # At T=2.5, some spins should have flipped
        assert not np.array_equal(model.spins, initial_spins)

    def test_metropolis_step_returns_accepted_count(self):
        """Test that step returns number of accepted moves."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        accepted = sampler.step()

        assert isinstance(accepted, int)
        assert 0 <= accepted <= model.n_spins

    def test_metropolis_step_updates_counters(self):
        """Test that step updates acceptance counters."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        sampler.step()

        assert sampler.n_attempted == model.n_spins
        assert sampler.n_accepted >= 0
        assert sampler.n_accepted <= sampler.n_attempted

    def test_metropolis_step_preserves_spin_values(self):
        """Test that spins remain ±1 after steps."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        for _ in range(100):
            sampler.step()

        assert np.all(np.abs(model.spins) == 1)


class TestMetropolisPhysics:
    """Tests for physical behavior of Metropolis sampler."""

    def test_metropolis_low_temp_stays_ordered(self):
        """Test that low temperature preserves order."""
        # At very low T, aligned state should remain mostly aligned
        model = Ising2D(size=16, temperature=0.5)  # Well below Tc
        model.initialize("up")
        sampler = MetropolisSampler(model, seed=42)

        initial_mag = model.get_magnetization_per_spin()

        # Run simulation
        for _ in range(100):
            sampler.step()

        final_mag = model.get_magnetization_per_spin()

        # Magnetization should stay high (close to 1)
        assert final_mag > 0.9
        assert abs(final_mag - initial_mag) < 0.2

    def test_metropolis_high_temp_disorders(self):
        """Test that high temperature disorders the system."""
        # At very high T, aligned state should disorder
        model = Ising2D(size=16, temperature=10.0)  # Well above Tc
        model.initialize("up")
        sampler = MetropolisSampler(model, seed=42)

        # Run enough steps for equilibration
        for _ in range(500):
            sampler.step()

        final_mag = abs(model.get_magnetization_per_spin())

        # Magnetization per spin should be close to 0 at high T
        # (for finite system, won't be exactly 0)
        assert final_mag < 0.3

    def test_metropolis_acceptance_rate_temperature_dependence(self):
        """Test that acceptance rate varies with temperature."""
        model_low_T = Ising2D(size=16, temperature=0.5)
        model_low_T.initialize("random")
        sampler_low_T = MetropolisSampler(model_low_T, seed=42)

        model_high_T = Ising2D(size=16, temperature=5.0)
        model_high_T.initialize("random")
        sampler_high_T = MetropolisSampler(model_high_T, seed=42)

        # Run both
        for _ in range(100):
            sampler_low_T.step()
            sampler_high_T.step()

        # High T should have higher acceptance rate
        assert sampler_high_T.get_acceptance_rate() > sampler_low_T.get_acceptance_rate()

    def test_metropolis_energy_decreases_from_random(self):
        """Test that energy generally decreases from random high-energy state."""
        model = Ising2D(size=16, temperature=1.5)  # Below Tc
        model.initialize("checkerboard")  # High energy state
        sampler = MetropolisSampler(model, seed=42)

        initial_energy = model.get_energy()

        # Run simulation
        for _ in range(500):
            sampler.step()

        final_energy = model.get_energy()

        # Energy should decrease (become more negative)
        assert final_energy < initial_energy


class TestMetropolisReproducibility:
    """Tests for reproducibility with seeds."""

    def test_metropolis_seed_reproducibility(self):
        """Test that same seed produces same results."""
        # First run
        model1 = Ising2D(size=16, temperature=2.269)
        model1.initialize("random")
        # Need to set seed BEFORE initializing random state
        model1.set_seed(123)
        model1.initialize("random")
        sampler1 = MetropolisSampler(model1, seed=42)

        for _ in range(50):
            sampler1.step()

        energy1 = model1.get_energy()
        mag1 = model1.get_magnetization()
        spins1 = model1.spins.copy()

        # Second run with same seeds
        model2 = Ising2D(size=16, temperature=2.269)
        model2.set_seed(123)
        model2.initialize("random")
        sampler2 = MetropolisSampler(model2, seed=42)

        for _ in range(50):
            sampler2.step()

        energy2 = model2.get_energy()
        mag2 = model2.get_magnetization()
        spins2 = model2.spins.copy()

        # Results should be identical
        assert energy1 == energy2
        assert mag1 == mag2
        np.testing.assert_array_equal(spins1, spins2)

    def test_metropolis_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        model1 = Ising2D(size=16, temperature=2.269)
        model1.set_seed(100)
        model1.initialize("random")
        sampler1 = MetropolisSampler(model1, seed=42)

        model2 = Ising2D(size=16, temperature=2.269)
        model2.set_seed(100)
        model2.initialize("random")
        sampler2 = MetropolisSampler(model2, seed=99)  # Different seed

        for _ in range(50):
            sampler1.step()
            sampler2.step()

        # Results should differ (very unlikely to be same by chance)
        assert not np.array_equal(model1.spins, model2.spins)


class TestMetropolisRun:
    """Tests for the run() method."""

    def test_metropolis_results_structure(self):
        """Test that run returns properly structured results."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        results = sampler.run(
            n_steps=1000,
            equilibration=100,
            measurement_interval=10,
            progress=False,
        )

        # Check return type
        assert isinstance(results, SimulationResults)

        # Check data arrays
        expected_measurements = 1000 // 10
        assert len(results.energy) == expected_measurements
        assert len(results.magnetization) == expected_measurements

        # Check metadata
        assert results.metadata["model_type"] == "ising2d"
        assert results.metadata["n_steps"] == 1000
        assert results.metadata["equilibration"] == 100
        assert results.metadata["measurement_interval"] == 10
        assert results.metadata["algorithm"] == "metropolissampler"
        assert results.metadata["seed"] == 42
        assert "elapsed_time" in results.metadata
        assert "acceptance_rate" in results.metadata

    def test_metropolis_results_with_configurations(self):
        """Test saving configurations."""
        model = Ising2D(size=8, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        results = sampler.run(
            n_steps=1000,
            equilibration=100,
            measurement_interval=10,
            save_configurations=True,
            configuration_interval=100,
            progress=False,
        )

        # Should have 1000 / 100 = 10 configurations
        assert len(results.configurations) == 10
        assert results.configurations[0].shape == (8, 8)

    def test_metropolis_results_statistics(self):
        """Test that results have valid statistics."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        results = sampler.run(
            n_steps=5000,
            equilibration=1000,
            measurement_interval=10,
            progress=False,
        )

        # Check statistics are computed
        assert results.energy_mean != 0
        assert results.energy_std > 0
        assert results.energy_err > 0
        assert results.energy_err < results.energy_std

    def test_metropolis_run_1d_model(self):
        """Test that sampler works with 1D model."""
        model = Ising1D(size=100, temperature=1.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        results = sampler.run(
            n_steps=1000,
            equilibration=100,
            measurement_interval=10,
            progress=False,
        )

        assert isinstance(results, SimulationResults)
        assert results.metadata["model_type"] == "ising1d"


class TestMetropolisAcceptanceRate:
    """Tests for acceptance rate tracking."""

    def test_get_acceptance_rate_initial(self):
        """Test acceptance rate is 0 initially."""
        model = Ising2D(size=10, temperature=2.0)
        sampler = MetropolisSampler(model, seed=42)

        assert sampler.get_acceptance_rate() == 0.0

    def test_get_acceptance_rate_after_steps(self):
        """Test acceptance rate is valid after steps."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        for _ in range(10):
            sampler.step()

        rate = sampler.get_acceptance_rate()

        assert 0.0 <= rate <= 1.0

    def test_reset_counters(self):
        """Test that counters can be reset."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")
        sampler = MetropolisSampler(model, seed=42)

        # Run some steps
        for _ in range(10):
            sampler.step()

        assert sampler.n_attempted > 0

        # Reset
        sampler.reset_counters()

        assert sampler.n_attempted == 0
        assert sampler.n_accepted == 0
        assert sampler.get_acceptance_rate() == 0.0
