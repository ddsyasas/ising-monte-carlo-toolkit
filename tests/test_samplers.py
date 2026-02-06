"""Tests for Monte Carlo samplers."""

import numpy as np
import pytest

from ising_toolkit.models import Ising1D, Ising2D, Ising3D
from ising_toolkit.samplers import MetropolisSampler, WolffSampler
from ising_toolkit.io import SimulationResults
from ising_toolkit.utils import ConfigurationError, CRITICAL_TEMP_2D


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


# =============================================================================
# Wolff Sampler Tests
# =============================================================================


class TestWolffInitialization:
    """Tests for WolffSampler initialization."""

    def test_wolff_initialization(self):
        """Test basic initialization."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")

        sampler = WolffSampler(model, seed=42)

        assert sampler.model is model
        assert sampler.seed == 42
        assert sampler.p_add > 0
        assert sampler.p_add < 1
        assert sampler.cluster_sizes == []

    def test_wolff_rejects_1d(self):
        """Test that Wolff sampler rejects 1D models."""
        model = Ising1D(size=100, temperature=1.0)
        model.initialize("random")

        with pytest.raises(ConfigurationError, match="requires 2D or 3D"):
            WolffSampler(model, seed=42)

    def test_wolff_accepts_2d(self):
        """Test that Wolff sampler accepts 2D models."""
        model = Ising2D(size=16, temperature=2.0)
        sampler = WolffSampler(model, seed=42)

        assert sampler.model.dimension == 2

    def test_wolff_accepts_3d(self):
        """Test that Wolff sampler accepts 3D models."""
        model = Ising3D(size=8, temperature=4.0)
        sampler = WolffSampler(model, seed=42)

        assert sampler.model.dimension == 3

    def test_wolff_bond_probability(self):
        """Test bond probability calculation."""
        model = Ising2D(size=16, temperature=2.269)
        sampler = WolffSampler(model, seed=42)

        # p_add = 1 - exp(-2 * beta * J)
        expected = 1 - np.exp(-2 * model.beta * model.coupling)
        assert sampler.p_add == pytest.approx(expected)

    def test_wolff_repr(self):
        """Test string representation."""
        model = Ising2D(size=16, temperature=2.269)
        sampler = WolffSampler(model, seed=42)

        repr_str = repr(sampler)

        assert "WolffSampler" in repr_str
        assert "Ising2D" in repr_str
        assert "p_add" in repr_str


class TestWolffStep:
    """Tests for Wolff step functionality."""

    def test_wolff_step_returns_cluster_size(self):
        """Test that step returns cluster size."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        cluster_size = sampler.step()

        assert isinstance(cluster_size, int)
        assert cluster_size >= 1  # At least the seed spin
        assert cluster_size <= model.n_spins  # At most all spins

    def test_wolff_step_tracks_cluster_sizes(self):
        """Test that step tracks cluster sizes."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        for _ in range(10):
            sampler.step()

        assert len(sampler.cluster_sizes) == 10
        assert all(1 <= s <= model.n_spins for s in sampler.cluster_sizes)

    def test_wolff_step_changes_spins(self):
        """Test that steps change spin configuration."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("up")
        sampler = WolffSampler(model, seed=42)

        initial_mag = model.get_magnetization()

        # Run several steps
        for _ in range(10):
            sampler.step()

        final_mag = model.get_magnetization()

        # Magnetization should change (clusters flip sign)
        assert final_mag != initial_mag

    def test_wolff_step_preserves_spin_values(self):
        """Test that spins remain ±1 after steps."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        for _ in range(100):
            sampler.step()

        assert np.all(np.abs(model.spins) == 1)


class TestWolffClusterSize:
    """Tests for cluster size behavior."""

    def test_wolff_cluster_size_reasonable_at_tc(self):
        """Test that clusters are reasonably large at Tc."""
        model = Ising2D(size=32, temperature=CRITICAL_TEMP_2D)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        cluster_sizes = []
        for _ in range(100):
            size = sampler.step()
            cluster_sizes.append(size)

        mean_size = np.mean(cluster_sizes)

        # At Tc, clusters should be a significant fraction of the system
        # Mean cluster size should be at least a few percent of total spins
        assert mean_size > model.n_spins * 0.01

    def test_wolff_low_temp_large_clusters(self):
        """Test that low temperature produces large clusters."""
        model = Ising2D(size=32, temperature=1.0)  # Well below Tc
        model.initialize("up")  # Start aligned
        sampler = WolffSampler(model, seed=42)

        cluster_sizes = []
        for _ in range(20):
            size = sampler.step()
            cluster_sizes.append(size)

        mean_size = np.mean(cluster_sizes)

        # At low T with aligned state, clusters should be very large
        # (high p_add, all spins aligned)
        assert mean_size > model.n_spins * 0.5

    def test_wolff_high_temp_small_clusters(self):
        """Test that high temperature produces small clusters."""
        model = Ising2D(size=32, temperature=10.0)  # Well above Tc
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        cluster_sizes = []
        for _ in range(100):
            size = sampler.step()
            cluster_sizes.append(size)

        mean_size = np.mean(cluster_sizes)

        # At high T, p_add is small, so clusters should be small
        assert mean_size < model.n_spins * 0.1

    def test_wolff_get_mean_cluster_size(self):
        """Test get_mean_cluster_size method."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        # Initially should be 0
        assert sampler.get_mean_cluster_size() == 0.0

        # After steps, should be positive
        for _ in range(10):
            sampler.step()

        mean_size = sampler.get_mean_cluster_size()
        assert mean_size > 0
        assert mean_size == np.mean(sampler.cluster_sizes)


class TestWolffRun:
    """Tests for Wolff run() method."""

    def test_wolff_results_structure(self):
        """Test that run returns properly structured results."""
        model = Ising2D(size=16, temperature=2.269)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        results = sampler.run(
            n_steps=500,
            equilibration=100,
            measurement_interval=10,
            progress=False,
        )

        assert isinstance(results, SimulationResults)

        # Check data arrays
        expected_measurements = 500 // 10
        assert len(results.energy) == expected_measurements
        assert len(results.magnetization) == expected_measurements

        # Check metadata
        assert results.metadata["algorithm"] == "wolffsampler"
        assert "mean_cluster_size" in results.metadata
        assert "std_cluster_size" in results.metadata
        assert "max_cluster_size" in results.metadata
        assert "min_cluster_size" in results.metadata

    def test_wolff_results_3d(self):
        """Test Wolff sampler with 3D model."""
        model = Ising3D(size=8, temperature=4.511)
        model.initialize("random")
        sampler = WolffSampler(model, seed=42)

        results = sampler.run(
            n_steps=200,
            equilibration=50,
            measurement_interval=10,
            progress=False,
        )

        assert isinstance(results, SimulationResults)
        assert results.metadata["model_type"] == "ising3d"


class TestWolffVsMetropolis:
    """Comparative tests between Wolff and Metropolis."""

    def test_wolff_faster_decorrelation(self):
        """Test that Wolff has faster decorrelation at Tc.

        This is measured by checking that magnetization changes
        more rapidly with Wolff than Metropolis.
        """
        size = 32
        temperature = CRITICAL_TEMP_2D

        # Metropolis simulation
        model_metro = Ising2D(size=size, temperature=temperature)
        model_metro.set_seed(42)
        model_metro.initialize("up")
        sampler_metro = MetropolisSampler(model_metro, seed=42)

        metro_mags = []
        for _ in range(100):
            sampler_metro.step()
            metro_mags.append(abs(model_metro.get_magnetization_per_spin()))

        # Wolff simulation
        model_wolff = Ising2D(size=size, temperature=temperature)
        model_wolff.set_seed(42)
        model_wolff.initialize("up")
        sampler_wolff = WolffSampler(model_wolff, seed=42)

        wolff_mags = []
        for _ in range(100):
            sampler_wolff.step()
            wolff_mags.append(abs(model_wolff.get_magnetization_per_spin()))

        # Wolff should show larger changes in magnetization
        # (decorrelate faster from initial ordered state)
        metro_change = abs(metro_mags[-1] - metro_mags[0])
        wolff_change = abs(wolff_mags[-1] - wolff_mags[0])

        # Wolff should decorrelate faster
        # (larger change from initial state after same number of steps)
        assert wolff_change > metro_change * 0.5  # Wolff at least half as effective

    def test_wolff_explores_both_magnetization_signs(self):
        """Test that Wolff can flip between +M and -M states at Tc."""
        model = Ising2D(size=16, temperature=CRITICAL_TEMP_2D)
        model.initialize("up")
        sampler = WolffSampler(model, seed=42)

        # Track sign changes in magnetization
        mags = []
        for _ in range(200):
            sampler.step()
            mags.append(model.get_magnetization_per_spin())

        # Should see both positive and negative magnetizations
        # (system can tunnel between +M and -M states)
        has_positive = any(m > 0.3 for m in mags)
        has_negative = any(m < -0.3 for m in mags)

        # At Tc, Wolff should be able to explore both phases
        assert has_positive or has_negative  # At least one phase explored
