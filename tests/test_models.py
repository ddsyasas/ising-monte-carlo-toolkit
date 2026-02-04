"""Tests for Ising model implementations."""

import numpy as np
import pytest

from ising_toolkit.models import Ising1D
from ising_toolkit.utils import ConfigurationError


class TestIsing1DInitialization:
    """Tests for Ising1D initialization."""

    def test_initialization_size(self):
        """Test that model initializes with correct size."""
        model = Ising1D(size=10, temperature=2.0)
        assert model.size == 10
        assert model.n_spins == 10
        assert len(model.spins) == 10

    def test_initialization_size_large(self):
        """Test initialization with larger sizes."""
        model = Ising1D(size=1000, temperature=1.0)
        assert model.size == 1000

    def test_initialization_invalid_size(self):
        """Test that invalid sizes raise errors."""
        with pytest.raises(ConfigurationError):
            Ising1D(size=1, temperature=2.0)  # Too small
        with pytest.raises(ConfigurationError):
            Ising1D(size=0, temperature=2.0)
        with pytest.raises(ConfigurationError):
            Ising1D(size=-5, temperature=2.0)

    def test_initialization_temperature(self):
        """Test that temperature is set correctly."""
        model = Ising1D(size=10, temperature=2.5)
        assert model.temperature == 2.5
        assert model.beta == pytest.approx(1.0 / 2.5)

    def test_initialization_invalid_temperature(self):
        """Test that invalid temperatures raise errors."""
        with pytest.raises(ConfigurationError):
            Ising1D(size=10, temperature=0)
        with pytest.raises(ConfigurationError):
            Ising1D(size=10, temperature=-1.0)

    def test_initialization_coupling(self):
        """Test that coupling constant is set correctly."""
        model = Ising1D(size=10, temperature=2.0, coupling=1.5)
        assert model.coupling == 1.5

    def test_initialization_boundary_periodic(self):
        """Test periodic boundary condition."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        assert model.boundary == "periodic"

    def test_initialization_boundary_fixed(self):
        """Test fixed boundary condition."""
        model = Ising1D(size=10, temperature=2.0, boundary="fixed")
        assert model.boundary == "fixed"

    def test_initialization_dimension(self):
        """Test that dimension is 1."""
        model = Ising1D(size=10, temperature=2.0)
        assert model.dimension == 1


class TestIsing1DStateInitialization:
    """Tests for spin state initialization."""

    def test_initialization_random(self):
        """Test random initialization produces +1 and -1 spins."""
        model = Ising1D(size=100, temperature=2.0)
        model.set_seed(42)
        model.initialize("random")

        # Should have both +1 and -1 spins
        assert np.any(model.spins == 1)
        assert np.any(model.spins == -1)
        # All spins should be ±1
        assert np.all(np.abs(model.spins) == 1)

    def test_initialization_up(self):
        """Test 'up' initialization sets all spins to +1."""
        model = Ising1D(size=50, temperature=2.0)
        model.initialize("up")

        assert np.all(model.spins == 1)
        assert model.get_magnetization() == 50

    def test_initialization_down(self):
        """Test 'down' initialization sets all spins to -1."""
        model = Ising1D(size=50, temperature=2.0)
        model.initialize("down")

        assert np.all(model.spins == -1)
        assert model.get_magnetization() == -50

    def test_initialization_checkerboard(self):
        """Test checkerboard initialization alternates spins."""
        model = Ising1D(size=10, temperature=2.0)
        model.initialize("checkerboard")

        # Even indices should be +1, odd should be -1
        for i in range(10):
            expected = 1 if i % 2 == 0 else -1
            assert model.spins[i] == expected

    def test_initialization_invalid_state(self):
        """Test that invalid state raises error."""
        model = Ising1D(size=10, temperature=2.0)
        with pytest.raises(ConfigurationError):
            model.initialize("invalid")

    def test_initialization_case_insensitive(self):
        """Test that state names are case-insensitive."""
        model = Ising1D(size=10, temperature=2.0)
        model.initialize("UP")
        assert np.all(model.spins == 1)

        model.initialize("Random")
        assert np.all(np.abs(model.spins) == 1)


class TestIsing1DEnergy:
    """Tests for energy calculations."""

    def test_energy_all_up_periodic(self):
        """Test energy when all spins up with periodic BC."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # E = -J * N (all bonds aligned, N bonds for periodic)
        expected_energy = -1.0 * 10
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_energy_all_up_fixed(self):
        """Test energy when all spins up with fixed BC."""
        model = Ising1D(size=10, temperature=2.0, boundary="fixed")
        model.initialize("up")

        # E = -J * (N-1) (all bonds aligned, N-1 bonds for fixed)
        expected_energy = -1.0 * 9
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_energy_all_down_periodic(self):
        """Test energy when all spins down (same as all up)."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("down")

        # Same energy as all up
        expected_energy = -1.0 * 10
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_energy_alternating_periodic(self):
        """Test energy for alternating pattern with periodic BC."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("checkerboard")

        # All bonds anti-aligned: E = +J * N
        expected_energy = 1.0 * 10
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_energy_alternating_fixed(self):
        """Test energy for alternating pattern with fixed BC."""
        model = Ising1D(size=10, temperature=2.0, boundary="fixed")
        model.initialize("checkerboard")

        # All bonds anti-aligned: E = +J * (N-1)
        expected_energy = 1.0 * 9
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_energy_with_coupling(self):
        """Test energy with non-unit coupling constant."""
        J = 2.5
        model = Ising1D(size=10, temperature=2.0, coupling=J, boundary="periodic")
        model.initialize("up")

        expected_energy = -J * 10
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_energy_per_spin(self):
        """Test energy per spin calculation."""
        model = Ising1D(size=100, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # e = E/N = -J for all aligned
        assert model.get_energy_per_spin() == pytest.approx(-1.0)


class TestIsing1DMagnetization:
    """Tests for magnetization calculations."""

    def test_magnetization_all_up(self):
        """Test magnetization when all spins up."""
        model = Ising1D(size=50, temperature=2.0)
        model.initialize("up")

        assert model.get_magnetization() == 50

    def test_magnetization_all_down(self):
        """Test magnetization when all spins down."""
        model = Ising1D(size=50, temperature=2.0)
        model.initialize("down")

        assert model.get_magnetization() == -50

    def test_magnetization_alternating(self):
        """Test magnetization for alternating pattern."""
        # Even size: equal +1 and -1
        model = Ising1D(size=10, temperature=2.0)
        model.initialize("checkerboard")
        assert model.get_magnetization() == 0

        # Odd size: one more +1 than -1
        model = Ising1D(size=11, temperature=2.0)
        model.initialize("checkerboard")
        assert model.get_magnetization() == 1

    def test_magnetization_per_spin(self):
        """Test magnetization per spin."""
        model = Ising1D(size=100, temperature=2.0)
        model.initialize("up")
        assert model.get_magnetization_per_spin() == pytest.approx(1.0)

        model.initialize("down")
        assert model.get_magnetization_per_spin() == pytest.approx(-1.0)

        model.initialize("checkerboard")
        assert model.get_magnetization_per_spin() == pytest.approx(0.0)


class TestIsing1DSpinFlip:
    """Tests for spin flip operations."""

    def test_flip_spin(self):
        """Test that flip_spin reverses spin value."""
        model = Ising1D(size=10, temperature=2.0)
        model.initialize("up")

        # Flip middle spin
        original = model.spins[5]
        model.flip_spin((5,))
        assert model.spins[5] == -original

        # Flip again to restore
        model.flip_spin((5,))
        assert model.spins[5] == original

    def test_flip_spin_energy_change(self):
        """Test that energy changes correctly after flip."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        initial_energy = model.get_energy()
        expected_dE = model.get_energy_change((5,))

        model.flip_spin((5,))
        new_energy = model.get_energy()

        assert new_energy - initial_energy == pytest.approx(expected_dE)

    def test_flip_spin_magnetization_change(self):
        """Test that magnetization changes by ±2 after flip."""
        model = Ising1D(size=10, temperature=2.0)
        model.initialize("up")

        initial_mag = model.get_magnetization()
        model.flip_spin((3,))
        new_mag = model.get_magnetization()

        # Flipping +1 -> -1 changes M by -2
        assert new_mag - initial_mag == -2


class TestIsing1DEnergyChange:
    """Tests for energy change calculations."""

    def test_energy_change_values_periodic(self):
        """Test that energy changes are only -4, -2, 0, 2, 4 for periodic BC."""
        model = Ising1D(size=100, temperature=2.0, boundary="periodic")
        model.set_seed(42)
        model.initialize("random")

        valid_changes = {-4.0, -2.0, 0.0, 2.0, 4.0}
        observed_changes = set()

        for i in range(model.size):
            dE = model.get_energy_change((i,))
            observed_changes.add(dE)
            assert dE in valid_changes, f"Invalid energy change: {dE}"

        # With random config, should see multiple different values
        assert len(observed_changes) >= 2

    def test_energy_change_values_fixed(self):
        """Test energy changes for fixed BC."""
        model = Ising1D(size=100, temperature=2.0, boundary="fixed")
        model.set_seed(42)
        model.initialize("random")

        # For fixed BC, edge spins can have dE = -2, 0, 2
        # Interior spins can have dE = -4, -2, 0, 2, 4
        valid_changes = {-4.0, -2.0, 0.0, 2.0, 4.0}

        for i in range(model.size):
            dE = model.get_energy_change((i,))
            assert dE in valid_changes, f"Invalid energy change: {dE}"

    def test_energy_change_all_aligned(self):
        """Test energy change when flipping aligned spin."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # All neighbors are +1, flipping any spin gives dE = 2*J*1*(1+1) = 4
        for i in range(model.size):
            assert model.get_energy_change((i,)) == pytest.approx(4.0)

    def test_energy_change_with_coupling(self):
        """Test energy change scales with coupling constant."""
        J = 2.0
        model = Ising1D(size=10, temperature=2.0, coupling=J, boundary="periodic")
        model.initialize("up")

        # dE = 2*J*s*(s_left + s_right) = 2*2*1*2 = 8
        expected_dE = 8.0
        assert model.get_energy_change((5,)) == pytest.approx(expected_dE)


class TestIsing1DNeighborSum:
    """Tests for neighbor sum calculations."""

    def test_neighbor_sum_all_up_periodic(self):
        """Test neighbor sum when all spins up (periodic)."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # All neighbors are +1, so sum is 2
        for i in range(model.size):
            assert model.get_neighbor_sum((i,)) == 2

    def test_neighbor_sum_all_up_fixed(self):
        """Test neighbor sum for fixed BC."""
        model = Ising1D(size=10, temperature=2.0, boundary="fixed")
        model.initialize("up")

        # Edge spins have 1 neighbor, interior have 2
        assert model.get_neighbor_sum((0,)) == 1  # Left edge
        assert model.get_neighbor_sum((9,)) == 1  # Right edge
        assert model.get_neighbor_sum((5,)) == 2  # Interior

    def test_neighbor_sum_checkerboard(self):
        """Test neighbor sum for alternating pattern."""
        model = Ising1D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("checkerboard")

        # Each spin has two anti-aligned neighbors
        for i in range(model.size):
            spin = model.spins[i]
            neighbor_sum = model.get_neighbor_sum((i,))
            # Neighbors are opposite sign
            assert neighbor_sum == -2 * spin


class TestIsing1DCopy:
    """Tests for model copying."""

    def test_copy_independence(self):
        """Test that copy is independent of original."""
        model = Ising1D(size=10, temperature=2.0)
        model.initialize("random")

        copy = model.copy()

        # Modify original
        model.flip_spin((0,))

        # Copy should be unchanged
        assert copy.spins[0] != model.spins[0]

    def test_copy_parameters(self):
        """Test that copy has same parameters."""
        model = Ising1D(size=20, temperature=3.5, coupling=1.5, boundary="fixed")
        model.initialize("up")

        copy = model.copy()

        assert copy.size == model.size
        assert copy.temperature == model.temperature
        assert copy.coupling == model.coupling
        assert copy.boundary == model.boundary
        np.testing.assert_array_equal(copy.spins, model.spins)


class TestIsing1DRandomSite:
    """Tests for random site selection."""

    def test_random_site_range(self):
        """Test that random sites are within valid range."""
        model = Ising1D(size=10, temperature=2.0)

        for _ in range(100):
            site = model.random_site()
            assert len(site) == 1
            assert 0 <= site[0] < model.size

    def test_random_site_distribution(self):
        """Test that random sites are approximately uniform."""
        model = Ising1D(size=10, temperature=2.0)
        model.set_seed(42)

        counts = np.zeros(model.size)
        n_samples = 10000

        for _ in range(n_samples):
            site = model.random_site()
            counts[site[0]] += 1

        # Each site should be selected ~1000 times
        expected = n_samples / model.size
        for count in counts:
            assert abs(count - expected) < 0.15 * expected  # Within 15%


class TestIsing1DSetTemperature:
    """Tests for temperature changes."""

    def test_set_temperature(self):
        """Test setting new temperature."""
        model = Ising1D(size=10, temperature=2.0)

        model.set_temperature(3.0)

        assert model.temperature == 3.0
        assert model.beta == pytest.approx(1.0 / 3.0)

    def test_set_temperature_updates_acceptance(self):
        """Test that acceptance probabilities are updated."""
        model = Ising1D(size=10, temperature=2.0)
        old_probs = model.acceptance_probs.copy()

        model.set_temperature(1.0)
        new_probs = model.acceptance_probs

        # Lower temperature -> lower acceptance probabilities
        assert new_probs[4] < old_probs[4]
        assert new_probs[2] < old_probs[2]

    def test_set_temperature_invalid(self):
        """Test that invalid temperature raises error."""
        model = Ising1D(size=10, temperature=2.0)

        with pytest.raises(ConfigurationError):
            model.set_temperature(0)
        with pytest.raises(ConfigurationError):
            model.set_temperature(-1.0)


class TestIsing1DRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ output."""
        model = Ising1D(size=10, temperature=2.0, coupling=1.0, boundary="periodic")
        repr_str = repr(model)

        assert "Ising1D" in repr_str
        assert "size=10" in repr_str
        assert "temperature=2.0" in repr_str
        assert "periodic" in repr_str
