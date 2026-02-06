"""Tests for Ising model implementations."""

import numpy as np
import pytest

from ising_toolkit.models import Ising1D, Ising2D, Ising3D, create_model
from ising_toolkit.utils import ConfigurationError, CRITICAL_TEMP_2D, CRITICAL_TEMP_3D


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


# =============================================================================
# 2D Ising Model Tests
# =============================================================================


class TestIsing2DInitialization:
    """Tests for Ising2D initialization."""

    def test_2d_initialization(self):
        """Test that 2D model initializes correctly."""
        model = Ising2D(size=10, temperature=2.0)

        assert model.size == 10
        assert model.n_spins == 100  # 10 * 10
        assert model.shape == (10, 10)
        assert model.spins.shape == (10, 10)
        assert model.dimension == 2
        assert model.n_neighbors == 4

    def test_2d_initialization_parameters(self):
        """Test that parameters are set correctly."""
        model = Ising2D(
            size=16, temperature=2.269, coupling=1.5, boundary="fixed"
        )

        assert model.size == 16
        assert model.temperature == pytest.approx(2.269)
        assert model.coupling == 1.5
        assert model.boundary == "fixed"

    def test_2d_initialization_invalid(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ConfigurationError):
            Ising2D(size=1, temperature=2.0)  # Too small
        with pytest.raises(ConfigurationError):
            Ising2D(size=10, temperature=-1.0)  # Invalid temp


class TestIsing2DStateInitialization:
    """Tests for 2D spin state initialization."""

    def test_2d_initialization_up(self):
        """Test 'up' initialization."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("up")

        assert np.all(model.spins == 1)

    def test_2d_initialization_down(self):
        """Test 'down' initialization."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("down")

        assert np.all(model.spins == -1)

    def test_2d_initialization_random(self):
        """Test random initialization."""
        model = Ising2D(size=20, temperature=2.0)
        model.set_seed(42)
        model.initialize("random")

        # Should have both +1 and -1
        assert np.any(model.spins == 1)
        assert np.any(model.spins == -1)
        # All values should be ±1
        assert np.all(np.abs(model.spins) == 1)

    def test_2d_initialization_checkerboard(self):
        """Test checkerboard initialization."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("checkerboard")

        # Check pattern: (i+j) even -> +1, odd -> -1
        for i in range(model.size):
            for j in range(model.size):
                expected = 1 if (i + j) % 2 == 0 else -1
                assert model.spins[i, j] == expected


class TestIsing2DEnergy:
    """Tests for 2D energy calculations."""

    def test_2d_energy_ground_state_periodic(self):
        """Test energy for ground state (all aligned) with periodic BC.

        For periodic BC: E = -J * 2 * N (2 bonds per spin on average)
        """
        L = 10
        N = L * L
        model = Ising2D(size=L, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # Each spin has 4 neighbors, but each bond counted once
        # Total bonds = 2 * N for periodic (N horizontal + N vertical)
        expected_energy = -1.0 * 2 * N
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_2d_energy_ground_state_fixed(self):
        """Test energy for ground state with fixed BC."""
        L = 10
        model = Ising2D(size=L, temperature=2.0, boundary="fixed")
        model.initialize("up")

        # For fixed BC: (L-1)*L horizontal + L*(L-1) vertical = 2*L*(L-1)
        n_bonds = 2 * L * (L - 1)
        expected_energy = -1.0 * n_bonds
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_2d_energy_antiferromagnetic_periodic(self):
        """Test energy for antiferromagnetic (checkerboard) state.

        All bonds anti-aligned: E = +J * 2 * N
        """
        L = 10
        N = L * L
        model = Ising2D(size=L, temperature=2.0, boundary="periodic")
        model.initialize("checkerboard")

        # All neighbors are anti-aligned
        expected_energy = 1.0 * 2 * N
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_2d_energy_antiferromagnetic_fixed(self):
        """Test energy for checkerboard with fixed BC."""
        L = 10
        model = Ising2D(size=L, temperature=2.0, boundary="fixed")
        model.initialize("checkerboard")

        n_bonds = 2 * L * (L - 1)
        expected_energy = 1.0 * n_bonds
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_2d_energy_with_coupling(self):
        """Test energy scales with coupling constant."""
        L = 10
        N = L * L
        J = 2.0
        model = Ising2D(size=L, temperature=2.0, coupling=J, boundary="periodic")
        model.initialize("up")

        expected_energy = -J * 2 * N
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_2d_energy_per_spin(self):
        """Test energy per spin for ground state."""
        model = Ising2D(size=20, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # e = E/N = -2J for ground state (each spin contributes -J * 4 / 2)
        assert model.get_energy_per_spin() == pytest.approx(-2.0)


class TestIsing2DMagnetization:
    """Tests for 2D magnetization calculations."""

    def test_2d_magnetization_ground_state(self):
        """Test magnetization for all-up state."""
        L = 10
        N = L * L
        model = Ising2D(size=L, temperature=2.0)
        model.initialize("up")

        assert model.get_magnetization() == N
        assert model.get_magnetization_per_spin() == pytest.approx(1.0)

    def test_2d_magnetization_all_down(self):
        """Test magnetization for all-down state."""
        L = 10
        N = L * L
        model = Ising2D(size=L, temperature=2.0)
        model.initialize("down")

        assert model.get_magnetization() == -N
        assert model.get_magnetization_per_spin() == pytest.approx(-1.0)

    def test_2d_magnetization_checkerboard(self):
        """Test magnetization for checkerboard (should be ~0)."""
        # Even size: exactly zero
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("checkerboard")

        assert model.get_magnetization() == 0
        assert model.get_magnetization_per_spin() == pytest.approx(0.0)


class TestIsing2DPeriodicBoundary:
    """Tests for periodic boundary conditions in 2D."""

    def test_2d_periodic_boundary_neighbors(self):
        """Test that periodic BC wraps correctly."""
        model = Ising2D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # Corner (0, 0) should have 4 neighbors due to wrapping
        # Neighbors: (9, 0), (1, 0), (0, 9), (0, 1)
        assert model.get_neighbor_sum((0, 0)) == 4

        # Set corner neighbors to -1 to verify wrapping
        model._spins[9, 0] = -1  # Top wraps to bottom
        model._spins[0, 9] = -1  # Left wraps to right

        # Now sum should be 4 - 2 - 2 = 0 (two +1, two -1)
        assert model.get_neighbor_sum((0, 0)) == 0

    def test_2d_periodic_boundary_energy_consistency(self):
        """Test energy calculation with periodic BC."""
        model = Ising2D(size=5, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # Flip edge spin and verify energy change
        site = (0, 0)
        expected_dE = model.get_energy_change(site)
        initial_E = model.get_energy()

        model.flip_spin(site)
        final_E = model.get_energy()

        assert final_E - initial_E == pytest.approx(expected_dE)


class TestIsing2DEnergyChange:
    """Tests for 2D energy change calculations."""

    def test_2d_energy_change_values_periodic(self):
        """Test that energy changes are only -8, -4, 0, 4, 8 for periodic BC."""
        model = Ising2D(size=20, temperature=2.0, boundary="periodic")
        model.set_seed(42)
        model.initialize("random")

        valid_changes = {-8.0, -4.0, 0.0, 4.0, 8.0}
        observed_changes = set()

        for i in range(model.size):
            for j in range(model.size):
                dE = model.get_energy_change((i, j))
                observed_changes.add(dE)
                assert dE in valid_changes, f"Invalid energy change: {dE}"

        # Should observe multiple values with random config
        assert len(observed_changes) >= 3

    def test_2d_energy_change_all_aligned(self):
        """Test energy change for flipping aligned spin."""
        model = Ising2D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # All neighbors +1, so dE = 2*J*1*4 = 8
        for i in range(model.size):
            for j in range(model.size):
                assert model.get_energy_change((i, j)) == pytest.approx(8.0)

    def test_2d_energy_change_consistency(self):
        """Test that predicted energy change matches actual change."""
        model = Ising2D(size=10, temperature=2.0, boundary="periodic")
        model.set_seed(123)
        model.initialize("random")

        # Test multiple random sites
        for _ in range(20):
            site = model.random_site()
            initial_E = model.get_energy()
            expected_dE = model.get_energy_change(site)

            model.flip_spin(site)
            final_E = model.get_energy()

            assert final_E - initial_E == pytest.approx(expected_dE)

            # Flip back for next iteration
            model.flip_spin(site)


class TestIsing2DFlipSpin:
    """Tests for 2D spin flip operations."""

    def test_2d_flip_preserves_lattice_values(self):
        """Test that flip only changes values to ±1."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")

        # Flip many spins
        for _ in range(100):
            site = model.random_site()
            model.flip_spin(site)

        # All values should still be ±1
        assert np.all(np.abs(model.spins) == 1)

    def test_2d_flip_spin_value(self):
        """Test that flip reverses spin."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("up")

        site = (5, 5)
        original = model.spins[site]
        model.flip_spin(site)

        assert model.spins[site] == -original

    def test_2d_flip_magnetization_change(self):
        """Test magnetization changes by ±2 after flip."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("up")

        initial_mag = model.get_magnetization()
        model.flip_spin((5, 5))
        new_mag = model.get_magnetization()

        assert new_mag - initial_mag == -2


class TestIsing2DNeighborSum:
    """Tests for 2D neighbor sum calculations."""

    def test_2d_neighbor_sum_all_up(self):
        """Test neighbor sum when all spins up."""
        model = Ising2D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("up")

        # All neighbors +1, so sum is 4
        for i in range(model.size):
            for j in range(model.size):
                assert model.get_neighbor_sum((i, j)) == 4

    def test_2d_neighbor_sum_checkerboard(self):
        """Test neighbor sum for checkerboard pattern."""
        model = Ising2D(size=10, temperature=2.0, boundary="periodic")
        model.initialize("checkerboard")

        # Each spin has 4 anti-aligned neighbors
        for i in range(model.size):
            for j in range(model.size):
                spin = model.spins[i, j]
                neighbor_sum = model.get_neighbor_sum((i, j))
                assert neighbor_sum == -4 * spin

    def test_2d_neighbor_sum_fixed_bc(self):
        """Test neighbor sum for fixed BC (edges have fewer neighbors)."""
        model = Ising2D(size=10, temperature=2.0, boundary="fixed")
        model.initialize("up")

        # Corner has 2 neighbors
        assert model.get_neighbor_sum((0, 0)) == 2
        assert model.get_neighbor_sum((9, 9)) == 2

        # Edge (non-corner) has 3 neighbors
        assert model.get_neighbor_sum((0, 5)) == 3
        assert model.get_neighbor_sum((5, 0)) == 3

        # Interior has 4 neighbors
        assert model.get_neighbor_sum((5, 5)) == 4


class TestIsing2DCopy:
    """Tests for 2D model copying."""

    def test_2d_copy_independence(self):
        """Test that copy is independent."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("random")

        copy = model.copy()
        model.flip_spin((5, 5))

        assert copy.spins[5, 5] != model.spins[5, 5]

    def test_2d_copy_parameters(self):
        """Test that copy has same parameters."""
        model = Ising2D(size=16, temperature=2.5, coupling=1.5, boundary="fixed")
        model.initialize("up")

        copy = model.copy()

        assert copy.size == model.size
        assert copy.temperature == model.temperature
        assert copy.coupling == model.coupling
        assert copy.boundary == model.boundary
        np.testing.assert_array_equal(copy.spins, model.spins)


class TestIsing2DRandomSite:
    """Tests for 2D random site selection."""

    def test_2d_random_site_range(self):
        """Test random sites are in valid range."""
        model = Ising2D(size=10, temperature=2.0)

        for _ in range(100):
            site = model.random_site()
            assert len(site) == 2
            assert 0 <= site[0] < model.size
            assert 0 <= site[1] < model.size

    def test_2d_random_site_tuple(self):
        """Test that random_site returns a tuple."""
        model = Ising2D(size=10, temperature=2.0)
        site = model.random_site()

        assert isinstance(site, tuple)
        assert len(site) == 2


class TestIsing2DConfigurationImage:
    """Tests for configuration image output."""

    def test_2d_configuration_image_shape(self):
        """Test image has correct shape."""
        model = Ising2D(size=32, temperature=2.0)
        model.initialize("random")

        img = model.get_configuration_image()

        assert img.shape == (32, 32)

    def test_2d_configuration_image_values(self):
        """Test image contains only ±1."""
        model = Ising2D(size=32, temperature=2.0)
        model.initialize("random")

        img = model.get_configuration_image()

        assert np.all(np.abs(img) == 1)

    def test_2d_configuration_image_copy(self):
        """Test image is a copy, not a view."""
        model = Ising2D(size=10, temperature=2.0)
        model.initialize("up")

        img = model.get_configuration_image()
        img[0, 0] = -1

        # Original should be unchanged
        assert model.spins[0, 0] == 1


class TestIsing2DCriticalTemperature:
    """Tests for critical temperature awareness."""

    def test_2d_is_near_critical(self):
        """Test detection of near-critical temperature."""
        model = Ising2D(size=10, temperature=CRITICAL_TEMP_2D)

        assert model.is_near_critical(tolerance=0.01)

    def test_2d_not_near_critical(self):
        """Test detection of far-from-critical temperature."""
        model = Ising2D(size=10, temperature=1.0)  # Well below Tc
        assert not model.is_near_critical(tolerance=0.1)

        model = Ising2D(size=10, temperature=4.0)  # Well above Tc
        assert not model.is_near_critical(tolerance=0.1)


class TestIsing2DRepr:
    """Tests for 2D string representation."""

    def test_2d_repr(self):
        """Test __repr__ output."""
        model = Ising2D(size=16, temperature=2.269, boundary="periodic")
        repr_str = repr(model)

        assert "Ising2D" in repr_str
        assert "size=16" in repr_str
        assert "n_spins=256" in repr_str
        assert "periodic" in repr_str


# =============================================================================
# 3D Ising Model Tests
# =============================================================================


class TestIsing3DInitialization:
    """Tests for Ising3D initialization."""

    def test_3d_initialization(self):
        """Test that 3D model initializes correctly."""
        model = Ising3D(size=8, temperature=4.0)

        assert model.size == 8
        assert model.n_spins == 512  # 8^3
        assert model.shape == (8, 8, 8)
        assert model.spins.shape == (8, 8, 8)
        assert model.dimension == 3

    def test_3d_n_neighbors(self):
        """Test that 3D model has 6 neighbors per spin."""
        model = Ising3D(size=8, temperature=4.0)
        assert model.n_neighbors == 6

    def test_3d_initialization_parameters(self):
        """Test that parameters are set correctly."""
        model = Ising3D(
            size=10, temperature=4.511, coupling=1.5, boundary="fixed"
        )

        assert model.size == 10
        assert model.temperature == pytest.approx(4.511)
        assert model.coupling == 1.5
        assert model.boundary == "fixed"

    def test_3d_initialization_invalid(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ConfigurationError):
            Ising3D(size=1, temperature=4.0)  # Too small
        with pytest.raises(ConfigurationError):
            Ising3D(size=8, temperature=-1.0)  # Invalid temp


class TestIsing3DStateInitialization:
    """Tests for 3D spin state initialization."""

    def test_3d_initialization_up(self):
        """Test 'up' initialization."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("up")

        assert np.all(model.spins == 1)

    def test_3d_initialization_down(self):
        """Test 'down' initialization."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("down")

        assert np.all(model.spins == -1)

    def test_3d_initialization_random(self):
        """Test random initialization."""
        model = Ising3D(size=10, temperature=4.0)
        model.set_seed(42)
        model.initialize("random")

        # Should have both +1 and -1
        assert np.any(model.spins == 1)
        assert np.any(model.spins == -1)
        # All values should be ±1
        assert np.all(np.abs(model.spins) == 1)

    def test_3d_initialization_checkerboard(self):
        """Test checkerboard initialization (3D Néel state)."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("checkerboard")

        # Check pattern: (i+j+k) even -> +1, odd -> -1
        for i in range(model.size):
            for j in range(model.size):
                for k in range(model.size):
                    expected = 1 if (i + j + k) % 2 == 0 else -1
                    assert model.spins[i, j, k] == expected


class TestIsing3DEnergy:
    """Tests for 3D energy calculations."""

    def test_3d_energy_ground_state_periodic(self):
        """Test energy for ground state (all aligned) with periodic BC.

        For periodic BC: E = -J * 3 * N (3 bonds per spin on average)
        """
        L = 8
        N = L ** 3
        model = Ising3D(size=L, temperature=4.0, boundary="periodic")
        model.initialize("up")

        # Total bonds = 3 * N for periodic (N in each of x, y, z directions)
        expected_energy = -1.0 * 3 * N
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_3d_energy_ground_state_fixed(self):
        """Test energy for ground state with fixed BC."""
        L = 8
        model = Ising3D(size=L, temperature=4.0, boundary="fixed")
        model.initialize("up")

        # For fixed BC: 3 * L^2 * (L-1) bonds
        n_bonds = 3 * L * L * (L - 1)
        expected_energy = -1.0 * n_bonds
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_3d_energy_antiferromagnetic_periodic(self):
        """Test energy for antiferromagnetic (checkerboard) state.

        All bonds anti-aligned: E = +J * 3 * N
        """
        L = 8
        N = L ** 3
        model = Ising3D(size=L, temperature=4.0, boundary="periodic")
        model.initialize("checkerboard")

        # All neighbors are anti-aligned
        expected_energy = 1.0 * 3 * N
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_3d_energy_with_coupling(self):
        """Test energy scales with coupling constant."""
        L = 6
        N = L ** 3
        J = 2.0
        model = Ising3D(size=L, temperature=4.0, coupling=J, boundary="periodic")
        model.initialize("up")

        expected_energy = -J * 3 * N
        assert model.get_energy() == pytest.approx(expected_energy)

    def test_3d_energy_per_spin(self):
        """Test energy per spin for ground state."""
        model = Ising3D(size=10, temperature=4.0, boundary="periodic")
        model.initialize("up")

        # e = E/N = -3J for ground state
        assert model.get_energy_per_spin() == pytest.approx(-3.0)


class TestIsing3DMagnetization:
    """Tests for 3D magnetization calculations."""

    def test_3d_magnetization_ground_state(self):
        """Test magnetization for all-up state."""
        L = 8
        N = L ** 3
        model = Ising3D(size=L, temperature=4.0)
        model.initialize("up")

        assert model.get_magnetization() == N
        assert model.get_magnetization_per_spin() == pytest.approx(1.0)

    def test_3d_magnetization_checkerboard(self):
        """Test magnetization for checkerboard (should be 0 for even size)."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("checkerboard")

        assert model.get_magnetization() == 0
        assert model.get_magnetization_per_spin() == pytest.approx(0.0)


class TestIsing3DEnergyChange:
    """Tests for 3D energy change calculations."""

    def test_3d_energy_change_values(self):
        """Test that energy changes are only -12, -8, -4, 0, 4, 8, 12."""
        model = Ising3D(size=10, temperature=4.0, boundary="periodic")
        model.set_seed(42)
        model.initialize("random")

        valid_changes = {-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0}
        observed_changes = set()

        # Sample many sites
        for _ in range(500):
            site = model.random_site()
            dE = model.get_energy_change(site)
            observed_changes.add(dE)
            assert dE in valid_changes, f"Invalid energy change: {dE}"

        # Should observe multiple values with random config
        assert len(observed_changes) >= 4

    def test_3d_energy_change_all_aligned(self):
        """Test energy change for flipping aligned spin."""
        model = Ising3D(size=8, temperature=4.0, boundary="periodic")
        model.initialize("up")

        # All neighbors +1, so dE = 2*J*1*6 = 12
        site = (4, 4, 4)
        assert model.get_energy_change(site) == pytest.approx(12.0)

    def test_3d_energy_change_consistency(self):
        """Test that predicted energy change matches actual change."""
        model = Ising3D(size=8, temperature=4.0, boundary="periodic")
        model.set_seed(123)
        model.initialize("random")

        # Test multiple random sites
        for _ in range(20):
            site = model.random_site()
            initial_E = model.get_energy()
            expected_dE = model.get_energy_change(site)

            model.flip_spin(site)
            final_E = model.get_energy()

            assert final_E - initial_E == pytest.approx(expected_dE)

            # Flip back for next iteration
            model.flip_spin(site)


class TestIsing3DNeighborSum:
    """Tests for 3D neighbor sum calculations."""

    def test_3d_neighbor_sum_all_up(self):
        """Test neighbor sum when all spins up."""
        model = Ising3D(size=8, temperature=4.0, boundary="periodic")
        model.initialize("up")

        # All neighbors +1, so sum is 6
        site = (4, 4, 4)
        assert model.get_neighbor_sum(site) == 6

    def test_3d_neighbor_sum_checkerboard(self):
        """Test neighbor sum for checkerboard pattern."""
        model = Ising3D(size=8, temperature=4.0, boundary="periodic")
        model.initialize("checkerboard")

        # Each spin has 6 anti-aligned neighbors
        site = (4, 4, 4)
        spin = model.spins[site]
        neighbor_sum = model.get_neighbor_sum(site)
        assert neighbor_sum == -6 * spin

    def test_3d_neighbor_sum_fixed_bc(self):
        """Test neighbor sum for fixed BC."""
        model = Ising3D(size=8, temperature=4.0, boundary="fixed")
        model.initialize("up")

        # Corner has 3 neighbors
        assert model.get_neighbor_sum((0, 0, 0)) == 3

        # Edge has 4 neighbors
        assert model.get_neighbor_sum((0, 0, 4)) == 4

        # Face has 5 neighbors
        assert model.get_neighbor_sum((0, 4, 4)) == 5

        # Interior has 6 neighbors
        assert model.get_neighbor_sum((4, 4, 4)) == 6


class TestIsing3DFlipSpin:
    """Tests for 3D spin flip operations."""

    def test_3d_flip_preserves_lattice_values(self):
        """Test that flip only changes values to ±1."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("random")

        # Flip many spins
        for _ in range(100):
            site = model.random_site()
            model.flip_spin(site)

        # All values should still be ±1
        assert np.all(np.abs(model.spins) == 1)

    def test_3d_flip_spin_value(self):
        """Test that flip reverses spin."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("up")

        site = (4, 4, 4)
        original = model.spins[site]
        model.flip_spin(site)

        assert model.spins[site] == -original


class TestIsing3DCopy:
    """Tests for 3D model copying."""

    def test_3d_copy_independence(self):
        """Test that copy is independent."""
        model = Ising3D(size=8, temperature=4.0)
        model.initialize("random")

        copy = model.copy()
        model.flip_spin((4, 4, 4))

        assert copy.spins[4, 4, 4] != model.spins[4, 4, 4]

    def test_3d_copy_parameters(self):
        """Test that copy has same parameters."""
        model = Ising3D(size=8, temperature=4.5, coupling=1.5, boundary="fixed")
        model.initialize("up")

        copy = model.copy()

        assert copy.size == model.size
        assert copy.temperature == model.temperature
        assert copy.coupling == model.coupling
        assert copy.boundary == model.boundary
        np.testing.assert_array_equal(copy.spins, model.spins)


class TestIsing3DRandomSite:
    """Tests for 3D random site selection."""

    def test_3d_random_site_range(self):
        """Test random sites are in valid range."""
        model = Ising3D(size=8, temperature=4.0)

        for _ in range(100):
            site = model.random_site()
            assert len(site) == 3
            assert 0 <= site[0] < model.size
            assert 0 <= site[1] < model.size
            assert 0 <= site[2] < model.size

    def test_3d_random_site_tuple(self):
        """Test that random_site returns a tuple."""
        model = Ising3D(size=8, temperature=4.0)
        site = model.random_site()

        assert isinstance(site, tuple)
        assert len(site) == 3


class TestIsing3DConfigurationSlice:
    """Tests for 3D configuration slice output."""

    def test_3d_configuration_slice_shape(self):
        """Test slice has correct shape."""
        model = Ising3D(size=16, temperature=4.0)
        model.initialize("random")

        slice_xy = model.get_configuration_slice(axis=2, index=8)
        assert slice_xy.shape == (16, 16)

        slice_xz = model.get_configuration_slice(axis=1, index=8)
        assert slice_xz.shape == (16, 16)

        slice_yz = model.get_configuration_slice(axis=0, index=8)
        assert slice_yz.shape == (16, 16)

    def test_3d_configuration_slice_values(self):
        """Test slice contains only ±1."""
        model = Ising3D(size=16, temperature=4.0)
        model.initialize("random")

        slice_xy = model.get_configuration_slice()
        assert np.all(np.abs(slice_xy) == 1)


class TestIsing3DCriticalTemperature:
    """Tests for 3D critical temperature awareness."""

    def test_3d_is_near_critical(self):
        """Test detection of near-critical temperature."""
        model = Ising3D(size=8, temperature=CRITICAL_TEMP_3D)

        assert model.is_near_critical(tolerance=0.01)

    def test_3d_not_near_critical(self):
        """Test detection of far-from-critical temperature."""
        model = Ising3D(size=8, temperature=2.0)  # Well below Tc
        assert not model.is_near_critical(tolerance=0.1)

        model = Ising3D(size=8, temperature=8.0)  # Well above Tc
        assert not model.is_near_critical(tolerance=0.1)


class TestIsing3DRepr:
    """Tests for 3D string representation."""

    def test_3d_repr(self):
        """Test __repr__ output."""
        model = Ising3D(size=8, temperature=4.511, boundary="periodic")
        repr_str = repr(model)

        assert "Ising3D" in repr_str
        assert "size=8" in repr_str
        assert "n_spins=512" in repr_str
        assert "periodic" in repr_str


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateModel:
    """Tests for the create_model factory function."""

    def test_create_ising1d(self):
        """Test creating 1D model with 'ising1d'."""
        model = create_model("ising1d", size=10, temperature=2.0)

        assert isinstance(model, Ising1D)
        assert model.size == 10
        assert model.temperature == 2.0

    def test_create_ising2d(self):
        """Test creating 2D model with 'ising2d'."""
        model = create_model("ising2d", size=16, temperature=2.269)

        assert isinstance(model, Ising2D)
        assert model.size == 16
        assert model.n_spins == 256

    def test_create_ising3d(self):
        """Test creating 3D model with 'ising3d'."""
        model = create_model("ising3d", size=8, temperature=4.511)

        assert isinstance(model, Ising3D)
        assert model.size == 8
        assert model.n_spins == 512

    def test_create_shorthand_1d(self):
        """Test creating 1D model with shorthand '1d'."""
        model = create_model("1d", size=20, temperature=1.5)

        assert isinstance(model, Ising1D)
        assert model.size == 20

    def test_create_shorthand_2d(self):
        """Test creating 2D model with shorthand '2d'."""
        model = create_model("2d", size=32, temperature=2.0)

        assert isinstance(model, Ising2D)
        assert model.size == 32

    def test_create_shorthand_3d(self):
        """Test creating 3D model with shorthand '3d'."""
        model = create_model("3d", size=10, temperature=4.0)

        assert isinstance(model, Ising3D)
        assert model.size == 10

    def test_create_case_insensitive(self):
        """Test that model type is case-insensitive."""
        model1 = create_model("ISING2D", size=8, temperature=2.0)
        model2 = create_model("Ising2D", size=8, temperature=2.0)
        model3 = create_model("ising2d", size=8, temperature=2.0)

        assert isinstance(model1, Ising2D)
        assert isinstance(model2, Ising2D)
        assert isinstance(model3, Ising2D)

    def test_create_with_all_parameters(self):
        """Test creating model with all optional parameters."""
        model = create_model(
            "2d",
            size=16,
            temperature=2.5,
            coupling=1.5,
            boundary="fixed"
        )

        assert isinstance(model, Ising2D)
        assert model.size == 16
        assert model.temperature == 2.5
        assert model.coupling == 1.5
        assert model.boundary == "fixed"

    def test_create_invalid_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("ising4d", size=8, temperature=2.0)

        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("invalid", size=8, temperature=2.0)

    def test_create_error_message_shows_valid_types(self):
        """Test that error message includes valid options."""
        with pytest.raises(ValueError) as exc_info:
            create_model("bad_type", size=8, temperature=2.0)

        error_msg = str(exc_info.value)
        assert "1d" in error_msg
        assert "2d" in error_msg
        assert "3d" in error_msg
