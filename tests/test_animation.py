"""Tests for animation module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if matplotlib not available
plt = pytest.importorskip("matplotlib.pyplot")

from ising_toolkit.visualization.animation import (  # noqa: E402
    create_spin_animation,
    create_observable_animation,
    create_temperature_sweep_animation,
    create_domain_growth_animation,
    _check_animation_dependencies,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_configurations():
    """Create sample spin configurations for animation."""
    np.random.seed(42)
    L = 16
    n_frames = 10

    configs = []
    for i in range(n_frames):
        # Simulate evolution: start ordered, become disordered
        p = i / (n_frames - 1)  # 0 to 1
        spins = np.ones((L, L), dtype=int)
        # Randomly flip some spins
        mask = np.random.random((L, L)) < p * 0.5
        spins[mask] = -1
        configs.append(spins)

    return configs


@pytest.fixture
def sample_configurations_3d():
    """Create sample 3D configurations (for testing error handling)."""
    np.random.seed(42)
    return [np.random.choice([-1, 1], size=(8, 8, 8)) for _ in range(5)]


@pytest.fixture
def sample_observable_data(sample_configurations):
    """Create sample observable data for animation."""
    n_frames = len(sample_configurations)
    times = np.arange(n_frames) * 100  # MC steps
    # Magnetization decreasing over time
    observable = np.array([np.mean(c) for c in sample_configurations])
    return times, observable


@pytest.fixture
def sample_temperature_data():
    """Create sample temperature sweep data."""
    np.random.seed(42)
    L = 16
    temperatures = np.linspace(1.5, 3.5, 10)

    configs = []
    for T in temperatures:
        # Low T: mostly ordered, High T: disordered
        if T < 2.27:
            spins = np.ones((L, L), dtype=int)
            # Add some noise
            mask = np.random.random((L, L)) < 0.1
            spins[mask] = -1
        else:
            spins = np.random.choice([-1, 1], size=(L, L))
        configs.append(spins)

    return temperatures, configs


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests for create_spin_animation
# ============================================================================

class TestCreateSpinAnimation:
    """Tests for create_spin_animation function."""

    def test_creates_gif(self, sample_configurations, output_dir):
        """Test creating a GIF animation."""
        pytest.importorskip("PIL")

        output_path = output_dir / "spin_evolution.gif"

        create_spin_animation(
            sample_configurations,
            str(output_path),
            fps=5,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_parameters(self, sample_configurations, output_dir):
        """Test with custom parameters."""
        pytest.importorskip("PIL")

        output_path = output_dir / "custom.gif"

        create_spin_animation(
            sample_configurations,
            str(output_path),
            fps=2,
            cmap='coolwarm',
            figsize=(4, 4),
            dpi=50,
            title_template='Frame {step}',
            add_colorbar=True,
        )

        assert output_path.exists()

    def test_empty_configurations_raises(self, output_dir):
        """Test error for empty configuration list."""
        with pytest.raises(ValueError, match="At least one"):
            create_spin_animation([], str(output_dir / "empty.gif"))

    def test_wrong_dimensions_raises(self, sample_configurations_3d, output_dir):
        """Test error for non-2D configurations."""
        with pytest.raises(ValueError, match="2D"):
            create_spin_animation(
                sample_configurations_3d,
                str(output_dir / "3d.gif")
            )

    def test_single_frame(self, output_dir):
        """Test animation with single frame."""
        pytest.importorskip("PIL")

        config = [np.random.choice([-1, 1], size=(8, 8))]
        output_path = output_dir / "single.gif"

        create_spin_animation(config, str(output_path))

        assert output_path.exists()

    def test_title_template(self, sample_configurations, output_dir):
        """Test custom title template."""
        pytest.importorskip("PIL")

        output_path = output_dir / "titled.gif"

        create_spin_animation(
            sample_configurations,
            str(output_path),
            title_template='T=2.27, MC step {step}',
        )

        assert output_path.exists()


# ============================================================================
# Tests for create_observable_animation
# ============================================================================

class TestCreateObservableAnimation:
    """Tests for create_observable_animation function."""

    def test_creates_gif(self, sample_configurations, sample_observable_data, output_dir):
        """Test creating observable animation."""
        pytest.importorskip("PIL")

        times, observable = sample_observable_data
        output_path = output_dir / "observable.gif"

        create_observable_animation(
            times,
            observable,
            sample_configurations,
            str(output_path),
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_custom_observable_name(
        self, sample_configurations, sample_observable_data, output_dir
    ):
        """Test custom observable name."""
        pytest.importorskip("PIL")

        times, observable = sample_observable_data
        output_path = output_dir / "energy.gif"

        create_observable_animation(
            times,
            observable,
            sample_configurations,
            str(output_path),
            observable_name='Energy per spin',
        )

        assert output_path.exists()

    def test_mismatched_lengths_raises(self, sample_configurations, output_dir):
        """Test error for mismatched array lengths."""
        times = np.arange(5)  # Wrong length
        observable = np.random.random(5)

        with pytest.raises(ValueError, match="same length"):
            create_observable_animation(
                times,
                observable,
                sample_configurations,
                str(output_dir / "mismatch.gif"),
            )

    def test_empty_configurations_raises(self, output_dir):
        """Test error for empty configurations."""
        with pytest.raises(ValueError, match="At least one"):
            create_observable_animation(
                np.array([]),
                np.array([]),
                [],
                str(output_dir / "empty.gif"),
            )


# ============================================================================
# Tests for create_temperature_sweep_animation
# ============================================================================

class TestCreateTemperatureSweepAnimation:
    """Tests for create_temperature_sweep_animation function."""

    def test_creates_gif(self, sample_temperature_data, output_dir):
        """Test creating temperature sweep animation."""
        pytest.importorskip("PIL")

        temperatures, configs = sample_temperature_data
        output_path = output_dir / "sweep.gif"

        create_temperature_sweep_animation(
            temperatures,
            configs,
            str(output_path),
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_with_Tc(self, sample_temperature_data, output_dir):
        """Test with critical temperature marked."""
        pytest.importorskip("PIL")

        temperatures, configs = sample_temperature_data
        output_path = output_dir / "sweep_tc.gif"

        create_temperature_sweep_animation(
            temperatures,
            configs,
            str(output_path),
            Tc=2.269,
        )

        assert output_path.exists()

    def test_mismatched_lengths_raises(self, output_dir):
        """Test error for mismatched array lengths."""
        temps = np.linspace(1.5, 3.5, 5)
        configs = [np.ones((8, 8)) for _ in range(3)]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            create_temperature_sweep_animation(
                temps,
                configs,
                str(output_dir / "mismatch.gif"),
            )


# ============================================================================
# Tests for create_domain_growth_animation
# ============================================================================

class TestCreateDomainGrowthAnimation:
    """Tests for create_domain_growth_animation function."""

    def test_creates_gif(self, sample_configurations, output_dir):
        """Test creating domain growth animation."""
        pytest.importorskip("PIL")

        output_path = output_dir / "domain.gif"

        create_domain_growth_animation(
            sample_configurations,
            str(output_path),
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_with_times(self, sample_configurations, output_dir):
        """Test with explicit time array."""
        pytest.importorskip("PIL")

        times = np.arange(len(sample_configurations)) * 50
        output_path = output_dir / "domain_times.gif"

        create_domain_growth_animation(
            sample_configurations,
            str(output_path),
            times=times,
        )

        assert output_path.exists()

    def test_empty_configurations_raises(self, output_dir):
        """Test error for empty configurations."""
        with pytest.raises(ValueError, match="At least one"):
            create_domain_growth_animation(
                [],
                str(output_dir / "empty.gif"),
            )

    def test_mismatched_times_raises(self, sample_configurations, output_dir):
        """Test error for mismatched times."""
        times = np.arange(3)  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            create_domain_growth_animation(
                sample_configurations,
                str(output_dir / "mismatch.gif"),
                times=times,
            )


# ============================================================================
# Tests for dependency checking
# ============================================================================

class TestDependencyCheck:
    """Tests for _check_animation_dependencies."""

    def test_returns_dict(self):
        """Test returns dictionary with expected keys."""
        result = _check_animation_dependencies()

        assert isinstance(result, dict)
        assert 'gif' in result
        assert 'mp4' in result

    def test_gif_available_with_pillow(self):
        """Test GIF availability when Pillow is installed."""
        try:
            import PIL  # noqa: F401
            result = _check_animation_dependencies()
            assert result['gif'] is True
        except ImportError:
            pytest.skip("Pillow not installed")


# ============================================================================
# Integration Tests
# ============================================================================

class TestAnimationIntegration:
    """Integration tests for animation functions."""

    def test_full_simulation_animation(self, output_dir):
        """Test creating animation from simulated evolution."""
        pytest.importorskip("PIL")

        np.random.seed(42)
        L = 16
        n_steps = 20

        # Simulate spin dynamics (simplified)
        configs = []
        spins = np.random.choice([-1, 1], size=(L, L))

        for step in range(n_steps):
            # Simple dynamics: random flips
            i, j = np.random.randint(0, L, 2)
            spins[i, j] *= -1
            if step % 2 == 0:
                configs.append(spins.copy())

        output_path = output_dir / "simulation.gif"
        create_spin_animation(configs, str(output_path), fps=5)

        assert output_path.exists()

    def test_animation_with_observables(self, output_dir):
        """Test observable animation with realistic data."""
        pytest.importorskip("PIL")

        np.random.seed(42)
        L = 16
        n_frames = 15

        configs = []
        magnetizations = []
        times = []

        spins = np.ones((L, L), dtype=int)

        for i in range(n_frames):
            # Flip random spins
            n_flips = L
            for _ in range(n_flips):
                x, y = np.random.randint(0, L, 2)
                spins[x, y] *= -1

            configs.append(spins.copy())
            magnetizations.append(np.mean(spins))
            times.append(i * 100)

        output_path = output_dir / "with_obs.gif"
        create_observable_animation(
            np.array(times),
            np.array(magnetizations),
            configs,
            str(output_path),
            observable_name='Magnetization',
            fps=5,
        )

        assert output_path.exists()

    def test_multiple_animations_same_data(
        self, sample_configurations, sample_observable_data, output_dir
    ):
        """Test creating multiple animations from same data."""
        pytest.importorskip("PIL")

        times, observable = sample_observable_data

        # Create different animation types
        create_spin_animation(
            sample_configurations,
            str(output_dir / "anim1.gif"),
            fps=5,
        )

        create_observable_animation(
            times,
            observable,
            sample_configurations,
            str(output_dir / "anim2.gif"),
            fps=5,
        )

        create_domain_growth_animation(
            sample_configurations,
            str(output_dir / "anim3.gif"),
            fps=5,
        )

        assert (output_dir / "anim1.gif").exists()
        assert (output_dir / "anim2.gif").exists()
        assert (output_dir / "anim3.gif").exists()
