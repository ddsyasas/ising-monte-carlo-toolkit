"""Tests for visualization plotting functions."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if matplotlib not available
plt = pytest.importorskip("matplotlib.pyplot")
pd = pytest.importorskip("pandas")

from ising_toolkit.visualization import (
    plot_observable_vs_temperature,
    plot_phase_diagram,
    plot_binder_cumulant,
    plot_scaling_collapse,
    plot_configuration,
    plot_spin_configuration,
    plot_configuration_comparison,
    plot_spin_configuration_3d,
    plot_spin_configuration_3d_slices,
    plot_energy_histogram,
    plot_time_series,
    # Analysis-specific plots
    plot_autocorrelation,
    plot_blocking_analysis,
    plot_correlation_time_comparison,
    plot_equilibration_check,
    plot_bootstrap_distribution,
    plot_finite_size_scaling,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_sweep_data():
    """Create sample temperature sweep data for testing."""
    np.random.seed(42)

    sizes = [8, 16, 32]
    temperatures = np.linspace(1.5, 3.5, 21)
    Tc = 2.269

    rows = []
    for size in sizes:
        for T in temperatures:
            # Approximate physics behavior
            t = (T - Tc) / Tc

            if T < Tc:
                mag = (1 - T / Tc) ** 0.125 + np.random.normal(0, 0.02)
                binder = 2/3 - 0.1 * t + np.random.normal(0, 0.01)
            else:
                mag = np.random.normal(0, 0.1) * (size ** -0.5)
                binder = t * 0.5 + np.random.normal(0, 0.01)

            energy = -2 + 0.5 * T + np.random.normal(0, 0.05)
            chi = size ** 1.75 * np.exp(-abs(t) * 5) + np.random.normal(0, 0.1)
            cv = size * np.exp(-abs(t) * 5) + np.random.normal(0, 0.1)

            rows.append({
                'temperature': T,
                'size': size,
                'energy_mean': energy,
                'energy_err': 0.01,
                'magnetization_mean': mag,
                'magnetization_err': 0.01,
                'abs_magnetization_mean': abs(mag),
                'abs_magnetization_err': 0.01,
                'heat_capacity': cv,
                'susceptibility': chi,
                'binder_cumulant': np.clip(binder, 0, 2/3),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_single_size_data():
    """Create sample data without size column."""
    np.random.seed(42)

    temperatures = np.linspace(1.5, 3.5, 21)
    Tc = 2.269

    rows = []
    for T in temperatures:
        t = (T - Tc) / Tc
        energy = -2 + 0.5 * T + np.random.normal(0, 0.05)
        mag = max(0, (1 - T / Tc)) ** 0.125 if T < Tc else 0.01

        rows.append({
            'temperature': T,
            'energy_mean': energy,
            'magnetization_mean': mag,
            'heat_capacity': np.exp(-abs(t) * 5) + 0.5,
            'susceptibility': np.exp(-abs(t) * 5) * 10,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_spin_config():
    """Create sample 2D spin configuration."""
    np.random.seed(42)
    L = 32
    spins = np.random.choice([-1, 1], size=(L, L))
    # Add some domain structure
    spins[:L//2, :L//2] = 1
    spins[L//2:, L//2:] = 1
    return spins


@pytest.fixture
def sample_energy_series():
    """Create sample energy time series."""
    np.random.seed(42)
    n = 1000
    # Simulate equilibrated energy fluctuations
    energy = -1.7 + np.cumsum(np.random.normal(0, 0.01, n)) * 0.01
    return energy


@pytest.fixture
def sample_spin_config_3d():
    """Create sample 3D spin configuration."""
    np.random.seed(42)
    L = 16
    spins = np.random.choice([-1, 1], size=(L, L, L))
    # Add some domain structure
    spins[:L//2, :L//2, :L//2] = 1
    spins[L//2:, L//2:, L//2:] = 1
    return spins


@pytest.fixture
def multiple_spin_configs():
    """Create multiple 2D spin configurations for comparison tests."""
    np.random.seed(42)
    L = 32

    # Ordered (low T)
    ordered = np.ones((L, L), dtype=int)
    ordered[:3, :3] = -1  # Small defect

    # Critical (T ~ Tc) - mixed domains
    critical = np.random.choice([-1, 1], size=(L, L))

    # Disordered (high T)
    disordered = np.random.choice([-1, 1], size=(L, L))

    return [ordered, critical, disordered]


@pytest.fixture
def reference_dir():
    """Get/create reference images directory."""
    ref_dir = Path(__file__).parent / "reference_images"
    ref_dir.mkdir(exist_ok=True)
    return ref_dir


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests for plot_observable_vs_temperature
# ============================================================================

class TestPlotObservableVsTemperature:
    """Tests for plot_observable_vs_temperature function."""

    def test_basic_plot(self, sample_sweep_data):
        """Test basic plot creation."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity'
        )

        assert ax is not None
        assert len(ax.lines) > 0
        plt.close()

    def test_returns_axes(self, sample_sweep_data):
        """Test function returns axes object."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'susceptibility'
        )

        assert hasattr(ax, 'plot')
        assert hasattr(ax, 'set_xlabel')
        plt.close()

    def test_custom_axes(self, sample_sweep_data):
        """Test plotting on provided axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_size_filtering(self, sample_sweep_data):
        """Test filtering by specific sizes."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            sizes=[8, 16]
        )

        # Should have curves for each size (legend also includes Tc line)
        legend = ax.get_legend()
        assert legend is not None
        size_labels = [t.get_text() for t in legend.texts if 'L =' in t.get_text()]
        assert len(size_labels) == 2
        plt.close()

    def test_single_size_data(self, sample_single_size_data):
        """Test with single-size data (no size column)."""
        ax = plot_observable_vs_temperature(
            sample_single_size_data,
            'heat_capacity'
        )

        assert ax is not None
        assert len(ax.lines) >= 1
        plt.close()

    def test_show_Tc_line(self, sample_sweep_data):
        """Test critical temperature line is shown."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            show_Tc=True
        )

        # Should have vertical line for Tc
        # Lines include data curves + critical line
        assert len(ax.lines) >= 3  # 3 sizes + Tc line
        plt.close()

    def test_custom_Tc(self, sample_sweep_data):
        """Test custom critical temperature."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            show_Tc=True,
            Tc=2.5
        )

        assert ax is not None
        plt.close()

    def test_no_Tc_line(self, sample_sweep_data):
        """Test plot without Tc line."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            show_Tc=False
        )

        # Should have only 3 lines (one per size)
        # Note: errorbar creates multiple line objects
        assert ax is not None
        plt.close()

    def test_error_bars(self, sample_sweep_data):
        """Test error bars are shown when available."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'energy_mean',
            show_errors=True
        )

        # Check errorbar containers exist
        assert ax is not None
        plt.close()

    def test_no_error_bars(self, sample_sweep_data):
        """Test plot without error bars."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'energy_mean',
            show_errors=False
        )

        assert ax is not None
        plt.close()

    def test_correct_labels(self, sample_sweep_data):
        """Test axes have correct labels."""
        ax = plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity'
        )

        assert 'Temperature' in ax.get_xlabel()
        assert 'Heat capacity' in ax.get_ylabel() or 'C' in ax.get_ylabel()
        plt.close()

    def test_save_to_file(self, sample_sweep_data, output_dir):
        """Test saving plot to file."""
        save_path = output_dir / "observable_plot.png"

        # Need to create figure manually to save
        fig, ax = plt.subplots()
        plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            ax=ax
        )
        fig.savefig(save_path)
        plt.close(fig)

        assert save_path.exists()
        assert save_path.stat().st_size > 0


# ============================================================================
# Tests for plot_phase_diagram
# ============================================================================

class TestPlotPhaseDiagram:
    """Tests for plot_phase_diagram function."""

    def test_basic_phase_diagram(self, sample_sweep_data):
        """Test basic phase diagram creation."""
        fig, axes = plot_phase_diagram(sample_sweep_data)

        assert fig is not None
        assert len(axes) >= 1
        plt.close(fig)

    def test_default_observables(self, sample_sweep_data):
        """Test default observables are plotted."""
        fig, axes = plot_phase_diagram(sample_sweep_data)

        # Default should include 4 observables
        assert len(axes) == 4
        plt.close(fig)

    def test_custom_observables(self, sample_sweep_data):
        """Test custom observable selection."""
        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            observables=['heat_capacity', 'susceptibility']
        )

        assert len(axes) == 2
        plt.close(fig)

    def test_single_observable(self, sample_sweep_data):
        """Test single observable plot."""
        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            observables=['heat_capacity']
        )

        assert len(axes) == 1
        plt.close(fig)

    def test_size_filtering(self, sample_sweep_data):
        """Test size filtering in phase diagram."""
        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            sizes=[8, 16]
        )

        # Each subplot should have 2 curves for sizes (may also have Tc line)
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                size_labels = [t.get_text() for t in legend.texts if 'L =' in t.get_text()]
                assert len(size_labels) == 2

        plt.close(fig)

    def test_custom_figsize(self, sample_sweep_data):
        """Test custom figure size."""
        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            figsize=(12, 10)
        )

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 10
        plt.close(fig)

    def test_save_figure(self, sample_sweep_data, output_dir):
        """Test saving phase diagram."""
        save_path = output_dir / "phase_diagram.png"

        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            save=str(save_path)
        )
        plt.close(fig)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_invalid_observable_ignored(self, sample_sweep_data):
        """Test invalid observables are ignored."""
        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            observables=['heat_capacity', 'nonexistent']
        )

        # Should only plot available observable
        assert len(axes) == 1
        plt.close(fig)

    def test_no_valid_observables_raises(self, sample_sweep_data):
        """Test error when no valid observables."""
        with pytest.raises(ValueError, match="No valid observables"):
            plot_phase_diagram(
                sample_sweep_data,
                observables=['fake1', 'fake2']
            )


# ============================================================================
# Tests for plot_binder_cumulant
# ============================================================================

class TestPlotBinderCumulant:
    """Tests for plot_binder_cumulant function."""

    def test_basic_binder_plot(self, sample_sweep_data):
        """Test basic Binder cumulant plot."""
        ax = plot_binder_cumulant(sample_sweep_data)

        assert ax is not None
        assert len(ax.lines) > 0
        plt.close()

    def test_size_filtering(self, sample_sweep_data):
        """Test size filtering."""
        ax = plot_binder_cumulant(
            sample_sweep_data,
            sizes=[8, 32]
        )

        # Should have 2 data curves + reference lines
        assert ax is not None
        plt.close()

    def test_reference_lines(self, sample_sweep_data):
        """Test reference lines are shown."""
        ax = plot_binder_cumulant(
            sample_sweep_data,
            show_reference_lines=True
        )

        # Check for horizontal lines
        assert ax is not None
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('2/3' in t for t in legend_texts)
        plt.close()

    def test_no_reference_lines(self, sample_sweep_data):
        """Test without reference lines."""
        ax = plot_binder_cumulant(
            sample_sweep_data,
            show_reference_lines=False
        )

        assert ax is not None
        plt.close()

    def test_crossing_detection(self, sample_sweep_data):
        """Test crossing point is shown."""
        ax = plot_binder_cumulant(
            sample_sweep_data,
            show_crossing=True
        )

        # Should show crossing point
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('T_c' in t for t in legend_texts)
        plt.close()

    def test_no_crossing(self, sample_sweep_data):
        """Test without crossing point."""
        ax = plot_binder_cumulant(
            sample_sweep_data,
            show_crossing=False
        )

        assert ax is not None
        plt.close()

    def test_custom_axes(self, sample_sweep_data):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_binder_cumulant(
            sample_sweep_data,
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_save_figure(self, sample_sweep_data, output_dir):
        """Test saving Binder plot."""
        save_path = output_dir / "binder.png"

        ax = plot_binder_cumulant(
            sample_sweep_data,
            save=str(save_path)
        )
        plt.close()

        assert save_path.exists()

    def test_correct_axis_labels(self, sample_sweep_data):
        """Test axis labels are correct."""
        ax = plot_binder_cumulant(sample_sweep_data)

        assert 'Temperature' in ax.get_xlabel()
        assert 'Binder' in ax.get_ylabel() or 'U' in ax.get_ylabel()
        plt.close()


# ============================================================================
# Tests for plot_scaling_collapse
# ============================================================================

class TestPlotScalingCollapse:
    """Tests for plot_scaling_collapse function."""

    def test_basic_collapse(self, sample_sweep_data):
        """Test basic scaling collapse."""
        ax = plot_scaling_collapse(
            sample_sweep_data,
            observable='abs_magnetization_mean',
            Tc=2.269,
            nu=1.0,
            exponent=0.125
        )

        assert ax is not None
        assert len(ax.lines) > 0
        plt.close()

    def test_size_filtering(self, sample_sweep_data):
        """Test size filtering in collapse."""
        ax = plot_scaling_collapse(
            sample_sweep_data,
            observable='abs_magnetization_mean',
            Tc=2.269,
            nu=1.0,
            exponent=0.125,
            sizes=[8, 16]
        )

        legend = ax.get_legend()
        assert len(legend.texts) == 2
        plt.close()

    def test_custom_axes(self, sample_sweep_data):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_scaling_collapse(
            sample_sweep_data,
            observable='abs_magnetization_mean',
            Tc=2.269,
            nu=1.0,
            exponent=0.125,
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_requires_size_column(self, sample_single_size_data):
        """Test error when no size column."""
        with pytest.raises(ValueError, match="size"):
            plot_scaling_collapse(
                sample_single_size_data,
                observable='magnetization_mean',
                Tc=2.269,
                nu=1.0,
                exponent=0.125
            )

    def test_save_figure(self, sample_sweep_data, output_dir):
        """Test saving collapse plot."""
        save_path = output_dir / "collapse.png"

        plot_scaling_collapse(
            sample_sweep_data,
            observable='abs_magnetization_mean',
            Tc=2.269,
            nu=1.0,
            exponent=0.125,
            save=str(save_path)
        )
        plt.close()

        assert save_path.exists()

    def test_correct_scaling_variables(self, sample_sweep_data):
        """Test axis labels show scaling variables."""
        ax = plot_scaling_collapse(
            sample_sweep_data,
            observable='abs_magnetization_mean',
            Tc=2.269,
            nu=1.0,
            exponent=0.125
        )

        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        assert 'T' in xlabel or 'L' in xlabel
        assert 'L' in ylabel
        plt.close()


# ============================================================================
# Tests for plot_configuration
# ============================================================================

class TestPlotConfiguration:
    """Tests for plot_configuration function."""

    def test_basic_config_plot(self, sample_spin_config):
        """Test basic spin configuration plot."""
        ax = plot_configuration(sample_spin_config)

        assert ax is not None
        plt.close()

    def test_returns_axes(self, sample_spin_config):
        """Test returns axes object."""
        ax = plot_configuration(sample_spin_config)

        assert hasattr(ax, 'imshow')
        plt.close()

    def test_custom_axes(self, sample_spin_config):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_configuration(
            sample_spin_config,
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_colorbar(self, sample_spin_config):
        """Test colorbar is shown."""
        ax = plot_configuration(
            sample_spin_config,
            show_colorbar=True
        )

        assert ax is not None
        plt.close()

    def test_no_colorbar(self, sample_spin_config):
        """Test without colorbar."""
        ax = plot_configuration(
            sample_spin_config,
            show_colorbar=False
        )

        assert ax is not None
        plt.close()

    def test_custom_colormap(self, sample_spin_config):
        """Test custom colormap."""
        ax = plot_configuration(
            sample_spin_config,
            cmap='coolwarm'
        )

        assert ax is not None
        plt.close()

    def test_with_title(self, sample_spin_config):
        """Test adding title."""
        title = "T = 2.0"
        ax = plot_configuration(
            sample_spin_config,
            title=title
        )

        assert ax.get_title() == title
        plt.close()

    def test_save_figure(self, sample_spin_config, output_dir):
        """Test saving configuration plot."""
        save_path = output_dir / "config.png"

        plot_configuration(
            sample_spin_config,
            save=str(save_path)
        )
        plt.close()

        assert save_path.exists()

    def test_aspect_ratio(self, sample_spin_config):
        """Test aspect ratio is equal."""
        ax = plot_configuration(sample_spin_config)

        assert ax.get_aspect() == 'equal' or ax.get_aspect() == 1.0
        plt.close()


# ============================================================================
# Tests for plot_spin_configuration
# ============================================================================

class TestPlotSpinConfiguration:
    """Tests for plot_spin_configuration function."""

    def test_basic_plot(self, sample_spin_config):
        """Test basic spin configuration plot."""
        ax = plot_spin_configuration(sample_spin_config)

        assert ax is not None
        plt.close()

    def test_returns_axes(self, sample_spin_config):
        """Test returns axes object."""
        ax = plot_spin_configuration(sample_spin_config)

        assert hasattr(ax, 'imshow')
        plt.close()

    def test_custom_axes(self, sample_spin_config):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_spin_configuration(sample_spin_config, ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_colorbar(self, sample_spin_config):
        """Test colorbar is shown when requested."""
        ax = plot_spin_configuration(sample_spin_config, show_colorbar=True)

        assert ax is not None
        plt.close()

    def test_no_colorbar_default(self, sample_spin_config):
        """Test colorbar is off by default."""
        ax = plot_spin_configuration(sample_spin_config)

        # Should still work without colorbar
        assert ax is not None
        plt.close()

    def test_custom_colormap(self, sample_spin_config):
        """Test custom colormap."""
        ax = plot_spin_configuration(sample_spin_config, cmap='coolwarm')

        assert ax is not None
        plt.close()

    def test_with_title(self, sample_spin_config):
        """Test adding title."""
        title = "T = 2.27"
        ax = plot_spin_configuration(sample_spin_config, title=title)

        assert ax.get_title() == title
        plt.close()

    def test_save_figure(self, sample_spin_config, output_dir):
        """Test saving configuration plot."""
        save_path = output_dir / "spin_config.png"

        plot_spin_configuration(sample_spin_config, save=str(save_path))
        plt.close()

        assert save_path.exists()

    def test_aspect_ratio(self, sample_spin_config):
        """Test aspect ratio is equal."""
        ax = plot_spin_configuration(sample_spin_config)

        assert ax.get_aspect() == 'equal' or ax.get_aspect() == 1.0
        plt.close()

    def test_raises_for_wrong_dimensions(self):
        """Test error for non-2D array."""
        spins_1d = np.array([1, -1, 1, -1])

        with pytest.raises(ValueError, match="2D"):
            plot_spin_configuration(spins_1d)

    def test_axis_labels(self, sample_spin_config):
        """Test axis labels are set correctly."""
        ax = plot_spin_configuration(sample_spin_config)

        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        plt.close()


# ============================================================================
# Tests for plot_configuration_comparison
# ============================================================================

class TestPlotConfigurationComparison:
    """Tests for plot_configuration_comparison function."""

    def test_basic_comparison(self, multiple_spin_configs):
        """Test basic comparison plot."""
        fig, axes = plot_configuration_comparison(multiple_spin_configs)

        assert fig is not None
        assert len(axes) == 3
        plt.close(fig)

    def test_with_titles(self, multiple_spin_configs):
        """Test with custom titles."""
        titles = ['T < Tc', 'T = Tc', 'T > Tc']
        fig, axes = plot_configuration_comparison(
            multiple_spin_configs,
            titles=titles
        )

        for ax, title in zip(axes, titles):
            assert ax.get_title() == title

        plt.close(fig)

    def test_custom_figsize(self, multiple_spin_configs):
        """Test custom figure size."""
        fig, axes = plot_configuration_comparison(
            multiple_spin_configs,
            figsize=(15, 5)
        )

        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 5
        plt.close(fig)

    def test_colorbar(self, multiple_spin_configs):
        """Test colorbar is shown."""
        fig, axes = plot_configuration_comparison(
            multiple_spin_configs,
            show_colorbar=True
        )

        assert fig is not None
        plt.close(fig)

    def test_no_colorbar(self, multiple_spin_configs):
        """Test without colorbar."""
        fig, axes = plot_configuration_comparison(
            multiple_spin_configs,
            show_colorbar=False
        )

        assert fig is not None
        plt.close(fig)

    def test_custom_colormap(self, multiple_spin_configs):
        """Test custom colormap."""
        fig, axes = plot_configuration_comparison(
            multiple_spin_configs,
            cmap='coolwarm'
        )

        assert fig is not None
        plt.close(fig)

    def test_save_figure(self, multiple_spin_configs, output_dir):
        """Test saving comparison plot."""
        save_path = output_dir / "comparison.png"

        fig, axes = plot_configuration_comparison(
            multiple_spin_configs,
            save=str(save_path)
        )
        plt.close(fig)

        assert save_path.exists()

    def test_single_config(self, sample_spin_config):
        """Test with single configuration."""
        fig, axes = plot_configuration_comparison([sample_spin_config])

        assert len(axes) == 1
        plt.close(fig)

    def test_empty_list_raises(self):
        """Test error for empty list."""
        with pytest.raises(ValueError, match="At least one"):
            plot_configuration_comparison([])

    def test_mismatched_titles_raises(self, multiple_spin_configs):
        """Test error when title count doesn't match."""
        with pytest.raises(ValueError, match="titles"):
            plot_configuration_comparison(
                multiple_spin_configs,
                titles=['Only', 'Two']
            )

    def test_wrong_dimensions_raises(self, sample_spin_config_3d):
        """Test error for non-2D configurations."""
        with pytest.raises(ValueError, match="2D"):
            plot_configuration_comparison([sample_spin_config_3d])


# ============================================================================
# Tests for plot_spin_configuration_3d
# ============================================================================

class TestPlotSpinConfiguration3d:
    """Tests for plot_spin_configuration_3d function."""

    def test_basic_plot(self, sample_spin_config_3d):
        """Test basic 3D slice plot."""
        ax = plot_spin_configuration_3d(sample_spin_config_3d)

        assert ax is not None
        plt.close()

    def test_default_slice_z(self, sample_spin_config_3d):
        """Test default slice is in z-direction."""
        ax = plot_spin_configuration_3d(sample_spin_config_3d)

        # Default axis should be z (axis 2)
        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'y'
        plt.close()

    def test_slice_axis_x(self, sample_spin_config_3d):
        """Test slicing along x-axis."""
        ax = plot_spin_configuration_3d(
            sample_spin_config_3d,
            slice_axis=0
        )

        assert ax.get_xlabel() == 'y'
        assert ax.get_ylabel() == 'z'
        plt.close()

    def test_slice_axis_y(self, sample_spin_config_3d):
        """Test slicing along y-axis."""
        ax = plot_spin_configuration_3d(
            sample_spin_config_3d,
            slice_axis=1
        )

        assert ax.get_xlabel() == 'x'
        assert ax.get_ylabel() == 'z'
        plt.close()

    def test_custom_slice_index(self, sample_spin_config_3d):
        """Test custom slice index."""
        ax = plot_spin_configuration_3d(
            sample_spin_config_3d,
            slice_index=5
        )

        assert 'z = 5' in ax.get_title()
        plt.close()

    def test_default_slice_is_middle(self, sample_spin_config_3d):
        """Test default slice is in middle."""
        L = sample_spin_config_3d.shape[2]
        ax = plot_spin_configuration_3d(sample_spin_config_3d)

        assert f'z = {L // 2}' in ax.get_title()
        plt.close()

    def test_custom_axes(self, sample_spin_config_3d):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_spin_configuration_3d(
            sample_spin_config_3d,
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_custom_title(self, sample_spin_config_3d):
        """Test custom title overrides default."""
        custom_title = "Custom Title"
        ax = plot_spin_configuration_3d(
            sample_spin_config_3d,
            title=custom_title
        )

        assert ax.get_title() == custom_title
        plt.close()

    def test_colorbar(self, sample_spin_config_3d):
        """Test colorbar is shown when requested."""
        ax = plot_spin_configuration_3d(
            sample_spin_config_3d,
            show_colorbar=True
        )

        assert ax is not None
        plt.close()

    def test_save_figure(self, sample_spin_config_3d, output_dir):
        """Test saving 3D slice plot."""
        save_path = output_dir / "spin_3d_slice.png"

        plot_spin_configuration_3d(
            sample_spin_config_3d,
            save=str(save_path)
        )
        plt.close()

        assert save_path.exists()

    def test_invalid_slice_axis_raises(self, sample_spin_config_3d):
        """Test error for invalid slice axis."""
        with pytest.raises(ValueError, match="slice_axis"):
            plot_spin_configuration_3d(
                sample_spin_config_3d,
                slice_axis=3
            )

    def test_invalid_slice_index_raises(self, sample_spin_config_3d):
        """Test error for out-of-bounds slice index."""
        L = sample_spin_config_3d.shape[2]

        with pytest.raises(ValueError, match="out of bounds"):
            plot_spin_configuration_3d(
                sample_spin_config_3d,
                slice_index=L + 1
            )

    def test_wrong_dimensions_raises(self, sample_spin_config):
        """Test error for non-3D array."""
        with pytest.raises(ValueError, match="3D"):
            plot_spin_configuration_3d(sample_spin_config)


# ============================================================================
# Tests for plot_spin_configuration_3d_slices
# ============================================================================

class TestPlotSpinConfiguration3dSlices:
    """Tests for plot_spin_configuration_3d_slices function."""

    def test_basic_slices(self, sample_spin_config_3d):
        """Test basic multi-slice plot."""
        fig, axes = plot_spin_configuration_3d_slices(sample_spin_config_3d)

        assert fig is not None
        assert len(axes) == 4  # Default n_slices
        plt.close(fig)

    def test_custom_n_slices(self, sample_spin_config_3d):
        """Test custom number of slices."""
        fig, axes = plot_spin_configuration_3d_slices(
            sample_spin_config_3d,
            n_slices=6
        )

        assert len(axes) == 6
        plt.close(fig)

    def test_slice_axis(self, sample_spin_config_3d):
        """Test different slice axes."""
        for axis in [0, 1, 2]:
            fig, axes = plot_spin_configuration_3d_slices(
                sample_spin_config_3d,
                slice_axis=axis
            )

            assert fig is not None
            plt.close(fig)

    def test_custom_figsize(self, sample_spin_config_3d):
        """Test custom figure size."""
        fig, axes = plot_spin_configuration_3d_slices(
            sample_spin_config_3d,
            figsize=(20, 5)
        )

        assert fig.get_figwidth() == 20
        assert fig.get_figheight() == 5
        plt.close(fig)

    def test_save_figure(self, sample_spin_config_3d, output_dir):
        """Test saving multi-slice plot."""
        save_path = output_dir / "spin_3d_slices.png"

        fig, axes = plot_spin_configuration_3d_slices(
            sample_spin_config_3d,
            save=str(save_path)
        )
        plt.close(fig)

        assert save_path.exists()

    def test_single_slice(self, sample_spin_config_3d):
        """Test with single slice."""
        fig, axes = plot_spin_configuration_3d_slices(
            sample_spin_config_3d,
            n_slices=1
        )

        assert len(axes) == 1
        plt.close(fig)

    def test_more_slices_than_size(self, sample_spin_config_3d):
        """Test requesting more slices than array size."""
        L = sample_spin_config_3d.shape[2]
        fig, axes = plot_spin_configuration_3d_slices(
            sample_spin_config_3d,
            n_slices=L + 10
        )

        # Should cap at L slices
        assert len(axes) == L
        plt.close(fig)

    def test_wrong_dimensions_raises(self, sample_spin_config):
        """Test error for non-3D array."""
        with pytest.raises(ValueError, match="3D"):
            plot_spin_configuration_3d_slices(sample_spin_config)

    def test_colorbar_present(self, sample_spin_config_3d):
        """Test colorbar is added to figure."""
        fig, axes = plot_spin_configuration_3d_slices(sample_spin_config_3d)

        # Figure should have a colorbar
        assert len(fig.axes) > len(axes)  # Extra axes for colorbar
        plt.close(fig)


# ============================================================================
# Tests for plot_energy_histogram
# ============================================================================

class TestPlotEnergyHistogram:
    """Tests for plot_energy_histogram function."""

    def test_basic_histogram(self, sample_energy_series):
        """Test basic histogram creation."""
        ax = plot_energy_histogram(sample_energy_series)

        assert ax is not None
        plt.close()

    def test_returns_axes(self, sample_energy_series):
        """Test returns axes object."""
        ax = plot_energy_histogram(sample_energy_series)

        assert hasattr(ax, 'hist')
        plt.close()

    def test_custom_axes(self, sample_energy_series):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_energy_histogram(
            sample_energy_series,
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_custom_bins(self, sample_energy_series):
        """Test custom bin count."""
        ax = plot_energy_histogram(
            sample_energy_series,
            bins=25
        )

        assert ax is not None
        plt.close()

    def test_density_normalization(self, sample_energy_series):
        """Test density normalization."""
        ax = plot_energy_histogram(
            sample_energy_series,
            density=True
        )

        ylabel = ax.get_ylabel()
        assert 'density' in ylabel.lower() or 'probability' in ylabel.lower()
        plt.close()

    def test_count_normalization(self, sample_energy_series):
        """Test count (non-density) mode."""
        ax = plot_energy_histogram(
            sample_energy_series,
            density=False
        )

        ylabel = ax.get_ylabel()
        assert 'count' in ylabel.lower() or 'Count' in ylabel
        plt.close()

    def test_show_mean_line(self, sample_energy_series):
        """Test mean line is shown."""
        ax = plot_energy_histogram(
            sample_energy_series,
            show_mean=True
        )

        # Check legend for mean
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('Mean' in t for t in legend_texts)
        plt.close()

    def test_no_mean_line(self, sample_energy_series):
        """Test without mean line."""
        ax = plot_energy_histogram(
            sample_energy_series,
            show_mean=False
        )

        assert ax is not None
        plt.close()

    def test_with_title(self, sample_energy_series):
        """Test adding title."""
        title = "Energy Distribution"
        ax = plot_energy_histogram(
            sample_energy_series,
            title=title
        )

        assert ax.get_title() == title
        plt.close()

    def test_save_figure(self, sample_energy_series, output_dir):
        """Test saving histogram."""
        save_path = output_dir / "histogram.png"

        plot_energy_histogram(
            sample_energy_series,
            save=str(save_path)
        )
        plt.close()

        assert save_path.exists()


# ============================================================================
# Tests for plot_time_series
# ============================================================================

class TestPlotTimeSeries:
    """Tests for plot_time_series function."""

    def test_basic_time_series(self, sample_energy_series):
        """Test basic time series plot."""
        ax = plot_time_series(sample_energy_series)

        assert ax is not None
        assert len(ax.lines) >= 1
        plt.close()

    def test_returns_axes(self, sample_energy_series):
        """Test returns axes object."""
        ax = plot_time_series(sample_energy_series)

        assert hasattr(ax, 'plot')
        plt.close()

    def test_custom_axes(self, sample_energy_series):
        """Test with custom axes."""
        fig, ax = plt.subplots()

        returned_ax = plot_time_series(
            sample_energy_series,
            ax=ax
        )

        assert returned_ax is ax
        plt.close(fig)

    def test_custom_name(self, sample_energy_series):
        """Test custom observable name."""
        name = "Magnetization"
        ax = plot_time_series(
            sample_energy_series,
            observable_name=name
        )

        assert name in ax.get_ylabel()
        plt.close()

    def test_show_mean(self, sample_energy_series):
        """Test mean line is shown."""
        ax = plot_time_series(
            sample_energy_series,
            show_mean=True
        )

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('Mean' in t for t in legend_texts)
        plt.close()

    def test_show_std(self, sample_energy_series):
        """Test standard deviation band."""
        ax = plot_time_series(
            sample_energy_series,
            show_std=True
        )

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('σ' in t or 'std' in t.lower() for t in legend_texts)
        plt.close()

    def test_no_mean_std(self, sample_energy_series):
        """Test without mean and std."""
        ax = plot_time_series(
            sample_energy_series,
            show_mean=False,
            show_std=False
        )

        assert ax is not None
        # Should have no legend when mean/std disabled
        plt.close()

    def test_correct_x_label(self, sample_energy_series):
        """Test x-axis label is Monte Carlo step."""
        ax = plot_time_series(sample_energy_series)

        xlabel = ax.get_xlabel()
        assert 'step' in xlabel.lower() or 'monte carlo' in xlabel.lower()
        plt.close()

    def test_save_figure(self, sample_energy_series, output_dir):
        """Test saving time series plot."""
        save_path = output_dir / "timeseries.png"

        plot_time_series(
            sample_energy_series,
            save=str(save_path)
        )
        plt.close()

        assert save_path.exists()


# ============================================================================
# Visual Regression Tests
# ============================================================================

class TestVisualRegression:
    """Visual regression tests - compare against reference images."""

    @pytest.fixture(autouse=True)
    def setup_mpl(self):
        """Configure matplotlib for reproducible output."""
        plt.rcdefaults()
        np.random.seed(42)

    def _image_similarity(self, img1_path, img2_path, threshold=0.99):
        """Compare two images for similarity."""
        from PIL import Image

        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))

        if img1.shape != img2.shape:
            return 0.0

        # Normalized correlation
        img1_norm = img1.astype(float) / 255
        img2_norm = img2.astype(float) / 255

        correlation = np.corrcoef(img1_norm.flatten(), img2_norm.flatten())[0, 1]
        return correlation

    def test_phase_diagram_visual(self, sample_sweep_data, reference_dir, output_dir):
        """Visual test for phase diagram."""
        pytest.importorskip("PIL")

        ref_path = reference_dir / "phase_diagram_ref.png"
        test_path = output_dir / "phase_diagram_test.png"

        # Generate test image
        fig, axes = plot_phase_diagram(
            sample_sweep_data,
            observables=['heat_capacity', 'susceptibility'],
            sizes=[8, 16, 32],
            save=str(test_path)
        )
        plt.close(fig)

        # If reference doesn't exist, create it
        if not ref_path.exists():
            import shutil
            shutil.copy(test_path, ref_path)
            pytest.skip("Reference image created, re-run to compare")

        # Compare
        similarity = self._image_similarity(ref_path, test_path)
        assert similarity > 0.95, f"Image similarity {similarity:.3f} < 0.95"

    def test_binder_cumulant_visual(self, sample_sweep_data, reference_dir, output_dir):
        """Visual test for Binder cumulant plot."""
        pytest.importorskip("PIL")

        ref_path = reference_dir / "binder_ref.png"
        test_path = output_dir / "binder_test.png"

        ax = plot_binder_cumulant(
            sample_sweep_data,
            save=str(test_path)
        )
        plt.close()

        if not ref_path.exists():
            import shutil
            shutil.copy(test_path, ref_path)
            pytest.skip("Reference image created, re-run to compare")

        similarity = self._image_similarity(ref_path, test_path)
        assert similarity > 0.95, f"Image similarity {similarity:.3f} < 0.95"

    def test_configuration_visual(self, sample_spin_config, reference_dir, output_dir):
        """Visual test for spin configuration plot."""
        pytest.importorskip("PIL")

        ref_path = reference_dir / "config_ref.png"
        test_path = output_dir / "config_test.png"

        plot_configuration(
            sample_spin_config,
            save=str(test_path)
        )
        plt.close()

        if not ref_path.exists():
            import shutil
            shutil.copy(test_path, ref_path)
            pytest.skip("Reference image created, re-run to compare")

        similarity = self._image_similarity(ref_path, test_path)
        assert similarity > 0.95, f"Image similarity {similarity:.3f} < 0.95"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPlottingIntegration:
    """Integration tests for plotting functions."""

    def test_combined_figure(self, sample_sweep_data, sample_spin_config, output_dir):
        """Test creating a combined multi-panel figure."""
        fig = plt.figure(figsize=(12, 10))

        # Phase diagram panel
        ax1 = fig.add_subplot(2, 2, 1)
        plot_observable_vs_temperature(
            sample_sweep_data,
            'heat_capacity',
            ax=ax1
        )

        # Binder panel
        ax2 = fig.add_subplot(2, 2, 2)
        plot_binder_cumulant(
            sample_sweep_data,
            ax=ax2,
            show_reference_lines=False
        )

        # Configuration panel
        ax3 = fig.add_subplot(2, 2, 3)
        plot_configuration(
            sample_spin_config,
            ax=ax3,
            show_colorbar=False
        )

        # Histogram panel
        energy = np.random.normal(-1.7, 0.1, 1000)
        ax4 = fig.add_subplot(2, 2, 4)
        plot_energy_histogram(energy, ax=ax4)

        plt.tight_layout()

        save_path = output_dir / "combined.png"
        fig.savefig(save_path)
        plt.close(fig)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_all_observables(self, sample_sweep_data):
        """Test plotting all available observables."""
        observables = [
            'energy_mean',
            'magnetization_mean',
            'abs_magnetization_mean',
            'heat_capacity',
            'susceptibility',
            'binder_cumulant',
        ]

        for obs in observables:
            if obs in sample_sweep_data.columns:
                ax = plot_observable_vs_temperature(
                    sample_sweep_data,
                    obs
                )
                assert ax is not None
                plt.close()

    def test_style_compatibility(self, sample_sweep_data):
        """Test plotting with different styles."""
        from ising_toolkit.visualization import style_context

        styles = ['publication', 'presentation', 'notebook', 'minimal']

        for style in styles:
            with style_context(style):
                fig, axes = plot_phase_diagram(
                    sample_sweep_data,
                    observables=['heat_capacity']
                )
                assert fig is not None
                plt.close(fig)


# ============================================================================
# Tests for plot_autocorrelation
# ============================================================================

class TestPlotAutocorrelation:
    """Tests for plot_autocorrelation function."""

    @pytest.fixture
    def correlated_data(self):
        """Create correlated time series data."""
        np.random.seed(42)
        n = 2000
        # AR(1) process with correlation
        rho = 0.9
        data = np.zeros(n)
        data[0] = np.random.normal()
        for i in range(1, n):
            data[i] = rho * data[i-1] + np.sqrt(1 - rho**2) * np.random.normal()
        return data

    def test_basic_plot(self, correlated_data):
        """Test basic autocorrelation plot."""
        ax = plot_autocorrelation(correlated_data)

        assert ax is not None
        assert len(ax.lines) >= 1
        plt.close()

    def test_custom_max_lag(self, correlated_data):
        """Test with custom max_lag."""
        ax = plot_autocorrelation(correlated_data, max_lag=50)

        assert ax is not None
        assert ax.get_xlim()[1] <= 55  # Should respect max_lag
        plt.close()

    def test_show_tau(self, correlated_data):
        """Test tau annotation is shown."""
        ax = plot_autocorrelation(correlated_data, show_tau=True)

        # Check legend contains tau
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('τ' in t or 'tau' in t.lower() for t in legend_texts)
        plt.close()

    def test_no_tau(self, correlated_data):
        """Test without tau annotation."""
        ax = plot_autocorrelation(correlated_data, show_tau=False)

        assert ax is not None
        plt.close()

    def test_confidence_interval(self, correlated_data):
        """Test confidence interval is shown."""
        ax = plot_autocorrelation(correlated_data, show_confidence=True)

        # Should have CI lines
        assert ax is not None
        plt.close()

    def test_custom_axes(self, correlated_data):
        """Test with custom axes."""
        fig, ax = plt.subplots()
        returned_ax = plot_autocorrelation(correlated_data, ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_save_figure(self, correlated_data, output_dir):
        """Test saving figure."""
        save_path = output_dir / "acf.png"
        plot_autocorrelation(correlated_data, save=str(save_path))
        plt.close()

        assert save_path.exists()

    def test_zero_variance_raises(self):
        """Test error for constant data."""
        constant_data = np.ones(100)

        with pytest.raises(ValueError, match="zero variance"):
            plot_autocorrelation(constant_data)


# ============================================================================
# Tests for plot_blocking_analysis
# ============================================================================

class TestPlotBlockingAnalysis:
    """Tests for plot_blocking_analysis function."""

    @pytest.fixture
    def correlated_data(self):
        """Create correlated time series data."""
        np.random.seed(42)
        n = 2000
        rho = 0.9
        data = np.zeros(n)
        data[0] = np.random.normal()
        for i in range(1, n):
            data[i] = rho * data[i-1] + np.sqrt(1 - rho**2) * np.random.normal()
        return data

    def test_basic_plot(self, correlated_data):
        """Test basic blocking analysis plot."""
        ax = plot_blocking_analysis(correlated_data)

        assert ax is not None
        plt.close()

    def test_show_plateau(self, correlated_data):
        """Test plateau is shown."""
        ax = plot_blocking_analysis(correlated_data, show_plateau=True)

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('Plateau' in t for t in legend_texts)
        plt.close()

    def test_no_plateau(self, correlated_data):
        """Test without plateau."""
        ax = plot_blocking_analysis(correlated_data, show_plateau=False)

        assert ax is not None
        plt.close()

    def test_custom_max_block_size(self, correlated_data):
        """Test custom max block size."""
        ax = plot_blocking_analysis(correlated_data, max_block_size=100)

        assert ax is not None
        plt.close()

    def test_custom_axes(self, correlated_data):
        """Test with custom axes."""
        fig, ax = plt.subplots()
        returned_ax = plot_blocking_analysis(correlated_data, ax=ax)

        assert returned_ax is ax
        plt.close(fig)

    def test_save_figure(self, correlated_data, output_dir):
        """Test saving figure."""
        save_path = output_dir / "blocking.png"
        plot_blocking_analysis(correlated_data, save=str(save_path))
        plt.close()

        assert save_path.exists()

    def test_log_scale(self, correlated_data):
        """Test log-log scale is used."""
        ax = plot_blocking_analysis(correlated_data)

        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        plt.close()


# ============================================================================
# Tests for plot_correlation_time_comparison
# ============================================================================

class TestPlotCorrelationTimeComparison:
    """Tests for plot_correlation_time_comparison function."""

    @pytest.fixture
    def correlated_data(self):
        """Create correlated data."""
        np.random.seed(42)
        n = 1000
        rho = 0.8
        data = np.zeros(n)
        data[0] = np.random.normal()
        for i in range(1, n):
            data[i] = rho * data[i-1] + np.sqrt(1 - rho**2) * np.random.normal()
        return data

    def test_basic_comparison(self, correlated_data):
        """Test basic comparison plot."""
        fig, axes = plot_correlation_time_comparison(correlated_data)

        assert fig is not None
        assert len(axes) == 2
        plt.close(fig)

    def test_custom_figsize(self, correlated_data):
        """Test custom figure size."""
        fig, axes = plot_correlation_time_comparison(
            correlated_data,
            figsize=(15, 6)
        )

        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 6
        plt.close(fig)

    def test_save_figure(self, correlated_data, output_dir):
        """Test saving figure."""
        save_path = output_dir / "comparison.png"
        fig, axes = plot_correlation_time_comparison(
            correlated_data,
            save=str(save_path)
        )
        plt.close(fig)

        assert save_path.exists()


# ============================================================================
# Tests for plot_equilibration_check
# ============================================================================

class TestPlotEquilibrationCheck:
    """Tests for plot_equilibration_check function."""

    @pytest.fixture
    def equilibrating_data(self):
        """Create data with initial equilibration period."""
        np.random.seed(42)
        n = 1000
        # Initial transient followed by equilibrium
        transient = np.linspace(0, -1.5, 200) + np.random.normal(0, 0.1, 200)
        equilibrium = np.random.normal(-1.5, 0.1, 800)
        return np.concatenate([transient, equilibrium])

    def test_basic_plot(self, equilibrating_data):
        """Test basic equilibration check plot."""
        ax = plot_equilibration_check(equilibrating_data)

        assert ax is not None
        plt.close()

    def test_custom_window(self, equilibrating_data):
        """Test custom window size."""
        ax = plot_equilibration_check(equilibrating_data, window_size=50)

        assert ax is not None
        plt.close()

    def test_custom_name(self, equilibrating_data):
        """Test custom observable name."""
        ax = plot_equilibration_check(
            equilibrating_data,
            observable_name='Energy'
        )

        assert 'Energy' in ax.get_ylabel()
        plt.close()

    def test_save_figure(self, equilibrating_data, output_dir):
        """Test saving figure."""
        save_path = output_dir / "equilibration.png"
        plot_equilibration_check(equilibrating_data, save=str(save_path))
        plt.close()

        assert save_path.exists()


# ============================================================================
# Tests for plot_bootstrap_distribution
# ============================================================================

class TestPlotBootstrapDistribution:
    """Tests for plot_bootstrap_distribution function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for bootstrap."""
        np.random.seed(42)
        return np.random.normal(10, 2, 500)

    def test_basic_plot(self, sample_data):
        """Test basic bootstrap distribution plot."""
        ax = plot_bootstrap_distribution(sample_data)

        assert ax is not None
        plt.close()

    def test_mean_statistic(self, sample_data):
        """Test mean statistic."""
        ax = plot_bootstrap_distribution(sample_data, statistic='mean')

        assert 'Mean' in ax.get_title()
        plt.close()

    def test_std_statistic(self, sample_data):
        """Test std statistic."""
        ax = plot_bootstrap_distribution(sample_data, statistic='std')

        assert 'Std' in ax.get_title()
        plt.close()

    def test_show_ci(self, sample_data):
        """Test confidence interval is shown."""
        ax = plot_bootstrap_distribution(sample_data, show_ci=True)

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('95%' in t or 'CI' in t for t in legend_texts)
        plt.close()

    def test_custom_n_bootstrap(self, sample_data):
        """Test custom number of bootstrap samples."""
        ax = plot_bootstrap_distribution(sample_data, n_bootstrap=500)

        assert '500' in ax.get_title()
        plt.close()

    def test_invalid_statistic_raises(self, sample_data):
        """Test error for invalid statistic."""
        with pytest.raises(ValueError, match="Unknown statistic"):
            plot_bootstrap_distribution(sample_data, statistic='invalid')

    def test_save_figure(self, sample_data, output_dir):
        """Test saving figure."""
        save_path = output_dir / "bootstrap.png"
        plot_bootstrap_distribution(sample_data, save=str(save_path))
        plt.close()

        assert save_path.exists()


# ============================================================================
# Tests for plot_finite_size_scaling
# ============================================================================

class TestPlotFiniteSizeScaling:
    """Tests for plot_finite_size_scaling function."""

    @pytest.fixture
    def scaling_data(self):
        """Create scaling data."""
        sizes = [8, 16, 32, 64]
        # chi ~ L^(gamma/nu) ~ L^1.75 for 2D Ising
        observables = {L: 0.5 * L ** 1.75 for L in sizes}
        errors = {L: 0.05 * L ** 1.75 for L in sizes}
        return sizes, observables, errors

    def test_basic_plot(self, scaling_data):
        """Test basic finite-size scaling plot."""
        sizes, observables, _ = scaling_data
        ax = plot_finite_size_scaling(sizes, observables)

        assert ax is not None
        plt.close()

    def test_with_errors(self, scaling_data):
        """Test with error bars."""
        sizes, observables, errors = scaling_data
        ax = plot_finite_size_scaling(sizes, observables, errors=errors)

        assert ax is not None
        plt.close()

    def test_with_exponent(self, scaling_data):
        """Test with theoretical exponent."""
        sizes, observables, _ = scaling_data
        ax = plot_finite_size_scaling(
            sizes, observables,
            exponent=1.75
        )

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('Theory' in t for t in legend_texts)
        plt.close()

    def test_log_scale(self, scaling_data):
        """Test log-log scale."""
        sizes, observables, _ = scaling_data
        ax = plot_finite_size_scaling(sizes, observables, log_scale=True)

        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        plt.close()

    def test_linear_scale(self, scaling_data):
        """Test linear scale."""
        sizes, observables, _ = scaling_data
        ax = plot_finite_size_scaling(sizes, observables, log_scale=False)

        assert ax.get_xscale() == 'linear'
        assert ax.get_yscale() == 'linear'
        plt.close()

    def test_show_fit(self, scaling_data):
        """Test power law fit is shown."""
        sizes, observables, _ = scaling_data
        ax = plot_finite_size_scaling(sizes, observables, show_fit=True)

        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.texts]
        assert any('Fit' in t for t in legend_texts)
        plt.close()

    def test_custom_observable_name(self, scaling_data):
        """Test custom observable name."""
        sizes, observables, _ = scaling_data
        ax = plot_finite_size_scaling(
            sizes, observables,
            observable_name='Susceptibility'
        )

        assert 'Susceptibility' in ax.get_ylabel()
        plt.close()

    def test_save_figure(self, scaling_data, output_dir):
        """Test saving figure."""
        sizes, observables, _ = scaling_data
        save_path = output_dir / "fss.png"
        plot_finite_size_scaling(sizes, observables, save=str(save_path))
        plt.close()

        assert save_path.exists()
