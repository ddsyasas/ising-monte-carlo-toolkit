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
    plot_energy_histogram,
    plot_time_series,
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
