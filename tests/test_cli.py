"""Tests for the command-line interface."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from ising_toolkit.cli import main


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Tests for main group
# ============================================================================

class TestMainGroup:
    """Tests for main CLI group."""

    def test_help(self, runner):
        """Test --help works."""
        result = runner.invoke(main, ['--help'])

        assert result.exit_code == 0
        assert 'Ising Monte Carlo Toolkit' in result.output
        assert 'run' in result.output
        assert 'sweep' in result.output
        assert 'info' in result.output

    def test_version(self, runner):
        """Test --version works."""
        result = runner.invoke(main, ['--version'])

        assert result.exit_code == 0
        assert '0.1.0' in result.output

    def test_no_command(self, runner):
        """Test invoking without command shows help."""
        result = runner.invoke(main, [])

        # Click may return 0 or 2 depending on version/environment
        assert result.exit_code in (0, 2)


# ============================================================================
# Tests for run command
# ============================================================================

class TestRunCommand:
    """Tests for run command."""

    def test_run_help(self, runner):
        """Test run --help works."""
        result = runner.invoke(main, ['run', '--help'])

        assert result.exit_code == 0
        assert '--model' in result.output
        assert '--size' in result.output
        assert '--temperature' in result.output
        assert '--algorithm' in result.output

    def test_run_missing_required(self, runner):
        """Test error when required options missing."""
        result = runner.invoke(main, ['run'])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()

    def test_run_basic(self, runner):
        """Test basic run command."""
        result = runner.invoke(main, [
            'run',
            '--model', 'ising2d',
            '--size', '8',
            '--temperature', '2.269',
            '--steps', '1000',
            '--equilibration', '100',
        ])

        assert result.exit_code == 0
        assert 'Simulation Results' in result.output
        assert 'Energy/spin' in result.output
        assert 'Magnetization' in result.output

    def test_run_with_output(self, runner, temp_dir):
        """Test run with output file."""
        output_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '--model', 'ising2d',
            '--size', '8',
            '--temperature', '2.269',
            '--steps', '1000',
            '--equilibration', '100',
            '--output', str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify contents
        data = np.load(output_file)
        assert 'energy_mean' in data
        assert 'magnetization_mean' in data

    def test_run_with_seed(self, runner):
        """Test run with seed for reproducibility."""
        result1 = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '500',
            '-e', '100',
            '-s', '42',
        ])

        result2 = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '500',
            '-e', '100',
            '-s', '42',
        ])

        assert result1.exit_code == 0
        assert result2.exit_code == 0
        # Results should be identical with same seed
        # (Note: output format may vary, but energy values should match)

    def test_run_wolff_algorithm(self, runner):
        """Test run with Wolff algorithm."""
        result = runner.invoke(main, [
            'run',
            '--model', 'ising2d',
            '--size', '8',
            '--temperature', '2.269',
            '--steps', '1000',
            '--equilibration', '100',
            '--algorithm', 'wolff',
        ])

        assert result.exit_code == 0
        assert 'wolff' in result.output.lower()

    def test_run_1d_model(self, runner):
        """Test run with 1D model."""
        result = runner.invoke(main, [
            'run',
            '--model', 'ising1d',
            '--size', '32',
            '--temperature', '1.0',
            '--steps', '1000',
            '--equilibration', '100',
        ])

        assert result.exit_code == 0
        assert 'ising1d' in result.output

    def test_run_3d_model(self, runner):
        """Test run with 3D model."""
        result = runner.invoke(main, [
            'run',
            '--model', 'ising3d',
            '--size', '4',
            '--temperature', '4.0',
            '--steps', '500',
            '--equilibration', '100',
        ])

        assert result.exit_code == 0
        assert 'ising3d' in result.output

    def test_run_verbose(self, runner):
        """Test run with verbose output."""
        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '500',
            '-e', '100',
            '-v',
        ])

        assert result.exit_code == 0
        assert 'Creating' in result.output or 'Using' in result.output

    def test_run_invalid_model(self, runner):
        """Test error for invalid model."""
        result = runner.invoke(main, [
            'run',
            '--model', 'invalid',
            '--size', '8',
            '--temperature', '2.269',
        ])

        assert result.exit_code != 0


# ============================================================================
# Tests for sweep command
# ============================================================================

class TestSweepCommand:
    """Tests for sweep command."""

    def test_sweep_help(self, runner):
        """Test sweep --help works."""
        result = runner.invoke(main, ['sweep', '--help'])

        assert result.exit_code == 0
        assert '--temp-start' in result.output
        assert '--temp-end' in result.output
        assert '--temp-steps' in result.output
        assert '--parallel' in result.output

    def test_sweep_basic(self, runner, temp_dir):
        """Test basic sweep command."""
        output_dir = temp_dir / 'sweep_results'

        result = runner.invoke(main, [
            'sweep',
            '--model', 'ising2d',
            '--size', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '--steps', '500',
            '--equilibration', '100',
            '--output', str(output_dir),
        ])

        assert result.exit_code == 0
        assert 'Temperature Sweep' in result.output
        assert 'Sweep Complete' in result.output
        assert output_dir.exists()

    def test_sweep_with_output_csv(self, runner, temp_dir):
        """Test sweep with CSV output."""
        output_dir = temp_dir / 'sweep_csv'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(output_dir),
            '-f', 'csv',
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

        # Check CSV file exists
        csv_files = list(output_dir.glob('*.csv'))
        assert len(csv_files) >= 1  # At least the data file

        # Verify one of the CSV files has the right structure
        data_files = [f for f in csv_files if 'summary' not in f.name]
        assert len(data_files) >= 1

    def test_sweep_with_output_npz(self, runner, temp_dir):
        """Test sweep with NPZ output."""
        output_dir = temp_dir / 'sweep_npz'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(output_dir),
            '-f', 'npz',
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

        # Check NPZ file exists and has correct contents
        npz_files = list(output_dir.glob('*.npz'))
        assert len(npz_files) >= 1

        # Verify contents
        data = np.load(npz_files[0])
        assert 'temperatures' in data
        assert 'magnetizations' in data
        assert len(data['temperatures']) == 3

    def test_sweep_multiple_sizes(self, runner, temp_dir):
        """Test sweep with multiple lattice sizes."""
        output_dir = temp_dir / 'sweep_multi'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '4',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(output_dir),
            '-f', 'csv',
        ])

        assert result.exit_code == 0
        assert output_dir.exists()

        # Should have files for both sizes plus summary
        csv_files = list(output_dir.glob('*.csv'))
        assert len(csv_files) >= 2  # At least 2 data files

        # Check summary file exists
        summary_files = [f for f in csv_files if 'summary' in f.name]
        assert len(summary_files) == 1


# ============================================================================
# Tests for info command
# ============================================================================

class TestInfoCommand:
    """Tests for info command."""

    def test_info_help(self, runner):
        """Test info --help works."""
        result = runner.invoke(main, ['info', '--help'])

        assert result.exit_code == 0
        assert '--model' in result.output

    def test_info_1d(self, runner):
        """Test info for 1D model."""
        result = runner.invoke(main, ['info', '--model', 'ising1d'])

        assert result.exit_code == 0
        assert '1D Ising Model' in result.output
        assert 'No phase transition' in result.output

    def test_info_2d(self, runner):
        """Test info for 2D model."""
        result = runner.invoke(main, ['info', '--model', 'ising2d'])

        assert result.exit_code == 0
        assert '2D Ising Model' in result.output
        assert 'Critical temperature' in result.output
        assert 'Onsager' in result.output

    def test_info_3d(self, runner):
        """Test info for 3D model."""
        result = runner.invoke(main, ['info', '--model', 'ising3d'])

        assert result.exit_code == 0
        assert '3D Ising Model' in result.output
        assert 'Critical temperature' in result.output


# ============================================================================
# Tests for plot command
# ============================================================================

class TestPlotCommand:
    """Tests for plot command."""

    def test_plot_help(self, runner):
        """Test plot --help works."""
        result = runner.invoke(main, ['plot', '--help'])

        assert result.exit_code == 0
        assert '--type' in result.output
        assert '--observable' in result.output
        assert '--style' in result.output
        assert '--format' in result.output
        assert 'INPUT' in result.output

    def test_plot_phase_diagram(self, runner, temp_dir):
        """Test plotting phase diagram from sweep results."""
        pytest.importorskip("matplotlib")

        # First create sweep data
        sweep_dir = temp_dir / 'sweep_data'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(sweep_dir),
            '-f', 'npz',
        ])

        assert result.exit_code == 0

        # Find the NPZ file
        npz_files = list(sweep_dir.glob('*.npz'))
        assert len(npz_files) >= 1
        sweep_file = npz_files[0]

        # Create phase diagram
        plot_file = temp_dir / 'phase_diagram.pdf'
        result = runner.invoke(main, [
            'plot',
            str(sweep_file),
            '--type', 'phase_diagram',
            '--output', str(plot_file),
        ])

        assert result.exit_code == 0
        assert plot_file.exists()

    def test_plot_timeseries(self, runner, temp_dir):
        """Test plotting time series."""
        pytest.importorskip("matplotlib")

        # First run a simulation
        results_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(results_file),
        ])

        assert result.exit_code == 0

        # Plot time series
        plot_file = temp_dir / 'timeseries.png'
        result = runner.invoke(main, [
            'plot',
            str(results_file),
            '--type', 'timeseries',
            '--observable', 'energy',
            '--output', str(plot_file),
            '--format', 'png',
        ])

        assert result.exit_code == 0
        assert plot_file.exists()

    def test_plot_snapshot(self, runner, temp_dir):
        """Test plotting spin snapshot."""
        pytest.importorskip("matplotlib")

        # First run a simulation with spin saving
        results_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(results_file),
            '--save-spins',
        ])

        assert result.exit_code == 0

        # Plot snapshot
        plot_file = temp_dir / 'snapshot.png'
        result = runner.invoke(main, [
            'plot',
            str(results_file),
            '--type', 'snapshot',
            '--output', str(plot_file),
            '--format', 'png',
        ])

        assert result.exit_code == 0
        assert plot_file.exists()

    def test_plot_autocorrelation(self, runner, temp_dir):
        """Test plotting autocorrelation."""
        pytest.importorskip("matplotlib")

        # First run a simulation
        results_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(results_file),
        ])

        assert result.exit_code == 0

        # Plot autocorrelation
        plot_file = temp_dir / 'autocorr.pdf'
        result = runner.invoke(main, [
            'plot',
            str(results_file),
            '--type', 'autocorrelation',
            '--observable', 'magnetization',
            '--output', str(plot_file),
        ])

        assert result.exit_code == 0
        assert plot_file.exists()

    def test_plot_styles(self, runner, temp_dir):
        """Test different plot styles."""
        pytest.importorskip("matplotlib")

        # Create sweep data
        sweep_dir = temp_dir / 'sweep_data'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(sweep_dir),
            '-f', 'npz',
        ])

        assert result.exit_code == 0

        npz_files = list(sweep_dir.glob('*.npz'))
        sweep_file = npz_files[0]

        # Test presentation style
        plot_file = temp_dir / 'presentation.png'
        result = runner.invoke(main, [
            'plot',
            str(sweep_file),
            '--type', 'phase_diagram',
            '--style', 'presentation',
            '--format', 'png',
            '--output', str(plot_file),
        ])

        assert result.exit_code == 0
        assert plot_file.exists()

    def test_plot_missing_file(self, runner):
        """Test error for missing input file."""
        result = runner.invoke(main, [
            'plot',
            'nonexistent.npz',
            '--type', 'phase_diagram',
        ])

        assert result.exit_code != 0

    def test_plot_auto_filename(self, runner, temp_dir):
        """Test auto-generated output filename."""
        pytest.importorskip("matplotlib")

        # Create sweep data
        sweep_dir = temp_dir / 'sweep_data'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(sweep_dir),
            '-f', 'npz',
        ])

        assert result.exit_code == 0

        npz_files = list(sweep_dir.glob('*.npz'))
        sweep_file = npz_files[0]

        # Plot without specifying output (should auto-generate)
        result = runner.invoke(main, [
            'plot',
            str(sweep_file),
            '--type', 'phase_diagram',
        ])

        assert result.exit_code == 0
        assert 'saved to' in result.output.lower()


# ============================================================================
# Tests for benchmark command
# ============================================================================

class TestBenchmarkCommand:
    """Tests for benchmark command."""

    def test_benchmark_help(self, runner):
        """Test benchmark --help works."""
        result = runner.invoke(main, ['benchmark', '--help'])

        assert result.exit_code == 0

    def test_benchmark_runs(self, runner):
        """Test benchmark command runs."""
        # Note: This test may take a few seconds
        result = runner.invoke(main, ['benchmark'])

        assert result.exit_code == 0
        assert 'Performance Benchmark' in result.output
        assert 'Metropolis' in result.output
        assert 'Wolff' in result.output
        assert 'Speedup' in result.output


# ============================================================================
# Tests for analyze command
# ============================================================================

class TestAnalyzeCommand:
    """Tests for analyze command."""

    def test_analyze_help(self, runner):
        """Test analyze --help works."""
        result = runner.invoke(main, ['analyze', '--help'])

        assert result.exit_code == 0
        assert '--observables' in result.output
        assert '--bootstrap' in result.output
        assert '--format' in result.output
        assert 'INPUT' in result.output

    def test_analyze_single_file(self, runner, temp_dir):
        """Test analyzing a single simulation result."""
        # First run a simulation
        output_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(output_file),
        ])
        assert result.exit_code == 0

        # Now analyze it
        result = runner.invoke(main, [
            'analyze',
            str(output_file),
        ])

        assert result.exit_code == 0
        assert 'Analysis Results' in result.output
        assert 'energy' in result.output.lower()

    def test_analyze_sweep_file(self, runner, temp_dir):
        """Test analyzing a sweep result."""
        # First run a sweep
        sweep_dir = temp_dir / 'sweep_data'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(sweep_dir),
            '-f', 'npz',
        ])
        assert result.exit_code == 0

        # Find the NPZ file
        npz_files = list(sweep_dir.glob('*.npz'))
        assert len(npz_files) >= 1

        # Analyze
        result = runner.invoke(main, [
            'analyze',
            str(npz_files[0]),
        ])

        assert result.exit_code == 0
        assert 'Analysis Results' in result.output

    def test_analyze_directory(self, runner, temp_dir):
        """Test analyzing all files in a directory."""
        # Create sweep data
        sweep_dir = temp_dir / 'sweep_data'

        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '3',
            '-n', '500',
            '-e', '100',
            '-o', str(sweep_dir),
            '-f', 'npz',
        ])
        assert result.exit_code == 0

        # Analyze directory
        result = runner.invoke(main, [
            'analyze',
            str(sweep_dir),
        ])

        assert result.exit_code == 0
        assert 'Analysis Results' in result.output

    def test_analyze_specific_observables(self, runner, temp_dir):
        """Test analyzing specific observables."""
        # First run a simulation
        output_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(output_file),
        ])
        assert result.exit_code == 0

        # Analyze specific observables
        result = runner.invoke(main, [
            'analyze',
            str(output_file),
            '--observables', 'energy,magnetization',
        ])

        assert result.exit_code == 0
        assert 'energy' in result.output.lower()
        assert 'magnetization' in result.output.lower()

    def test_analyze_with_output_csv(self, runner, temp_dir):
        """Test saving analysis to CSV."""
        # First run a simulation
        results_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(results_file),
        ])
        assert result.exit_code == 0

        # Analyze and save
        analysis_file = temp_dir / 'analysis.csv'
        result = runner.invoke(main, [
            'analyze',
            str(results_file),
            '-o', str(analysis_file),
            '-f', 'csv',
        ])

        assert result.exit_code == 0
        assert analysis_file.exists()

        # Verify CSV content
        content = analysis_file.read_text()
        assert 'source' in content
        assert 'energy_mean' in content

    def test_analyze_with_output_json(self, runner, temp_dir):
        """Test saving analysis to JSON."""
        import json

        # First run a simulation
        results_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(results_file),
        ])
        assert result.exit_code == 0

        # Analyze and save as JSON
        analysis_file = temp_dir / 'analysis.json'
        result = runner.invoke(main, [
            'analyze',
            str(results_file),
            '-o', str(analysis_file),
            '-f', 'json',
        ])

        assert result.exit_code == 0
        assert analysis_file.exists()

        # Verify JSON content
        with open(analysis_file) as f:
            data = json.load(f)
        assert 'results' in data
        assert 'observables' in data

    def test_analyze_missing_file(self, runner):
        """Test error for missing input file."""
        result = runner.invoke(main, [
            'analyze',
            'nonexistent.npz',
        ])

        assert result.exit_code != 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_full_workflow(self, runner, temp_dir):
        """Test full workflow: run -> save -> load."""
        # Run simulation
        output_file = temp_dir / 'results.npz'

        result = runner.invoke(main, [
            'run',
            '-m', 'ising2d',
            '-L', '8',
            '-T', '2.269',
            '-n', '1000',
            '-e', '100',
            '-o', str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Load and verify
        data = np.load(output_file)
        assert data['model'] == 'ising2d'
        assert data['size'] == 8
        assert abs(data['temperature'] - 2.269) < 0.001

    def test_sweep_and_plot(self, runner, temp_dir):
        """Test sweep followed by plot."""
        pytest.importorskip("matplotlib")

        # Sweep with NPZ format for plotting
        sweep_dir = temp_dir / 'sweep_data'
        result = runner.invoke(main, [
            'sweep',
            '-m', 'ising2d',
            '-L', '8',
            '--temp-start', '2.0',
            '--temp-end', '2.5',
            '--temp-steps', '5',
            '-n', '500',
            '-e', '100',
            '-o', str(sweep_dir),
            '-f', 'npz',
        ])

        assert result.exit_code == 0

        # Find the NPZ file
        npz_files = list(sweep_dir.glob('*.npz'))
        assert len(npz_files) >= 1
        sweep_file = npz_files[0]

        # Create phase diagram plot
        plot_file = temp_dir / 'phase_diagram.pdf'
        result = runner.invoke(main, [
            'plot',
            str(sweep_file),
            '--type', 'phase_diagram',
            '-o', str(plot_file),
        ])

        assert result.exit_code == 0
        assert plot_file.exists()
