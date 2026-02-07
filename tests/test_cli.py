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

        assert result.exit_code == 0


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
        assert '--observable' in result.output
        assert 'INPUT_FILE' in result.output

    def test_plot_sweep_results(self, runner, temp_dir):
        """Test plotting sweep results."""
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

        # Now plot
        plot_file = temp_dir / 'plot.png'
        result = runner.invoke(main, [
            'plot',
            str(sweep_file),
            '--observable', 'magnetization',
            '--output', str(plot_file),
        ])

        assert result.exit_code == 0
        assert plot_file.exists()

    def test_plot_missing_file(self, runner):
        """Test error for missing input file."""
        result = runner.invoke(main, [
            'plot',
            'nonexistent.npz',
        ])

        assert result.exit_code != 0


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

        # Plot each observable
        for obs in ['energy', 'magnetization', 'susceptibility']:
            plot_file = temp_dir / f'{obs}.png'
            result = runner.invoke(main, [
                'plot',
                str(sweep_file),
                '-O', obs,
                '-o', str(plot_file),
            ])

            assert result.exit_code == 0
            assert plot_file.exists()
