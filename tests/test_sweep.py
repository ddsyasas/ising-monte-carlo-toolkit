"""Tests for temperature sweep functionality."""

import numpy as np
import pytest

from ising_toolkit.models import Ising2D, Ising1D
from ising_toolkit.analysis import TemperatureSweep
from ising_toolkit.utils.constants import CRITICAL_TEMP_2D


class TestTemperatureSweepInit:
    """Tests for TemperatureSweep initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        temps = np.linspace(1.5, 3.5, 5)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=100,
        )

        assert sweep.model_class == Ising2D
        assert sweep.size == 8
        assert len(sweep.temperatures) == 5
        assert sweep.n_steps == 100
        assert sweep.equilibration == 10  # Default: n_steps // 10
        assert sweep.algorithm == 'metropolis'

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=16,
            temperatures=[2.0, 2.5, 3.0],
            n_steps=1000,
            equilibration=200,
            measurement_interval=5,
            algorithm='wolff',
            seed=42,
        )

        assert sweep.equilibration == 200
        assert sweep.measurement_interval == 5
        assert sweep.algorithm == 'wolff'
        assert sweep.seed == 42


class TestRunSingle:
    """Tests for run_single method."""

    def test_run_single_returns_dict(self):
        """Test run_single returns dictionary with expected keys."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=100,
            seed=42,
        )

        result = sweep.run_single(2.0)

        assert isinstance(result, dict)
        expected_keys = {
            'temperature',
            'energy_mean', 'energy_std', 'energy_err',
            'magnetization_mean', 'magnetization_std', 'magnetization_err',
            'abs_magnetization_mean', 'abs_magnetization_std', 'abs_magnetization_err',
            'heat_capacity', 'susceptibility', 'binder_cumulant',
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_run_single_correct_temperature(self):
        """Test run_single uses correct temperature."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=100,
        )

        result = sweep.run_single(2.5)
        assert result['temperature'] == 2.5

    def test_run_single_reasonable_values(self):
        """Test run_single returns reasonable observable values."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=[2.0],
            n_steps=500,
            seed=42,
        )

        result = sweep.run_single(2.0)

        # Energy should be negative for ferromagnetic coupling
        assert result['energy_mean'] < 0

        # Magnetization per spin should be in [-1, 1]
        assert -1 <= result['magnetization_mean'] <= 1

        # Absolute magnetization should be positive
        assert result['abs_magnetization_mean'] >= 0

        # Heat capacity should be positive
        assert result['heat_capacity'] >= 0

        # Susceptibility should be positive
        assert result['susceptibility'] >= 0

        # Binder cumulant should be in reasonable range
        assert -0.5 <= result['binder_cumulant'] <= 1.0

    def test_run_single_reproducible(self):
        """Test run_single is reproducible with seed."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=100,
            seed=42,
        )

        result1 = sweep.run_single(2.0)
        result2 = sweep.run_single(2.0)

        assert result1['energy_mean'] == result2['energy_mean']
        assert result1['magnetization_mean'] == result2['magnetization_mean']


class TestRun:
    """Tests for run method."""

    def test_run_sequential(self):
        """Test sequential run (n_workers=1)."""
        temps = [2.0, 2.5, 3.0]
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=temps,
            n_steps=50,
            seed=42,
        )

        results = sweep.run(n_workers=1, progress=False)

        # Check we got results for all temperatures
        assert len(results) == 3

    def test_run_returns_sorted_by_temperature(self):
        """Test results are sorted by temperature."""
        temps = [3.0, 2.0, 2.5]  # Unsorted
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=temps,
            n_steps=50,
            seed=42,
        )

        results = sweep.run(n_workers=1, progress=False)

        # Check temperatures are sorted
        result_temps = [r['temperature'] for r in sweep._results_list]
        assert result_temps == sorted(result_temps)

    def test_run_parallel(self):
        """Test parallel run (n_workers > 1)."""
        temps = [2.0, 2.5, 3.0]
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=temps,
            n_steps=50,
            seed=42,
        )

        results = sweep.run(n_workers=2, progress=False)

        # Check we got results for all temperatures
        assert len(results) == 3

    def test_run_stores_results(self):
        """Test run stores results in results property."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
        )

        sweep.run(n_workers=1, progress=False)

        assert sweep.results is not None
        assert sweep._results_list is not None

    def test_run_with_pandas(self):
        """Test run returns DataFrame when pandas available."""
        pytest.importorskip("pandas")

        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
        )

        df = sweep.run(n_workers=1, progress=False)

        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert 'temperature' in df.columns
        assert len(df) == 2


class TestPhaseTransitionSignature:
    """Tests for phase transition signatures in 2D Ising model."""

    @pytest.fixture
    def sweep_results(self):
        """Run a sweep around the critical temperature."""
        temps = np.linspace(1.8, 2.8, 11)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=1000,
            algorithm='metropolis',
            seed=42,
        )
        sweep.run(n_workers=1, progress=False)
        return sweep

    def test_magnetization_decreases_with_temperature(self, sweep_results):
        """Test |M| decreases as temperature increases through Tc."""
        results = sweep_results._results_list

        # Get magnetization at low and high T
        low_T_mag = results[0]['abs_magnetization_mean']  # T = 1.8
        high_T_mag = results[-1]['abs_magnetization_mean']  # T = 2.8

        # Magnetization should be higher at low T
        assert low_T_mag > high_T_mag

    def test_susceptibility_peaks_near_Tc(self, sweep_results):
        """Test susceptibility has maximum near Tc."""
        results = sweep_results._results_list
        temps = [r['temperature'] for r in results]
        chi = [r['susceptibility'] for r in results]

        # Find temperature of maximum susceptibility
        max_idx = np.argmax(chi)
        T_max_chi = temps[max_idx]

        # Should be near Tc ≈ 2.269
        assert abs(T_max_chi - CRITICAL_TEMP_2D) < 0.5

    def test_heat_capacity_peaks_near_Tc(self, sweep_results):
        """Test heat capacity has maximum near Tc."""
        results = sweep_results._results_list
        temps = [r['temperature'] for r in results]
        C = [r['heat_capacity'] for r in results]

        # Find temperature of maximum heat capacity
        max_idx = np.argmax(C)
        T_max_C = temps[max_idx]

        # Should be near Tc ≈ 2.269
        assert abs(T_max_C - CRITICAL_TEMP_2D) < 0.5

    def test_binder_cumulant_transition(self, sweep_results):
        """Test Binder cumulant transitions from ~2/3 to ~0."""
        results = sweep_results._results_list

        # At low T, U → 2/3
        U_low_T = results[0]['binder_cumulant']
        assert U_low_T > 0.4

        # At high T, U → 0
        U_high_T = results[-1]['binder_cumulant']
        assert U_high_T < 0.4


class TestFindCriticalTemperature:
    """Tests for find_critical_temperature method."""

    def test_find_Tc_susceptibility(self):
        """Test finding Tc using susceptibility peak."""
        temps = np.linspace(1.8, 2.8, 11)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=500,
            seed=42,
        )
        sweep.run(n_workers=1, progress=False)

        Tc_est = sweep.find_critical_temperature(method='susceptibility')

        # Should be reasonably close to true Tc
        assert abs(Tc_est - CRITICAL_TEMP_2D) < 0.5

    def test_find_Tc_heat_capacity(self):
        """Test finding Tc using heat capacity peak."""
        temps = np.linspace(1.8, 2.8, 11)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=500,
            seed=42,
        )
        sweep.run(n_workers=1, progress=False)

        Tc_est = sweep.find_critical_temperature(method='heat_capacity')

        # Should be reasonably close to true Tc
        assert abs(Tc_est - CRITICAL_TEMP_2D) < 0.5

    def test_find_Tc_binder(self):
        """Test finding Tc using Binder cumulant slope."""
        temps = np.linspace(1.8, 2.8, 11)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=temps,
            n_steps=500,
            seed=42,
        )
        sweep.run(n_workers=1, progress=False)

        Tc_est = sweep.find_critical_temperature(method='binder')

        # Should be reasonably close to true Tc
        assert abs(Tc_est - CRITICAL_TEMP_2D) < 0.5

    def test_find_Tc_no_results_raises(self):
        """Test finding Tc without results raises error."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=[2.0],
            n_steps=100,
        )

        with pytest.raises(RuntimeError, match="No results"):
            sweep.find_critical_temperature()

    def test_find_Tc_invalid_method_raises(self):
        """Test invalid method raises error."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
        )
        sweep.run(n_workers=1, progress=False)

        with pytest.raises(ValueError, match="Unknown method"):
            sweep.find_critical_temperature(method='invalid')


class TestPlotPhaseDiagram:
    """Tests for plot_phase_diagram method."""

    def test_plot_returns_fig_and_axes(self):
        """Test plotting returns figure and axes."""
        plt = pytest.importorskip("matplotlib.pyplot")

        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5, 3.0],
            n_steps=50,
        )
        sweep.run(n_workers=1, progress=False)

        fig, axes = sweep.plot_phase_diagram()

        assert fig is not None
        assert len(axes) > 0
        plt.close(fig)

    def test_plot_custom_observables(self):
        """Test plotting custom observables."""
        plt = pytest.importorskip("matplotlib.pyplot")

        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5, 3.0],
            n_steps=50,
        )
        sweep.run(n_workers=1, progress=False)

        fig, axes = sweep.plot_phase_diagram(
            observables=['energy_mean', 'susceptibility']
        )

        assert len(axes) == 2
        plt.close(fig)

    def test_plot_no_results_raises(self):
        """Test plotting without results raises error."""
        pytest.importorskip("matplotlib.pyplot")

        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=50,
        )

        with pytest.raises(RuntimeError, match="No results"):
            sweep.plot_phase_diagram()


class TestSaveResults:
    """Tests for save_results method."""

    def test_save_csv(self, tmp_path):
        """Test saving to CSV."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
        )
        sweep.run(n_workers=1, progress=False)

        path = tmp_path / "results.csv"
        sweep.save_results(str(path), format='csv')

        assert path.exists()

        # Check content
        with open(path) as f:
            content = f.read()
        assert 'temperature' in content
        assert '2.0' in content

    def test_save_json(self, tmp_path):
        """Test saving to JSON."""
        import json

        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
        )
        sweep.run(n_workers=1, progress=False)

        path = tmp_path / "results.json"
        sweep.save_results(str(path), format='json')

        assert path.exists()

        # Check content
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]['temperature'] == 2.0

    def test_save_no_results_raises(self, tmp_path):
        """Test saving without results raises error."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=50,
        )

        path = tmp_path / "results.csv"
        with pytest.raises(RuntimeError, match="No results"):
            sweep.save_results(str(path))

    def test_save_invalid_format_raises(self, tmp_path):
        """Test saving with invalid format raises error."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=50,
        )
        sweep.run(n_workers=1, progress=False)

        path = tmp_path / "results.xyz"
        with pytest.raises(ValueError, match="Unknown format"):
            sweep.save_results(str(path), format='xyz')


class TestWithDifferentModels:
    """Tests for TemperatureSweep with different model types."""

    def test_sweep_ising1d(self):
        """Test sweep works with 1D Ising model."""
        sweep = TemperatureSweep(
            model_class=Ising1D,
            size=20,
            temperatures=[1.0, 2.0, 3.0],
            n_steps=100,
            seed=42,
        )

        results = sweep.run(n_workers=1, progress=False)
        assert len(results) == 3

    def test_sweep_different_algorithms(self):
        """Test sweep works with different algorithms."""
        # Metropolis
        sweep_metro = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
            algorithm='metropolis',
            seed=42,
        )
        results_metro = sweep_metro.run(n_workers=1, progress=False)

        # Wolff
        sweep_wolff = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0, 2.5],
            n_steps=50,
            algorithm='wolff',
            seed=42,
        )
        results_wolff = sweep_wolff.run(n_workers=1, progress=False)

        # Both should return results
        assert len(results_metro) == 2
        assert len(results_wolff) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_temperature(self):
        """Test sweep with single temperature."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=[2.0],
            n_steps=50,
        )

        results = sweep.run(n_workers=1, progress=False)
        assert len(results) == 1

    def test_many_temperatures(self):
        """Test sweep with many temperatures."""
        temps = np.linspace(1.0, 4.0, 31)
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=4,
            temperatures=temps,
            n_steps=20,
        )

        results = sweep.run(n_workers=1, progress=False)
        assert len(results) == 31

    def test_very_low_temperature(self):
        """Test sweep at very low temperature (ordered phase)."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=[0.5],
            n_steps=200,
            seed=42,
        )

        sweep.run(n_workers=1, progress=False)
        result = sweep._results_list[0]

        # At low T, should be nearly ordered
        assert result['abs_magnetization_mean'] > 0.8

    def test_very_high_temperature(self):
        """Test sweep at very high temperature (disordered phase)."""
        sweep = TemperatureSweep(
            model_class=Ising2D,
            size=8,
            temperatures=[10.0],
            n_steps=200,
            seed=42,
        )

        sweep.run(n_workers=1, progress=False)
        result = sweep._results_list[0]

        # At high T, should be nearly disordered
        assert result['abs_magnetization_mean'] < 0.3
