"""Tests for finite-size scaling analysis."""

import numpy as np
import pytest

from ising_toolkit.models import Ising2D
from ising_toolkit.analysis import FiniteSizeScaling
from ising_toolkit.utils.constants import CRITICAL_TEMP_2D


class TestFiniteSizeScalingInit:
    """Tests for FiniteSizeScaling initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=np.linspace(2.0, 2.5, 5),
            n_steps=100,
        )

        assert fss.model_class == Ising2D
        assert fss.sizes == [8, 16]
        assert len(fss.temperatures) == 5
        assert fss.n_steps == 100

    def test_init_sizes_sorted(self):
        """Test sizes are sorted."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[16, 8, 32],
            temperatures=[2.0, 2.5],
            n_steps=100,
        )

        assert fss.sizes == [8, 16, 32]

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=[2.0, 2.5],
            n_steps=1000,
            equilibration=200,
            algorithm='wolff',
            seed=42,
        )

        assert fss.equilibration == 200
        assert fss.algorithm == 'wolff'
        assert fss.seed == 42


class TestRun:
    """Tests for run method."""

    def test_run_returns_results(self):
        """Test run returns results."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8],
            temperatures=[2.0, 2.5],
            n_steps=50,
            seed=42,
        )

        results = fss.run(n_workers=1, progress=False)

        # Should have 2 sizes * 2 temperatures = 4 rows
        assert len(results) == 4

    def test_run_has_size_column(self):
        """Test results include size column."""
        pytest.importorskip("pandas")

        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8],
            temperatures=[2.0, 2.5],
            n_steps=50,
        )

        df = fss.run(n_workers=1, progress=False)

        assert 'size' in df.columns
        assert set(df['size'].unique()) == {4, 8}

    def test_run_stores_by_size(self):
        """Test results are stored by size."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8],
            temperatures=[2.0, 2.5],
            n_steps=50,
        )

        fss.run(n_workers=1, progress=False)

        assert 4 in fss._results_by_size
        assert 8 in fss._results_by_size
        assert len(fss._results_by_size[4]) == 2
        assert len(fss._results_by_size[8]) == 2


class TestFindCriticalTemperature:
    """Tests for find_critical_temperature method."""

    @pytest.fixture
    def fss_results(self):
        """Run FSS analysis around Tc."""
        temps = np.linspace(2.0, 2.5, 11)
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=temps,
            n_steps=500,
            algorithm='metropolis',
            seed=42,
        )
        fss.run(n_workers=1, progress=False)
        return fss

    def test_find_Tc_binder_near_exact(self, fss_results):
        """Test Binder crossing gives Tc near exact value."""
        Tc, Tc_err = fss_results.find_critical_temperature(method='binder')

        # Should be within 0.3 of exact Tc ≈ 2.269
        assert abs(Tc - CRITICAL_TEMP_2D) < 0.3

    def test_find_Tc_susceptibility_near_exact(self, fss_results):
        """Test susceptibility peak gives Tc near exact value."""
        Tc, Tc_err = fss_results.find_critical_temperature(method='susceptibility')

        # Should be within 0.3 of exact Tc
        assert abs(Tc - CRITICAL_TEMP_2D) < 0.3

    def test_find_Tc_returns_tuple(self, fss_results):
        """Test find_Tc returns (Tc, error) tuple."""
        result = fss_results.find_critical_temperature()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_find_Tc_no_results_raises(self):
        """Test find_Tc without results raises error."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=[2.0, 2.5],
            n_steps=100,
        )

        with pytest.raises(RuntimeError, match="No results"):
            fss.find_critical_temperature()

    def test_find_Tc_invalid_method_raises(self, fss_results):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            fss_results.find_critical_temperature(method='invalid')

    def test_find_Tc_needs_multiple_sizes(self):
        """Test Binder crossing needs at least 2 sizes."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8],
            temperatures=[2.0, 2.5],
            n_steps=50,
        )
        fss.run(n_workers=1, progress=False)

        with pytest.raises(ValueError, match="at least 2 sizes"):
            fss.find_critical_temperature(method='binder')


class TestExtractExponents:
    """Tests for exponent extraction methods."""

    @pytest.fixture
    def fss_results(self):
        """Run FSS analysis with multiple sizes."""
        temps = np.linspace(2.1, 2.4, 11)
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16, 32],
            temperatures=temps,
            n_steps=1000,
            algorithm='metropolis',
            seed=42,
        )
        fss.run(n_workers=1, progress=False)
        return fss

    def test_extract_beta_returns_tuple(self, fss_results):
        """Test extract_exponent_beta returns (value, error)."""
        result = fss_results.extract_exponent_beta(Tc=CRITICAL_TEMP_2D)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extract_beta_near_exact(self, fss_results):
        """Test β/ν is near exact value 1/8 = 0.125."""
        beta_nu, error = fss_results.extract_exponent_beta(Tc=CRITICAL_TEMP_2D)

        # Should be positive (magnetization decreases with L at Tc)
        assert beta_nu > 0

        # Should be reasonably close to 0.125
        # Allow wide margin for small sizes and short runs
        assert abs(beta_nu - 0.125) < 0.15

    def test_extract_gamma_returns_tuple(self, fss_results):
        """Test extract_exponent_gamma returns (value, error)."""
        result = fss_results.extract_exponent_gamma(Tc=CRITICAL_TEMP_2D)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extract_gamma_near_exact(self, fss_results):
        """Test γ/ν is near exact value 7/4 = 1.75."""
        gamma_nu, error = fss_results.extract_exponent_gamma(Tc=CRITICAL_TEMP_2D)

        # Should be positive (susceptibility increases with L at Tc)
        assert gamma_nu > 0

        # Should be reasonably close to 1.75
        # Allow wide margin for finite-size effects
        assert abs(gamma_nu - 1.75) < 0.5

    def test_extract_nu_returns_tuple(self, fss_results):
        """Test extract_exponent_nu returns (value, error)."""
        result = fss_results.extract_exponent_nu(Tc=CRITICAL_TEMP_2D)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extract_nu_near_exact(self, fss_results):
        """Test 1/ν is near exact value 1."""
        one_over_nu, error = fss_results.extract_exponent_nu(Tc=CRITICAL_TEMP_2D)

        # Should be positive
        assert one_over_nu > 0

        # Should be reasonably close to 1
        assert abs(one_over_nu - 1.0) < 0.5


class TestPlotBinderCrossing:
    """Tests for plot_binder_crossing method."""

    def test_plot_returns_ax(self):
        """Test plotting returns axes."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8],
            temperatures=[2.0, 2.3, 2.5],
            n_steps=50,
        )
        fss.run(n_workers=1, progress=False)

        ax = fss.plot_binder_crossing()

        assert ax is not None
        plt.close('all')

    def test_plot_no_results_raises(self):
        """Test plotting without results raises error."""
        pytest.importorskip("matplotlib.pyplot")

        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=[2.0, 2.5],
            n_steps=100,
        )

        with pytest.raises(RuntimeError, match="No results"):
            fss.plot_binder_crossing()


class TestPlotScalingCollapse:
    """Tests for plot_scaling_collapse method."""

    def test_plot_collapse_returns_ax(self):
        """Test collapse plotting returns axes."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8],
            temperatures=[2.0, 2.3, 2.5],
            n_steps=50,
        )
        fss.run(n_workers=1, progress=False)

        ax = fss.plot_scaling_collapse(
            observable='abs_magnetization_mean',
            Tc=CRITICAL_TEMP_2D,
            nu=1.0,
            exponent=0.125,
        )

        assert ax is not None
        plt.close('all')


class TestPlotExponentFit:
    """Tests for plot_exponent_fit method."""

    def test_plot_magnetization_fit(self):
        """Test magnetization fit plotting."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8, 16],
            temperatures=[2.0, 2.3, 2.5],
            n_steps=50,
        )
        fss.run(n_workers=1, progress=False)

        ax = fss.plot_exponent_fit(
            observable='magnetization',
            Tc=CRITICAL_TEMP_2D,
        )

        assert ax is not None
        plt.close('all')

    def test_plot_susceptibility_fit(self):
        """Test susceptibility fit plotting."""
        plt = pytest.importorskip("matplotlib.pyplot")

        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[4, 8, 16],
            temperatures=[2.0, 2.3, 2.5],
            n_steps=50,
        )
        fss.run(n_workers=1, progress=False)

        ax = fss.plot_exponent_fit(
            observable='susceptibility',
            Tc=CRITICAL_TEMP_2D,
        )

        assert ax is not None
        plt.close('all')


class TestSummary:
    """Tests for summary method."""

    def test_summary_returns_dict(self):
        """Test summary returns dictionary."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=np.linspace(2.0, 2.5, 7),
            n_steps=200,
            seed=42,
        )
        fss.run(n_workers=1, progress=False)

        summary = fss.summary()

        assert isinstance(summary, dict)
        assert 'Tc' in summary
        assert 'Tc_error' in summary

    def test_summary_with_Tc(self):
        """Test summary with provided Tc."""
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16],
            temperatures=np.linspace(2.0, 2.5, 7),
            n_steps=200,
        )
        fss.run(n_workers=1, progress=False)

        summary = fss.summary(Tc=CRITICAL_TEMP_2D)

        assert summary['Tc'] == CRITICAL_TEMP_2D
        assert summary['Tc_error'] == 0.0


class TestPhysicalConsistency:
    """Tests for physical consistency of FSS results."""

    @pytest.fixture
    def fss_detailed(self):
        """Run detailed FSS analysis."""
        temps = np.linspace(2.0, 2.5, 15)
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[8, 16, 32],
            temperatures=temps,
            n_steps=2000,
            algorithm='metropolis',
            seed=42,
        )
        fss.run(n_workers=1, progress=False)
        return fss

    def test_susceptibility_increases_with_size_at_Tc(self, fss_detailed):
        """Test χ increases with L at Tc."""
        chi_values = []
        for size in fss_detailed.sizes:
            chi = fss_detailed._get_observable_at_temp(
                size, CRITICAL_TEMP_2D, 'susceptibility'
            )
            chi_values.append(chi)

        # Susceptibility should increase with size at Tc
        assert chi_values[0] < chi_values[1] < chi_values[2]

    def test_magnetization_decreases_with_size_at_Tc(self, fss_detailed):
        """Test |M| decreases with L at Tc."""
        mag_values = []
        for size in fss_detailed.sizes:
            mag = fss_detailed._get_observable_at_temp(
                size, CRITICAL_TEMP_2D, 'abs_magnetization_mean'
            )
            mag_values.append(mag)

        # Magnetization should decrease with size at Tc
        assert mag_values[0] > mag_values[1] > mag_values[2]

    def test_binder_cumulant_crossing(self, fss_detailed):
        """Test Binder cumulant curves cross near Tc."""
        Tc_est, _ = fss_detailed.find_critical_temperature(method='binder')

        # Binder values at estimated Tc should be similar for different sizes
        U_values = []
        for size in fss_detailed.sizes:
            U = fss_detailed._get_observable_at_temp(
                size, Tc_est, 'binder_cumulant'
            )
            U_values.append(U)

        # At crossing, values should be similar (within 0.1)
        U_spread = max(U_values) - min(U_values)
        assert U_spread < 0.15

    def test_hyperscaling_relation(self, fss_detailed):
        """Test hyperscaling: 2β + γ = dν (d=2 for 2D)."""
        # Get exponents
        beta_nu, _ = fss_detailed.extract_exponent_beta(CRITICAL_TEMP_2D)
        gamma_nu, _ = fss_detailed.extract_exponent_gamma(CRITICAL_TEMP_2D)

        # Hyperscaling: 2β/ν + γ/ν = d = 2
        hyperscaling = 2 * beta_nu + gamma_nu

        # Should be close to 2 (allow for finite-size effects)
        assert abs(hyperscaling - 2.0) < 0.5


class TestExactComparison:
    """Tests comparing results to exact 2D Ising values."""

    @pytest.fixture
    def fss_accurate(self):
        """Run accurate FSS analysis with Wolff algorithm."""
        temps = np.linspace(2.1, 2.45, 15)
        fss = FiniteSizeScaling(
            model_class=Ising2D,
            sizes=[16, 32],
            temperatures=temps,
            n_steps=5000,
            algorithm='wolff',
            seed=42,
        )
        fss.run(n_workers=1, progress=False)
        return fss

    def test_Tc_within_5_percent(self, fss_accurate):
        """Test extracted Tc is within 5% of exact value."""
        Tc, _ = fss_accurate.find_critical_temperature(method='binder')

        relative_error = abs(Tc - CRITICAL_TEMP_2D) / CRITICAL_TEMP_2D
        assert relative_error < 0.05

    def test_beta_nu_reasonable(self, fss_accurate):
        """Test β/ν is in reasonable range."""
        beta_nu, error = fss_accurate.extract_exponent_beta(CRITICAL_TEMP_2D)

        # Exact: β/ν = 1/8 = 0.125
        # Allow range 0.05 to 0.25 for finite-size effects
        assert 0.05 < beta_nu < 0.25

    def test_gamma_nu_reasonable(self, fss_accurate):
        """Test γ/ν is in reasonable range."""
        gamma_nu, error = fss_accurate.extract_exponent_gamma(CRITICAL_TEMP_2D)

        # Exact: γ/ν = 7/4 = 1.75
        # Allow range 1.2 to 2.3 for finite-size effects
        assert 1.2 < gamma_nu < 2.3
